"""Deliberate run retry contracts; all transports and credentials are synthetic."""
import importlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

import pytest

from hermes_herald import callback, config, ledger, tools


@pytest.fixture
def origin(tmp_path, monkeypatch):
    route = {"url": "http://target.invalid/p/reviewer", "api_key": "synthetic-token"}
    owner = {"session_key": "chat-1", "session_id": "session-1", "parent_session_id": "session-1"}
    monkeypatch.setattr(config, "get_ledger_file_path", lambda: tmp_path / "ledger.db")
    monkeypatch.setattr(config, "get_state_file_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(config, "get_active_profile_name", lambda: "origin")
    monkeypatch.setattr(tools, "_resolve_profile", lambda *a, **k: (route, None))
    monkeypatch.setattr(callback, "capture_session_routing", lambda *a: dict(owner))
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)
    monkeypatch.setattr(tools, "_get_json", lambda *a, **k: {
        "features": {"runs_idempotency": {"supported": True, "durable": True, "retention_seconds": 86400}}
    })
    listeners = []
    monkeypatch.setattr(callback, "start_listener", lambda **kw: listeners.append(kw))
    return route, owner, listeners, tmp_path


def call(**overrides):
    args = {"profile": "reviewer", "message": "Review this", "delivery": "none"}
    args.update(overrides)
    return json.loads(tools.handle_dispatch_agent(args))


def test_schema_exposes_optional_explicit_retry_key():
    params = tools.DISPATCH_AGENT_SCHEMA["parameters"]
    assert params["properties"]["idempotency_key"]["type"] == "string"
    assert "idempotency_key" not in params["required"]


def test_header_and_durable_identity_exist_before_transport(origin, monkeypatch):
    seen = []
    def transport(req, **kw):
        key = req.get_header("Idempotency-key")
        assert key == "first-key"
        with sqlite3.connect(origin[3] / "ledger.db") as db:
            assert db.execute("SELECT COUNT(*) FROM dispatch_identities").fetchone()[0] == 1
        seen.append(req.data)
        return BytesIO(b'{"run_id":"original-run"}')
    monkeypatch.setattr(tools, "urlopen", transport)
    result = call(idempotency_key="first-key")
    assert result["run_id"] == "original-run"
    assert result["idempotency_key"] == "first-key"
    assert len(seen) == 1


def test_accepted_lost_reply_replays_original_bytes_after_reconstruction(origin, monkeypatch):
    requests = []
    def transport(req, **kw):
        requests.append((req.get_header("Idempotency-key"), req.data))
        if len(requests) == 1:
            raise URLError("accepted then disconnected")
        assert requests[1] == requests[0]
        return BytesIO(b'{"run_id":"original-run"}')
    monkeypatch.setattr(tools, "urlopen", transport)
    lost = call(delivery="callback", max_hops=3)
    assert lost["status"] == "error"
    key = lost["idempotency_key"]
    assert "reuse" in lost["retry_instruction"].lower()
    # Rebuild the plugin module, dropping any in-memory reservations.
    importlib.reload(tools)
    import hermes_herald.dispatch_identity as identities
    importlib.reload(identities)
    monkeypatch.setattr(tools, "_resolve_profile", lambda *a, **k: (origin[0], None))
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)
    monkeypatch.setattr(tools, "_get_json", lambda *a, **k: {
        "features": {"runs_idempotency": {"supported": True, "durable": True, "retention_seconds": 86400}}
    })
    monkeypatch.setattr(tools, "urlopen", transport)
    replay = call(delivery="callback", max_hops=3, idempotency_key=key)
    again = call(delivery="callback", max_hops=3, idempotency_key=key)
    assert replay["run_id"] == again["run_id"] == "original-run"
    assert replay["edge_id"] == again["edge_id"]
    assert len(requests) == 2
    assert len(origin[2]) == 1
    rows = ledger.list_dispatches()
    assert len(rows) == 1
    assert rows[0]["origin_session_id"] == "chat-1"
    assert len(tools._load_state()["runs"]) == 1


def test_native_hermes_durable_store_replays_acceptance_after_both_stores_reopen(origin, monkeypatch):
    import hashlib

    native = pytest.importorskip("gateway.platforms.api_server_run_idempotency")
    server_path = origin[3] / "synthetic-target.db"
    outcomes = []
    def transport(req, **kw):
        # Same full-body fingerprint contract as the inspected Hermes handler.
        fingerprint = hashlib.sha256(json.dumps(
            {"body": json.loads(req.data), "gateway_session_key": ""},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()).hexdigest()
        store = native.RunIdempotencyStore(str(server_path))
        try:
            outcome, record = store.reserve(
                "synthetic-auth-profile-scope", req.get_header("Idempotency-key"),
                fingerprint, "native-original-run", {"status": "queued"},
            )
        finally:
            store.close()
        outcomes.append(outcome)
        if len(outcomes) == 1:
            raise URLError("accepted then disconnected")
        return BytesIO(json.dumps({"run_id": record["run_id"]}).encode())
    monkeypatch.setattr(tools, "urlopen", transport)
    first = call(idempotency_key="native-stable-key")
    assert first["status"] == "error"
    import hermes_herald.dispatch_identity as identities
    importlib.reload(identities)
    second = call(idempotency_key="native-stable-key")
    assert second["run_id"] == "native-original-run"
    assert outcomes == ["created", "reused"]


@pytest.mark.parametrize("change", [
    {"message": "Changed"}, {"instructions": "Changed"}, {"delivery": "callback"},
    {"max_hops": 4}, {"reasoning_effort": "high"}, {"trace_id": "other"},
    {"model": "another-alias"}, {"provider": "another-provider"},
])
def test_changed_request_fails_before_second_post(origin, monkeypatch, change):
    posts = []
    monkeypatch.setattr(tools, "_post_json", lambda *a, **k: posts.append(a) or {"run_id": "run-1"})
    assert call(idempotency_key="same")["run_id"] == "run-1"
    conflict = call(idempotency_key="same", **change)
    assert conflict["status"] == "error"
    assert conflict["error_code"] == "idempotency_conflict"
    assert len(posts) == 1


@pytest.mark.parametrize("boundary", ["url", "api_key", "session", "model"])
def test_changed_route_or_owner_fails_closed(origin, monkeypatch, boundary):
    posts = []
    monkeypatch.setattr(tools, "_post_json", lambda *a, **k: posts.append(a) or {"run_id": "run-1"})
    call(idempotency_key="same")
    if boundary == "session":
        origin[1]["parent_session_id"] = "new-durable-session"
    else:
        origin[0][boundary] = "different"
    result = call(idempotency_key="same")
    assert result["status"] == "error"
    assert result["error_code"] == "idempotency_conflict"
    assert len(posts) == 1


def test_same_message_without_key_is_intentional_new_work(origin, monkeypatch):
    keys = []
    def post(*a, **kw):
        keys.append(kw.get("idempotency_key"))
        return {"run_id": f"run-{len(keys)}"}
    monkeypatch.setattr(tools, "_post_json", post)
    first, second = call(), call()
    assert first["run_id"] != second["run_id"]
    assert keys[0] and keys[1] and keys[0] != keys[1]
    assert len(ledger.list_dispatches()) == 2


def test_concurrent_same_key_has_one_submitter(origin, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    posts = []
    def post(*a, **kw):
        posts.append(kw)
        entered.set()
        assert release.wait(10)
        return {"run_id": "run-1"}
    monkeypatch.setattr(tools, "_post_json", post)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call, idempotency_key="same")
        assert entered.wait(10)
        try:
            second = pool.submit(call, idempotency_key="same").result(timeout=10)
            assert second["error_code"] == "idempotency_in_progress"
        finally:
            release.set()
        assert first.result(timeout=10)["run_id"] == "run-1"
    assert call(idempotency_key="same")["run_id"] == "run-1"
    assert len(posts) == 1
    assert len(ledger.list_dispatches()) == 1


def test_expired_lease_loser_returns_handle_without_duplicate_local_effects(origin, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    posts = []
    def post(*a, **kw):
        posts.append((a[2], kw["idempotency_key"]))
        if len(posts) == 1:
            entered.set()
            assert release.wait(10)
        return {"run_id": "original-run"}
    monkeypatch.setattr(tools, "_post_json", post)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call, idempotency_key="same", delivery="callback", max_hops=3)
        assert entered.wait(10)
        try:
            with sqlite3.connect(origin[3] / "ledger.db") as db:
                db.execute("UPDATE dispatch_identities SET lease_until=0")
            winner = call(idempotency_key="same", delivery="callback", max_hops=3)
        finally:
            release.set()
        loser = first.result(timeout=10)
    assert winner["run_id"] == loser["run_id"] == "original-run"
    assert loser["error_code"] == "idempotency_in_progress"
    assert posts[0] == posts[1]
    assert len(ledger.list_dispatches()) == len(origin[2]) == 1
    assert len(tools._load_state()["runs"]) == 1


def test_409_is_distinct_and_never_rotates_key(origin, monkeypatch):
    requests = []
    def transport(req, **kw):
        requests.append(req)
        raise HTTPError(req.full_url, 409, "Conflict", {}, BytesIO(b'key conflict'))
    monkeypatch.setattr(tools, "urlopen", transport)
    first = call(idempotency_key="same")
    second = call(idempotency_key="same")
    assert first["error_code"] == second["error_code"] == "idempotency_server_conflict"
    assert first["idempotency_key"] == "same"
    assert len(requests) == 1


@pytest.mark.parametrize("capability", [{}, {"supported": True, "durable": False, "retention_seconds": 86400}])
def test_ambiguous_retry_without_durable_server_capability_is_refused(origin, monkeypatch, capability):
    posts = []
    monkeypatch.setattr(tools, "_get_json", lambda *a, **k: {"features": {"runs_idempotency": capability}})
    def post(*a, **kw):
        posts.append(kw)
        raise RuntimeError("lost response")
    monkeypatch.setattr(tools, "_post_json", post)
    first = call(idempotency_key="same")
    assert first["idempotency_key"] == "same"
    second = call(idempotency_key="same")
    assert second["error_code"] == "idempotency_unsafe_replay"
    assert len(posts) == 1


def test_expired_ambiguous_identity_is_not_reposted(origin, monkeypatch):
    posts = []
    def post(*a, **kw):
        posts.append(kw)
        raise RuntimeError("lost response")
    monkeypatch.setattr(tools, "_post_json", post)
    call(idempotency_key="same")
    with sqlite3.connect(origin[3] / "ledger.db") as db:
        db.execute("UPDATE dispatch_identities SET replay_until=0")
    result = call(idempotency_key="same")
    assert result["error_code"] == "idempotency_unsafe_replay"
    assert len(posts) == 1


@pytest.mark.parametrize("key", ["", " bad", "bad ", "line\nbreak", "é", "x" * 256, None, 42])
def test_invalid_keys_are_not_repaired_or_sent(origin, monkeypatch, key):
    posts = []
    monkeypatch.setattr(tools, "_post_json", lambda *a, **k: posts.append(a))
    assert call(idempotency_key=key)["status"] == "error"
    assert posts == []


def test_ledger_contains_no_bearer_credential(origin, monkeypatch):
    monkeypatch.setattr(tools, "_post_json", lambda *a, **k: {"run_id": "run-1"})
    call(idempotency_key="same")
    with sqlite3.connect(origin[3] / "ledger.db") as db:
        assert origin[0]["api_key"] not in "\n".join(db.iterdump())


def test_incomplete_response_returns_generated_key_for_safe_replay(origin, monkeypatch):
    def post(*a, **kw):
        raise IncompleteRead(b"partial accepted response")
    monkeypatch.setattr(tools, "_post_json", post)
    result = call()
    assert result["status"] == "error"
    assert result["idempotency_key"]
    assert "reuse" in result["retry_instruction"].lower()


def test_abandoned_submitter_requires_lease_then_reuses_original_request(origin, monkeypatch):
    class ProcessExit(BaseException):
        pass
    requests = []
    def post(*a, **kw):
        requests.append((a[2], kw["idempotency_key"]))
        if len(requests) == 1:
            raise ProcessExit()
        return {"run_id": "original-run"}
    monkeypatch.setattr(tools, "_post_json", post)
    with pytest.raises(ProcessExit):
        call(idempotency_key="same")
    assert call(idempotency_key="same")["error_code"] == "idempotency_in_progress"
    with sqlite3.connect(origin[3] / "ledger.db") as db:
        db.execute("UPDATE dispatch_identities SET lease_until=0")
    assert call(idempotency_key="same")["run_id"] == "original-run"
    assert requests[0] == requests[1]


def test_identity_storage_failure_prevents_post(origin, monkeypatch):
    posts = []
    monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
    def broken():
        raise sqlite3.OperationalError("synthetic disk failure")
    monkeypatch.setattr(ledger, "_connect", broken)
    monkeypatch.setattr(tools, "_post_json", lambda *a, **k: posts.append(a))
    result = call(idempotency_key="same")
    assert result["status"] == "error"
    assert result["idempotency_key"] == "same"
    assert posts == []


def test_known_accepted_handle_survives_identity_completion_storage_failure(origin, monkeypatch):
    from hermes_herald.dispatch_identity import DispatchIdentity
    monkeypatch.setattr(tools, "_post_json", lambda *a, **k: {"run_id": "accepted-run"})
    def broken(*a):
        raise sqlite3.OperationalError("synthetic disk failure")
    monkeypatch.setattr(DispatchIdentity, "_store_response", broken)
    result = call(idempotency_key="same")
    assert result["run_id"] == "accepted-run"
    assert result["idempotency_key"] == "same"
    assert "warning" in result
    assert ledger.list_dispatches() == []


def test_changed_body_after_ambiguous_acceptance_never_posts(origin, monkeypatch):
    posts = []
    def post(*a, **kw):
        posts.append(kw)
        raise RuntimeError("accepted response lost")
    monkeypatch.setattr(tools, "_post_json", post)
    call(idempotency_key="same")
    assert call(idempotency_key="same", message="different")["error_code"] == "idempotency_conflict"
    assert len(posts) == 1


def test_lost_current_capability_refuses_previously_durable_replay(origin, monkeypatch):
    posts = []
    def post(*a, **kw):
        posts.append(kw)
        raise RuntimeError("accepted response lost")
    monkeypatch.setattr(tools, "_post_json", post)
    call(idempotency_key="same")
    monkeypatch.setattr(tools, "_get_json", lambda *a, **kw: {})
    assert call(idempotency_key="same")["error_code"] == "idempotency_unsafe_replay"
    assert len(posts) == 1


def test_known_success_never_reposts_even_after_retention_expires(origin, monkeypatch):
    posts = []
    monkeypatch.setattr(tools, "_post_json", lambda *a, **kw: posts.append(kw) or {"run_id": "original-run"})
    first = call(idempotency_key="same")
    with sqlite3.connect(origin[3] / "ledger.db") as db:
        db.execute("UPDATE dispatch_identities SET replay_until=0")
    monkeypatch.setattr(tools, "_get_json", lambda *a, **kw: {})
    second = call(idempotency_key="same")
    assert second["run_id"] == first["run_id"]
    assert len(posts) == 1


def test_known_callback_acceptance_survives_loss_of_delivery_support(origin, monkeypatch):
    posts = []
    monkeypatch.setattr(tools, "_post_json", lambda *a, **kw: posts.append(kw) or {"run_id": "original-run"})
    first = call(idempotency_key="same", delivery="callback")
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: False)
    retry = call(idempotency_key="same", delivery="callback")
    assert retry.get("run_id") == first["run_id"] == "original-run"
    assert retry["idempotency_replayed"] is True
    assert len(posts) == len(origin[2]) == len(ledger.list_dispatches()) == 1


def test_unknown_server_still_allows_initial_no_key_dispatch(origin, monkeypatch):
    monkeypatch.setattr(tools, "_get_json", lambda *a, **kw: {})
    monkeypatch.setattr(tools, "_post_json", lambda *a, **kw: {"run_id": "original-run"})
    result = call()
    assert result["status"] == "dispatched"
    assert result["run_id"] == "original-run"
    assert result["idempotency_key"]
    assert "did not advertise durable" in result["warning"]


@pytest.mark.parametrize("ambiguous", [False, True])
def test_missing_callback_support_never_submits_or_recommends_duplicate_work(origin, monkeypatch, ambiguous):
    posts = []
    def post(*a, **kw):
        posts.append(kw)
        raise RuntimeError("accepted response lost")
    monkeypatch.setattr(tools, "_post_json", post)
    if ambiguous:
        call(idempotency_key="same", delivery="callback")
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: False)
    result = call(idempotency_key="same", delivery="callback")
    assert result["status"] == "error"
    assert len(posts) == int(ambiguous)
    assert origin[2] == []
    if ambiguous:
        assert "acceptance is unknown" in result["error"]
        assert "dispatch_chat" not in result["error"]
    else:
        assert "No remote run was started" in result["error"]


@pytest.mark.parametrize("response", [{}, {"run_id": None}, {"run_id": 42}, []])
def test_malformed_acceptance_returns_retry_key(origin, monkeypatch, response):
    monkeypatch.setattr(tools, "_post_json", lambda *a, **kw: response)
    result = call()
    assert result["status"] == "error"
    assert result["error_code"] == "idempotency_ambiguous_response"
    assert result["idempotency_key"]
    assert "reuse" in result["retry_instruction"].lower()

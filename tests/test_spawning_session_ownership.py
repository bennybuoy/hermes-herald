"""Herald completions obey the host's durable spawning-session boundary.

All I/O is synthetic: real gateway session rows/reset and completion policy,
fake remote SSE bytes and fake child inference. No listener or service is opened.
"""

import asyncio
import json
import os
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_herald import callback, tools
from gateway.config import GatewayConfig, Platform
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session import SessionSource, SessionStore
from gateway.session_context import clear_session_vars, set_session_vars, scoped_current_session_id
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_state import AsyncSessionDB


@pytest.fixture(autouse=True)
def source_locality():
    plugin = Path(__file__).resolve().parents[1]
    upstream = Path(os.environ["HERMES_SOURCE_DIR"]).resolve()
    assert Path(callback.__file__).resolve() == plugin / "callback.py"
    assert Path(tools.__file__).resolve() == plugin / "tools.py"
    import gateway.run_notifications as notifications
    assert Path(notifications.__file__).resolve() == upstream / "gateway/run_notifications.py"
    assert Path.cwd().resolve() == upstream


@pytest.fixture
def spawning_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entry = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic-chat", chat_type="dm")
    )
    host = GatewayNotificationsMixin()
    host.session_store = store
    host._session_db = AsyncSessionDB(store._db)
    # Only transport readiness is stubbed; host session classification and the
    # durable claim API use the real disposable database.
    host._completion_delivery_ready = AsyncMock(return_value=True)
    host._current_session_run_generation = lambda key: 0
    monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
    monkeypatch.setattr(tools, "_save_state", lambda state: None)
    monkeypatch.setattr(tools, "_update_run_status", lambda *a, **kw: None)
    monkeypatch.setattr(callback, "_delivery_routes", {})
    monkeypatch.setattr(callback, "_listeners", {})
    monkeypatch.setattr(callback, "_cancel_flags", {})
    try:
        yield store, entry, host
    finally:
        store._db.close()
        reset_hermes_home_override(token)


def assert_host_ownership(event, store, spawned, host, replaced):
    current = store._entries[spawned.session_key]
    if replaced:
        assert current.session_id != spawned.session_id
        assert store._db.get_session(spawned.session_id)["end_reason"] == "session_reset"
        assert not store._db.get_session(current.session_id)["ended_at"]
    else:
        assert current.session_id == spawned.session_id
    claim = asyncio.run(host._preflight_completion_delivery(event))
    assert claim.proceed is (not replaced), "late S1 completion must not enter S2 after /new"
    assert event.get("parent_session_id") == spawned.session_id
    assert event["parent_session_id"] != spawned.session_key
    resolved = asyncio.run(host._resolve_async_delegation_session(current, event["parent_session_id"]))
    assert resolved is (None if replaced else current)
    assert store._entries[spawned.session_key] is current


class DeferredThread:
    """Hold the actual background target until after the simulated /new."""

    def __init__(self, *, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        pass

    def finish(self):
        self.target(*self.args)


class SyntheticSSE:
    def __init__(self, status):
        self.payload = b"data: " + json.dumps({
            "event": "run." + status, "output": "synthetic result",
            "error": "synthetic failure", "usage": {},
        }).encode() + b"\n\n"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, n):
        payload, self.payload = self.payload, b""
        return payload


@pytest.mark.parametrize("surface", ["gateway", "tui"])
@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("replaced", [False, True], ids=["active-S1", "new-S2"])
def test_remote_callback_pins_spawning_session(spawning_session, monkeypatch, surface, status, replaced):
    store, spawned, host = spawning_session
    parent = SimpleNamespace(session_id=spawned.session_id) if surface == "tui" else None
    tokens = set_session_vars(
        source=surface, session_id=spawned.session_id,
        session_key=spawned.session_key, ui_session_id="ui-tab" if parent else "",
    )
    monkeypatch.setattr(callback, "_get_current_session_key", lambda: spawned.session_key)
    completion_queue = queue.Queue()
    import tools.process_registry as registry
    monkeypatch.setattr(registry, "process_registry", SimpleNamespace(completion_queue=completion_queue))
    real_thread = threading.Thread

    def make_listener(**kwargs):
        if kwargs.get("name", "").startswith("dispatch-sse-"):
            return DeferredThread(**kwargs)
        return real_thread(**kwargs)

    monkeypatch.setattr(threading, "Thread", make_listener)
    monkeypatch.setattr(callback, "urlopen", lambda *a, **kw: SyntheticSSE(status))
    try:
        callback.start_listener("synthetic-run", "worker", "https://synthetic.invalid", "synthetic", "goal", parent_agent=parent)
        monkeypatch.setattr(threading, "Thread", real_thread)
        worker = callback._listeners["synthetic-run"]
        if replaced:
            replacement = store.reset_session(spawned.session_key)
            if parent:
                parent.session_id = replacement.session_id
        # Completion runs with unrelated ambient identity. Ownership is a
        # dispatch-time snapshot, not a late read of the parent/UI/chat route.
        with scoped_current_session_id("unrelated-background-session"):
            worker.finish()
        event = completion_queue.get_nowait()
        assert completion_queue.empty()
        assert event["status"] == status
        assert event["session_id"] == spawned.session_id
        assert event["session_key"] == (spawned.session_id if parent else spawned.session_key)
        assert event["origin_ui_session_id"] == ("ui-tab" if parent else "")
        assert_host_ownership(event, store, spawned, host, replaced)
    finally:
        clear_session_vars(tokens)


@pytest.mark.parametrize("outcome", ["success", "returned-failure", "exception"])
@pytest.mark.parametrize("replaced", [False, True], ids=["active-S1", "new-S2"])
def test_local_completion_pins_spawning_session(spawning_session, monkeypatch, outcome, replaced):
    store, spawned, host = spawning_session
    parent = SimpleNamespace(session_id=spawned.session_id, model="synthetic-model")
    tokens = set_session_vars(source="gateway", session_id=spawned.session_id, session_key=spawned.session_key)
    monkeypatch.setattr(callback, "_get_current_session_key", lambda: spawned.session_key)
    monkeypatch.setattr(tools, "_resolve_model_creds", lambda *a: {
        "model": "synthetic-model", "provider": None, "base_url": None,
        "api_key": None, "api_mode": None,
    })
    import tools.delegate_tool as delegation
    import tools.process_registry as registry
    completion_queue = queue.Queue()
    monkeypatch.setattr(registry, "process_registry", SimpleNamespace(completion_queue=completion_queue))
    monkeypatch.setattr(delegation, "_load_config", lambda: {})
    monkeypatch.setattr(delegation, "_get_child_timeout", lambda: None)
    monkeypatch.setattr(delegation, "_run_single_child", lambda **kw: {})
    child = SimpleNamespace(session_id="synthetic-child-session")

    def build_child(**kwargs):
        # Real child construction changes HERMES_SESSION_ID in the calling
        # ContextVar. Simulate that documented side effect without inference.
        import gateway.session_context as context
        context._SESSION_ID.set(child.session_id)
        return child

    monkeypatch.setattr(delegation, "_build_child_agent", build_child)

    def run_child(**kwargs):
        if outcome == "exception":
            raise RuntimeError("synthetic child failure")
        if outcome == "returned-failure":
            return {"status": "failed", "error": "synthetic child failure"}
        return {"status": "completed", "summary": "synthetic result"}

    monkeypatch.setattr(tools, "_run_child_with_timeout_policy", run_child)
    workers = []
    real_thread = threading.Thread

    def make_thread(**kwargs):
        if not kwargs.get("name", "").startswith("delegate-subagent-"):
            return real_thread(**kwargs)
        worker = DeferredThread(**kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(threading, "Thread", make_thread)
    try:
        response = json.loads(tools.handle_delegate_subagent({"goal": "synthetic goal"}, parent_agent=parent))
        monkeypatch.setattr(threading, "Thread", real_thread)
        assert response["status"] == "dispatched"
        assert len(workers) == 1
        if replaced:
            parent.session_id = store.reset_session(spawned.session_key).session_id
        workers[0].finish()
        event = completion_queue.get_nowait()
        assert completion_queue.empty()
        assert event["status"] == ("completed" if outcome == "success" else "failed")
        assert event["session_key"] == spawned.session_key
        assert_host_ownership(event, store, spawned, host, replaced)
        assert event["session_id"] == spawned.session_id
    finally:
        clear_session_vars(tokens)


def test_direct_remote_event_uses_session_id_not_chat_key(spawning_session):
    store, spawned, host = spawning_session
    store.reset_session(spawned.session_key)
    event = callback._build_delegation_event(
        "direct-run", "worker", "result", {}, "goal", spawned.session_id, spawned.session_key,
    )
    assert_host_ownership(event, store, spawned, host, True)


@pytest.mark.parametrize("parent_id,context_id,expected", [
    ("durable-parent", "ambient-child", "durable-parent"),
    ("", "durable-context", "durable-context"),
    ("", "", ""),
])
def test_session_ownership_is_independent_of_chat_and_ui_route(monkeypatch, parent_id, context_id, expected):
    values = {
        "HERMES_SESSION_SOURCE": "gateway",
        "HERMES_SESSION_ID": context_id,
        "HERMES_SESSION_KEY": "chat-key",
        "HERMES_UI_SESSION_ID": "ui-tab",
    }
    monkeypatch.setattr(callback, "_get_session_env", lambda name, default="": values.get(name, default))
    monkeypatch.setattr(callback, "_get_current_session_key", lambda: "chat-key")
    route = callback.capture_session_routing(SimpleNamespace(session_id=parent_id))
    assert route["parent_session_id"] == expected
    assert route["session_id"] == context_id
    assert route["session_key"] == "chat-key"
    assert route["origin_ui_session_id"] == "ui-tab"

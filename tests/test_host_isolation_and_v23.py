"""Tests for plugins.isolation: host hardening and Hermes v0.23.3+ /v1/runs compatibility."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_herald import callback, config, ledger, tools


class _OpaqueAgent:
    """Mimics hermes_cli.plugin_host_wire.Opaque across the host boundary."""

    def __getattr__(self, name: str):
        raise AttributeError(
            f"AIAgent.{name} cannot be accessed across the plugin host boundary"
        )


def _state_path() -> str:
    return os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")


@pytest.fixture(autouse=True)
def _clean_env_and_state(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_UI_SESSION_ID", raising=False)
    monkeypatch.setattr(config, "get_ledger_file_path", lambda: tmp_path / "herald.db")
    monkeypatch.setattr(config, "get_state_file_path", lambda: tmp_path / "runs.json")
    monkeypatch.setattr(config, "get_active_profile_name", lambda: "origin")
    with callback._pending_approvals_lock:
        callback._pending_approvals.clear()
        callback._pending_approval_queues.clear()
        callback._local_approval_responses.clear()
        callback._approval_notice_events.clear()
        callback._approval_notice_ids_by_run.clear()
    with callback._delivery_gate_lock:
        callback._delivery_routes.clear()
    with callback._listeners_lock:
        callback._listeners.clear()
    callback._cancel_flags.clear()
    with callback._session_ids_lock:
        callback._session_ids.clear()
    yield
    with callback._pending_approvals_lock:
        callback._pending_approvals.clear()
        callback._pending_approval_queues.clear()
        callback._local_approval_responses.clear()
        callback._approval_notice_events.clear()
        callback._approval_notice_ids_by_run.clear()
    with callback._delivery_gate_lock:
        callback._delivery_routes.clear()
    callback._cancel_flags.clear()


def _seed_pending(
    run_id: str,
    profile: str = "marie",
    *,
    request_id: str = "",
    execution_generation=None,
) -> dict:
    approval_data = {
        "run_id": run_id,
        "profile": profile,
        "command": "rm -rf /tmp/demo",
        "description": "needs approval",
        "pattern_key": "rm -rf:*",
        "pattern_keys": ["rm -rf:*"],
        "allow_permanent": True,
        "choices": ["once", "session", "always", "deny"],
        "timestamp": 1700000000,
        "requested_at": 1700000000.0,
        "delivery_id": f"approval-{run_id}",
        "origin_session_id": "origin-session",
        "origin_session_key": "origin-key",
        "origin_ui_session_id": "origin-ui",
    }
    if request_id:
        approval_data["request_id"] = request_id
    if execution_generation is not None:
        approval_data["execution_generation"] = execution_generation
    callback._set_pending_approval(run_id, approval_data)
    tools._persist_run(run_id, profile, "preview", model="glm-5.2")
    tools._update_pending_approval(run_id, approval_data)
    return approval_data


class _FakeSSEResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    def read1(self, _n: int = 4096) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    read = read1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# Host isolation: async delivery, approve_dispatch, session routing fallback
# ---------------------------------------------------------------------------


class TestHostIsolation:
    def test_async_delivery_supported_returns_false_immediately_in_host_process(
        self, monkeypatch
    ):
        fake_mod = types.ModuleType("gateway.session_context")
        called = []

        def _should_not_run():
            called.append(True)
            return True

        fake_mod.async_delivery_supported = _should_not_run
        monkeypatch.setitem(sys.modules, "gateway.session_context", fake_mod)
        monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")

        assert tools._async_delivery_supported() is False
        assert called == []

    def test_dispatch_agent_callback_fails_closed_before_post_in_host_mode(
        self, monkeypatch
    ):
        monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
        route = {"url": "http://target.invalid/p/marie", "api_key": "synthetic-key"}
        monkeypatch.setattr(tools, "_resolve_profile", lambda *a, **k: (route, None))

        with patch.object(tools, "_post_json") as post_mock, \
             patch.object(tools, "_get_json") as get_mock:
            result = json.loads(
                tools.handle_dispatch_agent(
                    {"profile": "marie", "message": "Do work", "delivery": "callback"},
                    parent_agent=_OpaqueAgent(),
                    session_id="host-sess-1",
                )
            )

        assert result["status"] == "error"
        assert "plugins.isolation: host" in result["error"]
        assert "delivery='none'" in result["error"]
        assert "check_dispatch" in result["error"]
        post_mock.assert_not_called()
        get_mock.assert_not_called()

    def test_host_mode_dispatch_none_preserves_session_ownership_and_ledger_origin(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
        monkeypatch.setattr(callback, "_get_session_env", lambda name, default="": "")
        route = {"url": "http://target.invalid/p/marie", "api_key": "synthetic-key"}
        monkeypatch.setattr(tools, "_resolve_profile", lambda *a, **k: (route, None))
        monkeypatch.setattr(
            tools,
            "_get_json",
            lambda *a, **k: {
                "features": {
                    "runs_idempotency": {
                        "supported": True,
                        "durable": True,
                        "retention_seconds": 86400,
                    }
                }
            },
        )
        monkeypatch.setattr(
            tools, "_post_json", lambda *a, **k: {"run_id": "run-host-1"}
        )

        # Verify against real plugin_host_wire.Opaque when upstream is present.
        opaque_parent = _OpaqueAgent()
        try:
            from hermes_cli.plugin_host_wire import Opaque

            opaque_parent = Opaque("AIAgent")
        except Exception:
            pass

        result = json.loads(
            tools.handle_dispatch_agent(
                {
                    "profile": "marie",
                    "message": "Run offline task",
                    "delivery": "none",
                    "idempotency_key": "host-idem-key-1",
                },
                parent_agent=opaque_parent,
                session_id="host-sess-A",
            )
        )
        assert result["status"] == "dispatched"
        assert result["run_id"] == "run-host-1"

        rows = ledger.list_dispatches()
        assert len(rows) == 1
        assert rows[0]["origin_session_id"] == "host-sess-A"

        state = tools._load_state()
        assert state["runs"][0]["session_id"] == "host-sess-A"

        with sqlite3.connect(tmp_path / "herald.db") as db:
            owner_json = db.execute(
                "SELECT owner_json FROM dispatch_identities WHERE idempotency_key=?",
                ("host-idem-key-1",),
            ).fetchone()[0]
        owner = json.loads(owner_json)
        assert owner["session"] == "host-sess-A"
        assert owner["session_key"] == "host-sess-A"

        # A different host-mode session reusing the same idempotency_key must
        # be rejected with idempotency_conflict.
        conflict = json.loads(
            tools.handle_dispatch_agent(
                {
                    "profile": "marie",
                    "message": "Run offline task",
                    "delivery": "none",
                    "idempotency_key": "host-idem-key-1",
                },
                parent_agent=opaque_parent,
                session_id="host-sess-B",
            )
        )
        assert conflict["status"] == "error"
        assert conflict["error_code"] == "idempotency_conflict"

    def test_host_mode_dispatch_chat_records_origin_session_id_with_opaque_parent(
        self, monkeypatch
    ):
        monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
        monkeypatch.setattr(callback, "_get_session_env", lambda name, default="": "")
        route = {"url": "http://target.invalid/p/marie", "api_key": "synthetic-key"}
        monkeypatch.setattr(tools, "_resolve_profile", lambda *a, **k: (route, None))
        monkeypatch.setattr(
            tools,
            "_post_json",
            lambda *a, **k: {"session": {"id": "remote-chat-1"}},
        )
        monkeypatch.setattr(
            tools,
            "_post_streaming_chat",
            lambda *a, **k: {
                "reply": "hello back",
                "session_id": "remote-chat-1",
                "usage": {"total_tokens": 12},
                "model": "gpt-5",
            },
        )

        result = json.loads(
            tools.handle_dispatch_chat(
                {"profile": "marie", "message": "Say hi"},
                parent_agent=_OpaqueAgent(),
                session_id="host-chat-origin",
            )
        )
        assert result["status"] == "completed"
        assert result["reply"] == "hello back"

        rows = ledger.list_dispatches(dispatch_type="chat")
        assert len(rows) == 1
        assert rows[0]["origin_session_id"] == "host-chat-origin"

    def test_capture_session_routing_fallback_and_precedence(self, monkeypatch):
        monkeypatch.setattr(callback, "_get_session_env", lambda name, default="": "")

        # Opaque parent + fallback session_id fills all session fields
        route = callback.capture_session_routing(
            _OpaqueAgent(), session_id="fallback-42"
        )
        assert route["session_id"] == "fallback-42"
        assert route["parent_session_id"] == "fallback-42"
        assert route["session_key"] == "fallback-42"

        # Live parent_agent.session_id takes precedence over fallback
        live_parent = types.SimpleNamespace(session_id="live-parent-99")
        live_route = callback.capture_session_routing(
            live_parent, session_id="fallback-42"
        )
        assert live_route["parent_session_id"] == "live-parent-99"
        assert live_route["session_key"] == "live-parent-99"

        # Empty fallback preserves empty routing
        empty_route = callback.capture_session_routing(None)
        assert empty_route["session_id"] == ""
        assert empty_route["parent_session_id"] == ""
        assert empty_route["session_key"] == ""

        # In host mode, per-call session_id overrides stale inherited env vars
        monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
        monkeypatch.setattr(
            callback,
            "_get_session_env",
            lambda name, default="": {
                "HERMES_SESSION_ID": "stale-boot-session",
                "HERMES_SESSION_KEY": "stale-boot-key",
                "HERMES_UI_SESSION_ID": "stale-boot-ui",
            }.get(name, default),
        )
        host_route = callback.capture_session_routing(
            _OpaqueAgent(), session_id="fresh-host-session"
        )
        assert host_route == {
            "session_id": "fresh-host-session",
            "parent_session_id": "fresh-host-session",
            "session_key": "fresh-host-session",
            "origin_ui_session_id": "",
        }

    def test_approve_dispatch_fails_closed_immediately_in_host_mode(
        self, monkeypatch
    ):
        monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
        _seed_pending("run-host-approve", "marie")

        with patch.object(tools, "_resolve_profile") as resolve_mock, \
             patch.object(tools, "_get_json") as get_mock, \
             patch.object(tools, "_request_dispatch_approval_consent") as consent_mock, \
             patch.object(tools, "_post_json") as post_mock:
            result = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": "run-host-approve",
                        "profile": "marie",
                        "choice": "deny",
                        "approval_id": "approval-run-host-approve",
                    },
                    parent_agent=_OpaqueAgent(),
                    session_id="origin-session",
                )
            )

        assert result["status"] == "error"
        assert "plugins.isolation: host" in result["error"]
        resolve_mock.assert_not_called()
        get_mock.assert_not_called()
        consent_mock.assert_not_called()
        post_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Hermes v0.23.3+ /v1/runs session_authority approval compatibility
# ---------------------------------------------------------------------------


class TestV23AuthorityApprovals:
    @pytest.fixture(autouse=True)
    def _in_process_owner(self, monkeypatch):
        monkeypatch.setattr(
            tools, "_request_dispatch_approval_consent", lambda **kwargs: True
        )
        monkeypatch.setattr(
            callback,
            "capture_session_routing",
            lambda parent_agent=None, session_id="": {
                "session_id": "origin-session",
                "parent_session_id": "origin-session",
                "session_key": "origin-key",
                "origin_ui_session_id": "origin-ui",
            },
        )
        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile, **kw: (
                {"url": "http://target.invalid", "api_key": "synthetic-key"},
                None,
            ),
        )

    def test_sse_approval_request_captures_request_id_and_execution_generation(
        self, monkeypatch
    ):
        run_id = "run-sse-auth-1"
        payload = {
            "event": "approval.request",
            "command": "rm -rf /tmp/x",
            "description": "dangerous",
            "request_id": "apr_target_001",
            "execution_generation": 7,
            "choices": ["once", "deny"],
        }
        stream_bytes = (
            f"data: {json.dumps(payload)}\n\n"
            'data: {"event":"run.completed","output":"ok"}\n\n'
        ).encode("utf-8")

        with callback._delivery_gate_lock:
            callback._delivery_routes[run_id] = {
                "origin_ui_session_id": "origin-ui",
                "parent_session_id": "origin-session",
            }
        delivered = []
        monkeypatch.setattr(
            callback,
            "_deliver_approval_required",
            lambda approval_data, sid, skey: delivered.append(dict(approval_data)),
        )
        monkeypatch.setattr(
            callback, "urlopen", lambda req, timeout=None: _FakeSSEResponse([stream_bytes])
        )

        outcome, _evt = callback._read_sse_stream(
            run_id,
            "marie",
            f"http://target.invalid/v1/runs/{run_id}/events",
            "synthetic-key",
            "preview",
            "origin-session",
            "origin-key",
            "",
            "",
            0.0,
        )
        assert outcome == "terminal"
        assert len(delivered) == 1
        assert delivered[0]["request_id"] == "apr_target_001"
        assert delivered[0]["execution_generation"] == 7

    def test_sse_approval_request_ignores_bool_generation_and_accepts_prompt_id(
        self, monkeypatch
    ):
        run_id = "run-sse-auth-2"
        payload = {
            "event": "approval.request",
            "command": "rm -rf /tmp/y",
            "description": "dangerous",
            "prompt_id": "apr_from_prompt_id",
            "execution_generation": True,
        }
        stream_bytes = (
            f"data: {json.dumps(payload)}\n\n"
            'data: {"event":"run.completed","output":"ok"}\n\n'
        ).encode("utf-8")

        delivered = []
        monkeypatch.setattr(
            callback,
            "_deliver_approval_required",
            lambda approval_data, sid, skey: delivered.append(dict(approval_data)),
        )
        monkeypatch.setattr(
            callback, "urlopen", lambda req, timeout=None: _FakeSSEResponse([stream_bytes])
        )

        callback._read_sse_stream(
            run_id,
            "marie",
            f"http://target.invalid/v1/runs/{run_id}/events",
            "synthetic-key",
            "preview",
            "origin-session",
            "origin-key",
            "",
            "",
            0.0,
        )
        assert len(delivered) == 1
        assert delivered[0]["request_id"] == "apr_from_prompt_id"
        assert "execution_generation" not in delivered[0]

    def test_authority_target_posts_exact_three_key_deny_body(self):
        run_id = "run-auth-deny-1"
        _seed_pending(
            run_id,
            "marie",
            request_id="apr_01",
            execution_generation=4,
        )
        status_payload = {
            "status": "waiting_for_approval",
            "execution_generation": 4,
            "pending_controls": [
                {"prompt_id": "apr_01", "kind": "approval", "command": "rm -rf /tmp/demo"}
            ],
        }

        with patch.object(tools, "_get_json", return_value=status_payload), \
             patch.object(tools, "_post_json", return_value={"status": "running"}) as post_mock:
            result = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": run_id,
                        "profile": "marie",
                        "choice": "deny",
                        "approval_id": f"approval-{run_id}",
                    }
                )
            )

        assert result["status"] == "running"
        assert result["choice"] == "deny"
        posted_body = post_mock.call_args.args[2]
        assert posted_body == {
            "request_id": "apr_01",
            "execution_generation": 4,
            "choice": "deny",
        }
        assert set(posted_body.keys()) == {
            "request_id",
            "execution_generation",
            "choice",
        }

    def test_authority_target_reads_generation_and_single_prompt_from_status(self):
        run_id = "run-auth-status-1"
        # SSE notice had neither execution_generation nor request_id; target
        # GET /v1/runs/{id} provides execution_generation + single approval prompt.
        _seed_pending(run_id, "marie")
        status_payload = {
            "status": "waiting_for_approval",
            "execution_generation": 9,
            "pending_controls": [
                {"prompt_id": "apr_only", "kind": "approval"},
                {"prompt_id": "clar_1", "kind": "clarification"},
            ],
        }

        with patch.object(tools, "_get_json", return_value=status_payload), \
             patch.object(tools, "_post_json", return_value={"status": "running"}) as post_mock:
            result = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": run_id,
                        "profile": "marie",
                        "choice": "deny",
                        "approval_id": f"approval-{run_id}",
                    }
                )
            )

        assert result["status"] == "running"
        assert post_mock.call_args.args[2] == {
            "request_id": "apr_only",
            "execution_generation": 9,
            "choice": "deny",
        }

    def test_authority_target_refuses_resolve_all_before_consent_and_post(self):
        run_id = "run-auth-bulk"
        _seed_pending(run_id, "marie", request_id="apr_01", execution_generation=2)
        status_payload = {
            "status": "waiting_for_approval",
            "execution_generation": 2,
            "pending_controls": [{"prompt_id": "apr_01", "kind": "approval"}],
        }

        with patch.object(tools, "_get_json", return_value=status_payload), \
             patch.object(tools, "_request_dispatch_approval_consent") as consent_mock, \
             patch.object(tools, "_post_json") as post_mock:
            result = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": run_id,
                        "profile": "marie",
                        "choice": "deny",
                        "resolve_all": True,
                        "approval_id": f"approval-{run_id}",
                    }
                )
            )

        assert result["status"] == "error"
        assert "resolve_all=true is not supported" in result["error"]
        consent_mock.assert_not_called()
        post_mock.assert_not_called()

    def test_authority_target_refuses_generation_mismatch_and_ambiguous_prompts(self):
        run_id = "run-auth-mismatch"
        _seed_pending(run_id, "marie", request_id="apr_01", execution_generation=2)

        with patch.object(
            tools,
            "_get_json",
            return_value={"status": "waiting_for_approval", "execution_generation": 3},
        ), patch.object(tools, "_post_json") as post_mock:
            mismatch = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": run_id,
                        "profile": "marie",
                        "choice": "deny",
                        "approval_id": f"approval-{run_id}",
                    }
                )
            )
        assert mismatch["status"] == "error"
        assert "generation mismatch" in mismatch["error"].lower()
        post_mock.assert_not_called()

        run_id2 = "run-auth-ambig"
        _seed_pending(run_id2, "marie")
        with patch.object(
            tools,
            "_get_json",
            return_value={
                "status": "waiting_for_approval",
                "execution_generation": 5,
                "pending_controls": [
                    {"prompt_id": "apr_a", "kind": "approval"},
                    {"prompt_id": "apr_b", "kind": "approval"},
                ],
            },
        ), patch.object(tools, "_post_json") as post_mock:
            ambig = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": run_id2,
                        "profile": "marie",
                        "choice": "deny",
                        "approval_id": f"approval-{run_id2}",
                    }
                )
            )
        assert ambig["status"] == "error"
        assert "multiple pending approval prompts" in ambig["error"]
        post_mock.assert_not_called()

        # Empty pending_controls list must also fail closed even when notice
        # carried a request_id (the prompt is no longer pending on the target).
        run_id3 = "run-auth-gone"
        _seed_pending(run_id3, "marie", request_id="apr_stale", execution_generation=5)
        with patch.object(
            tools,
            "_get_json",
            return_value={
                "status": "waiting_for_approval",
                "execution_generation": 5,
                "pending_controls": [],
            },
        ), patch.object(tools, "_post_json") as post_mock:
            gone = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": run_id3,
                        "profile": "marie",
                        "choice": "deny",
                        "approval_id": f"approval-{run_id3}",
                    }
                )
            )
        assert gone["status"] == "error"
        assert "no longer pending" in gone["error"]
        post_mock.assert_not_called()

    def test_legacy_target_preserves_exact_choice_and_all_body(self):
        run_id = "run-legacy-deny"
        _seed_pending(run_id, "marie", request_id="apr_legacy_ignored")

        with patch.object(
            tools, "_get_json", return_value={"status": "waiting_for_approval"}
        ), patch.object(
            tools, "_post_json", return_value={"resolved": 1, "status": "running"}
        ) as post_mock:
            result = json.loads(
                tools.handle_approve_dispatch(
                    {
                        "run_id": run_id,
                        "profile": "marie",
                        "choice": "deny",
                        "approval_id": f"approval-{run_id}",
                    }
                )
            )

        assert result["status"] == "running"
        assert post_mock.call_args.args[2] == {"choice": "deny", "all": False}


# ---------------------------------------------------------------------------
# Hermes v0.23.3+ status="unknown" handling
# ---------------------------------------------------------------------------


class TestV23UnknownRunStatus:
    def test_check_run_status_returns_unknown_event_immediately(self, monkeypatch):
        body = json.dumps({"status": "unknown", "duration_seconds": 12.5}).encode("utf-8")

        class _Resp:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(callback, "urlopen", lambda req, timeout=None: _Resp())
        evt = callback._check_run_status(
            "run-unk-1",
            "marie",
            "http://target.invalid",
            "synthetic-key",
            "task",
            "sess-1",
            "key-1",
            "",
            "",
        )
        assert evt is not None
        assert evt["status"] == "unknown"
        assert "resolve-unknown" in evt["error"]

    def test_poll_run_status_exits_immediately_on_unknown(self, monkeypatch):
        calls = []

        class _Resp:
            def read(self):
                return json.dumps({"status": "unknown"}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_open(req, timeout=None):
            calls.append(req.full_url)
            return _Resp()

        monkeypatch.setattr(callback, "urlopen", _fake_open)
        monkeypatch.setattr(callback.time, "sleep", lambda _s: (_ for _ in ()).throw(AssertionError("should not sleep on unknown")))

        evt = callback._poll_run_status(
            "run-unk-2",
            "marie",
            "http://target.invalid",
            "synthetic-key",
            "task",
            "sess-1",
            "key-1",
            "",
            "",
            0.0,
        )
        assert evt is not None
        assert evt["status"] == "unknown"
        assert "resolve-unknown" in evt["error"]
        assert len(calls) == 1

    def test_collect_dispatches_classifies_unknown_distinctly_from_running(
        self, monkeypatch
    ):
        route = {"url": "http://target.invalid", "api_key": "synthetic-key"}
        monkeypatch.setattr(tools, "_resolve_profile", lambda p, **k: (route, None))

        statuses = {
            "r-done": {"run_id": "r-done", "status": "completed", "output": "ok"},
            "r-run": {"run_id": "r-run", "status": "running"},
            "r-unk": {"run_id": "r-unk", "status": "unknown"},
            "r-fail": {"run_id": "r-fail", "status": "failed", "error": "boom"},
        }

        def _fake_get(url, api_key, timeout=10.0):
            run_id = url.rsplit("/", 1)[-1]
            return dict(statuses[run_id])

        monkeypatch.setattr(tools, "_get_json", _fake_get)

        result = json.loads(
            tools.handle_collect_dispatches(
                {
                    "run_ids": [
                        {"run_id": "r-done", "profile": "marie"},
                        {"run_id": "r-run", "profile": "marie"},
                        {"run_id": "r-unk", "profile": "marie"},
                        {"run_id": "r-fail", "profile": "marie"},
                    ]
                }
            )
        )

        assert [r["run_id"] for r in result["completed"]] == ["r-done"]
        assert [r["run_id"] for r in result["running"]] == ["r-run"]
        assert [r["run_id"] for r in result["unknown"]] == ["r-unk"]
        assert [r["run_id"] for r in result["failed"]] == ["r-fail"]
        assert "resolve-unknown" in result["unknown"][0]["note"]
        assert result["summary"] == {
            "total": 4,
            "completed": 1,
            "running": 1,
            "unknown": 1,
            "failed": 1,
        }


def test_plugin_yaml_comment_is_accurate():
    manifest_text = (Path(__file__).resolve().parents[1] / "plugin.yaml").read_text(
        encoding="utf-8"
    )
    assert "transport candidate" not in manifest_text
    assert "plugins.isolation: host" in manifest_text

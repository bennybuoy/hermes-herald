"""Regression tests for callback.py — cancel flags and stall detection."""
import os
import sys
import types
import tempfile
import threading
import time
import json
import queue
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from hermes_herald import callback
from hermes_herald import tools


class TestCancelFlags:
    """Cancel flag suppression in _deliver_to_session."""

    def test_cancelled_run_suppressed(self):
        """When cancel flag is set, _deliver_to_session does not deliver."""
        delivered = []

        class FakeQueue:
            def put(self, evt):
                delivered.append(evt)

        class FakeReg:
            completion_queue = FakeQueue()

        sys.modules["tools.process_registry"] = types.ModuleType("process_registry")
        sys.modules["tools.process_registry"].process_registry = FakeReg()

        run_id = "test-cancel-cb-1"
        flag = threading.Event()
        callback._cancel_flags[run_id] = flag
        flag.set()

        callback._deliver_to_session({"type": "test"}, run_id, "marie")
        assert len(delivered) == 0, "Should suppress cancelled delivery"
        assert run_id not in callback._cancel_flags

    def test_normal_delivery_proceeds(self):
        """Without cancel flag, delivery proceeds normally."""
        delivered = []

        class FakeQueue:
            def put(self, evt):
                delivered.append(evt)

        class FakeReg:
            completion_queue = FakeQueue()

        sys.modules["tools.process_registry"] = types.ModuleType("process_registry")
        sys.modules["tools.process_registry"].process_registry = FakeReg()

        run_id = "test-cancel-cb-2"
        flag = threading.Event()
        callback._cancel_flags[run_id] = flag

        callback._deliver_to_session({"type": "test"}, run_id, "marie")
        assert len(delivered) == 1, "Should deliver normally"
        assert run_id not in callback._cancel_flags

    def test_stop_listener_sets_flag(self):
        """stop_listener sets the cancel flag before removing the thread."""
        run_id = "test-stop-1"
        callback._listeners[run_id] = threading.Thread(target=lambda: None)
        assert callback.stop_listener(run_id) is True
        assert run_id in callback._cancel_flags
        assert callback._cancel_flags[run_id].is_set()
        assert run_id not in callback._listeners

    def test_delivery_and_cancellation_are_linearized(self):
        """Cancellation cannot slip between the delivery check and enqueue."""
        entered_put = threading.Event()
        release_put = threading.Event()
        delivered = []

        class BlockingQueue:
            def put(self, evt):
                entered_put.set()
                assert release_put.wait(2)
                delivered.append(evt)

        class FakeReg:
            completion_queue = BlockingQueue()

        module = types.ModuleType("process_registry")
        setattr(module, "process_registry", FakeReg())
        sys.modules["tools.process_registry"] = module

        run_id = "test-cancel-linearized"
        callback._listeners[run_id] = threading.Thread(target=lambda: None)
        delivery = threading.Thread(
            target=callback._deliver_to_session,
            args=({"type": "test"}, run_id, "marie"),
        )
        delivery.start()
        assert entered_put.wait(1)

        cancellation_done = threading.Event()

        def cancel():
            callback.stop_listener(run_id)
            cancellation_done.set()

        cancellation = threading.Thread(target=cancel)
        cancellation.start()
        time.sleep(0.05)
        assert not cancellation_done.is_set()

        release_put.set()
        delivery.join(1)
        cancellation.join(1)
        assert delivered == [{"type": "test"}]
        assert cancellation_done.is_set()

    def test_remote_stop_and_delivery_are_one_transaction(self):
        """A completion ready during a successful stop is suppressed."""
        request_started = threading.Event()
        release_request = threading.Event()
        delivered = []

        class FakeQueue:
            def put(self, evt):
                delivered.append(evt)

        module = types.ModuleType("process_registry")
        setattr(module, "process_registry", SimpleNamespace(completion_queue=FakeQueue()))
        sys.modules["tools.process_registry"] = module

        run_id = "test-cancel-http-race"
        callback._listeners[run_id] = threading.Thread(target=lambda: None)

        def stop_request():
            request_started.set()
            assert release_request.wait(2)
            return {"status": "stopping"}

        cancel = threading.Thread(
            target=callback.cancel_delivery_transaction,
            args=(run_id, stop_request),
        )
        cancel.start()
        assert request_started.wait(1)

        delivery = threading.Thread(
            target=callback._deliver_to_session,
            args=({"type": "terminal"}, run_id, "marie"),
        )
        delivery.start()
        time.sleep(0.05)
        assert delivered == []

        release_request.set()
        cancel.join(1)
        delivery.join(1)
        assert delivered == []

    def test_failed_remote_stop_releases_waiting_completion(self):
        """A failed stop preserves and delivers a completion that was waiting."""
        request_started = threading.Event()
        release_request = threading.Event()
        delivered = []
        errors = []

        class FakeQueue:
            def put(self, evt):
                delivered.append(evt)

        module = types.ModuleType("process_registry")
        setattr(module, "process_registry", SimpleNamespace(completion_queue=FakeQueue()))
        sys.modules["tools.process_registry"] = module

        run_id = "test-cancel-http-failure"

        def stop_request():
            request_started.set()
            assert release_request.wait(2)
            raise RuntimeError("network down")

        def cancel():
            try:
                callback.cancel_delivery_transaction(run_id, stop_request)
            except RuntimeError as exc:
                errors.append(str(exc))

        cancel_thread = threading.Thread(target=cancel)
        cancel_thread.start()
        assert request_started.wait(1)
        delivery = threading.Thread(
            target=callback._deliver_to_session,
            args=({"type": "terminal"}, run_id, "marie"),
        )
        delivery.start()

        release_request.set()
        cancel_thread.join(1)
        delivery.join(1)
        assert errors == ["network down"]
        assert delivered == [{"type": "terminal"}]


class TestSessionRouting:
    def test_tui_route_uses_context_and_parent_durable_session(self, monkeypatch):
        values = {
            "HERMES_SESSION_SOURCE": "tui",
            "HERMES_SESSION_ID": "context-session",
            "HERMES_UI_SESSION_ID": "ui-session-7",
            "HERMES_SESSION_KEY": "stale-key",
        }
        monkeypatch.setattr(
            callback,
            "_get_session_env",
            lambda name, default="": values.get(name, default),
        )
        monkeypatch.setattr(callback, "_get_current_session_key", lambda: "approval-key")

        route = callback.capture_session_routing(
            SimpleNamespace(session_id="durable-agent-session")
        )

        assert route == {
            "session_id": "context-session",
            "session_key": "durable-agent-session",
            "origin_ui_session_id": "ui-session-7",
        }

    def test_delegation_event_carries_origin_ui_session(self):
        run_id = "routing-event-1"
        callback._delivery_routes[run_id] = {
            "origin_ui_session_id": "ui-owner",
        }
        evt = callback._build_delegation_event(
            run_id, "marie", "done", {}, "goal", "sid", "key"
        )
        assert evt["origin_ui_session_id"] == "ui-owner"


class TestBuildDelegationEvent:
    """Test _build_delegation_event status handling."""

    def test_cancelled_status(self):
        """run.cancelled maps to status='cancelled', not 'failed'."""
        evt = callback._build_delegation_event(
            "run-1", "marie", "", {}, "test", "sess", "key",
            status="cancelled",
            error="run was cancelled",
        )
        assert evt["status"] == "cancelled"
        assert evt["error"] == "run was cancelled"

    def test_completed_status(self):
        """run.completed maps to status='completed'."""
        evt = callback._build_delegation_event(
            "run-2", "marie", "Hello!", {"total_tokens": 100},
            "test", "sess", "key",
            status="completed",
            model="glm-5.2",
            duration_seconds=3.14,
        )
        assert evt["status"] == "completed"
        assert evt["summary"] == "Hello!"
        assert evt["model"] == "glm-5.2"
        assert evt["duration_seconds"] == 3.14

    def test_failed_status(self):
        """run.failed maps to status='failed'."""
        evt = callback._build_delegation_event(
            "run-3", "marie", "", {}, "test", "sess", "key",
            status="failed",
            error="something broke",
        )
        assert evt["status"] == "failed"
        assert evt["error"] == "something broke"


def _state_path():
    return os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")


class _FakeResponse:
    """Minimal stand-in for an urlopen() SSE context manager.

    Yields the provided byte chunks in order from read(); once exhausted,
    read() returns b"" which ends the SSE loop.
    """

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def _sse(evt_dict):
    return b"data: " + json.dumps(evt_dict).encode() + b"\n\n"


def _setup_fake_registry():
    """Install a fake process_registry and return the delivered-events list."""
    delivered = []

    class FakeQueue:
        def put(self, evt):
            delivered.append(evt)

    class FakeReg:
        completion_queue = FakeQueue()

    mod = types.ModuleType("tools.process_registry")
    mod.process_registry = FakeReg()
    sys.modules["tools.process_registry"] = mod
    return delivered


@pytest.fixture(autouse=True)
def _isolate_callback_state():
    """Clear approval + cancel + listener state between tests."""
    with callback._pending_approvals_lock:
        callback._pending_approvals.clear()
        callback._pending_approval_queues.clear()
        callback._local_approval_responses.clear()
        callback._approval_notice_events.clear()
        callback._approval_notice_ids_by_run.clear()
    with callback._listeners_lock:
        callback._listeners.clear()
    callback._cancel_flags.clear()
    for p in (_state_path(),):
        if os.path.exists(p):
            os.unlink(p)
    yield
    with callback._pending_approvals_lock:
        callback._pending_approvals.clear()
        callback._pending_approval_queues.clear()
        callback._local_approval_responses.clear()
        callback._approval_notice_events.clear()
        callback._approval_notice_ids_by_run.clear()
    callback._cancel_flags.clear()


class TestApprovalRelay:
    """approval.request / approval.responded handling in _listen_sse."""

    def test_approval_request_delivered_and_stored(self):
        """approval.request delivers a notice and stores pending metadata."""
        delivered = _setup_fake_registry()
        run_id = "run-appr-1"
        # Persist a run record so _update_pending_approval has something to tag.
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")

        stream = [_sse({
            "event": "approval.request", "run_id": run_id,
            "command": "rm -rf data", "description": "destructive",
            "choices": ["once", "session", "always", "deny"],
            "timestamp": 1700000000,
        })]

        # After the stream closes, reconnection attempts hit a connection
        # error (server gone). Limit to 1 attempt so the test doesn't sleep.
        from urllib.error import URLError
        from unittest.mock import MagicMock
        gone_resp = URLError("connection refused")

        with patch.object(callback, "urlopen", side_effect=[_FakeResponse(stream), gone_resp]), \
             patch.object(callback.time, "sleep"), \
             patch.object(callback, "_SSE_RECONNECT_ATTEMPTS", 1):
            callback._listen_sse(
                run_id, "marie", "http://x", "key", "preview", "sess", "skey",
            )

        # One approval event delivered during the stream.
        approval_evts = [e for e in delivered if e.get("status") == "approval_required"]
        assert len(approval_evts) == 1
        evt = approval_evts[0]
        assert evt["type"] == "async_delegation"
        assert evt["status"] == "approval_required"
        assert "[DISPATCH APPROVAL REQUIRED]" in evt["summary"]
        assert "Profile: marie" in evt["summary"]
        assert f"Run: {run_id}" in evt["summary"]
        assert "rm -rf data" in evt["summary"]
        assert evt["approval"]["run_id"] == run_id
        assert evt["delegation_id"].startswith(f"dispatch-approval-{run_id[:12]}-")

    def test_approval_responded_clears_pending(self):
        """approval.responded clears pending metadata; listener continues."""
        delivered = _setup_fake_registry()
        run_id = "run-appr-2"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")

        stream = [
            _sse({
                "event": "approval.request", "run_id": run_id,
                "command": "rm -rf data", "description": "destructive",
                "choices": ["once", "session", "always", "deny"],
            }),
            _sse({"event": "approval.responded", "run_id": run_id}),
        ]

        # After the stream closes, reconnection hits connection error.
        from urllib.error import URLError
        gone_resp = URLError("connection refused")

        with patch.object(callback, "urlopen", side_effect=[_FakeResponse(stream), gone_resp]), \
             patch.object(callback.time, "sleep"), \
             patch.object(callback, "_SSE_RECONNECT_ATTEMPTS", 1), \
             patch.object(callback, "_retire_approval_delivery") as retire_delivery:
            callback._listen_sse(
                run_id, "marie", "http://x", "key", "preview", "sess", "skey",
            )

        # Only the approval.request notice was delivered as an approval event.
        approval_evts = [e for e in delivered if e.get("status") == "approval_required"]
        assert len(approval_evts) == 1
        # The exact FIFO head notice is retired; later requests would remain.
        retire_delivery.assert_any_call(
            run_id, approval_evts[0]["delegation_id"],
        )
        # Pending metadata cleared after exhaustion.
        assert callback.get_pending_approval(run_id) is None

    def test_retirement_removes_queued_notice(self):
        """Resolving before an idle drain removes the queued notice."""
        completion_queue = queue.Queue()
        mod = types.ModuleType("tools.process_registry")
        mod.process_registry = types.SimpleNamespace(completion_queue=completion_queue)
        sys.modules["tools.process_registry"] = mod

        evt = {
            "type": "async_delegation",
            "delegation_id": "dispatch-approval-run-race-1234567890",
            "session_key": "skey",
            "session_id": "sess",
            "status": "approval_required",
            "dispatched_at": 1.0,
            "completed_at": 1.0,
            "goal": "approval required",
        }
        callback._approval_notice_events[evt["delegation_id"]] = evt
        callback._approval_notice_ids_by_run["run-race"] = {evt["delegation_id"]}
        completion_queue.put(evt)
        callback._retire_approval_delivery("run-race", evt["delegation_id"])

        assert completion_queue.empty()
        assert evt["retired"] is True
        assert evt["session_key"] == callback._RETIRED_SESSION_KEY

    def test_retirement_invalidates_notice_popped_by_busy_poller(self):
        """A later requeue carries unowned routing and cannot target origin."""
        completion_queue = queue.Queue()
        mod = types.ModuleType("tools.process_registry")
        mod.process_registry = types.SimpleNamespace(completion_queue=completion_queue)
        sys.modules["tools.process_registry"] = mod

        evt = {
            "type": "async_delegation",
            "delegation_id": "dispatch-approval-run-inflight-123",
            "session_key": "origin",
            "session_id": "origin",
            "status": "approval_required",
        }
        callback._approval_notice_events[evt["delegation_id"]] = evt
        callback._approval_notice_ids_by_run["run-inflight"] = {
            evt["delegation_id"],
        }

        # Simulate a busy TUI poller holding the object before it requeues it.
        callback._retire_approval_delivery("run-inflight", evt["delegation_id"])
        completion_queue.put(evt)

        requeued = completion_queue.get_nowait()
        assert requeued is evt
        assert requeued["retired"] is True
        assert requeued["session_key"] == callback._RETIRED_SESSION_KEY
        assert requeued["origin_ui_session_id"] == callback._RETIRED_SESSION_KEY

    def test_multiple_requests_remain_fifo_and_promote_one_at_a_time(self):
        """The notice a human sees must remain the target's FIFO head."""
        completion_queue = queue.Queue()
        mod = types.ModuleType("tools.process_registry")
        mod.process_registry = types.SimpleNamespace(completion_queue=completion_queue)
        sys.modules["tools.process_registry"] = mod
        run_id = "run-susan-race"

        def publish(delivery_id):
            approval = {
                "run_id": run_id,
                "profile": "richard",
                "command": delivery_id,
                "description": "test",
                "choices": ["once", "session", "always", "deny"],
                "delivery_id": delivery_id,
            }
            if callback._set_pending_approval(run_id, approval):
                callback._deliver_approval_required(approval, "origin", "origin")
            return approval

        request_a_data = publish("request-a")
        request_a = completion_queue.get_nowait()
        publish("request-b")
        publish("request-c")
        assert completion_queue.empty()
        assert callback.get_pending_approval(run_id)["delivery_id"] == "request-a"

        request_b_data = callback._advance_pending_approval(
            run_id, request_a_data["delivery_id"],
        )
        assert request_a["retired"] is True
        callback._deliver_approval_required(request_b_data, "origin", "origin")
        request_b = completion_queue.get_nowait()
        assert callback.get_pending_approval(run_id)["delivery_id"] == "request-b"

        request_c_data = callback._advance_pending_approval(run_id, "request-b")
        assert request_b["retired"] is True
        callback._deliver_approval_required(request_c_data, "origin", "origin")
        request_c = completion_queue.get_nowait()
        assert callback.get_pending_approval(run_id)["delivery_id"] == "request-c"

        callback._clear_pending_approval_mem(run_id)
        assert request_c["retired"] is True
        assert callback.get_pending_approval(run_id) is None
        assert run_id not in callback._approval_notice_ids_by_run
        assert not callback._approval_notice_events

    def test_listener_continues_after_approval_events(self):
        """The SSE listener does NOT exit on approval events.

        After an approval.request with no terminal event, the only delivery
        is the approval notice. The listener keeps reading until the stream
        closes and then attempts reconnection.
        """
        delivered = _setup_fake_registry()
        run_id = "run-appr-3"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")

        stream = [
            _sse({
                "event": "approval.request", "run_id": run_id,
                "command": "c", "description": "d",
                "choices": ["once", "session", "always", "deny"],
            }),
        ]

        # After the stream closes, reconnection hits connection error.
        from urllib.error import URLError
        gone_resp = URLError("connection refused")

        with patch.object(callback, "urlopen", side_effect=[_FakeResponse(stream), gone_resp]), \
             patch.object(callback.time, "sleep"), \
             patch.object(callback, "_SSE_RECONNECT_ATTEMPTS", 1):
            callback._listen_sse(
                run_id, "marie", "http://x", "key", "preview", "sess", "skey",
            )

        # Exactly one approval delivery — no spurious completion/stall.
        approval_evts = [e for e in delivered if e.get("status") == "approval_required"]
        assert len(approval_evts) == 1
        assert approval_evts[0]["status"] == "approval_required"

    def test_terminal_event_clears_pending_after_approval(self):
        """run.completed after an approval clears pending metadata + delivers."""
        delivered = _setup_fake_registry()
        run_id = "run-appr-4"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")

        stream = [
            _sse({
                "event": "approval.request", "run_id": run_id,
                "command": "c", "description": "d",
                "choices": ["once", "session", "always", "deny"],
            }),
            _sse({
                "event": "run.completed", "run_id": run_id,
                "output": "done", "usage": {"total_tokens": 10},
                "model": "glm-5.2",
            }),
            b"",
        ]

        with patch.object(callback, "urlopen", return_value=_FakeResponse(stream)):
            callback._listen_sse(
                run_id, "marie", "http://x", "key", "preview", "sess", "skey",
            )

        # Two deliveries: the approval notice, then the completion.
        assert len(delivered) == 2
        statuses = [e["status"] for e in delivered]
        assert "approval_required" in statuses
        assert "completed" in statuses
        assert callback.get_pending_approval(run_id) is None

    def test_approval_responded_is_an_activity_event(self):
        """approval.responded is registered as an activity event."""
        assert "approval.responded" in callback._ACTIVITY_EVENTS
        assert "approval.request" in callback._ACTIVITY_EVENTS


class _FakeJSONResponse:
    """Minimal JSON response for status-polling regressions."""

    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _release_event(summary="done"):
    return {
        "type": "async_delegation",
        "status": "completed",
        "summary": summary,
        "error": "",
    }


def _listen_args():
    return (
        "run-release", "reviewer", "http://target", "secret",
        "preview", "session", "session-key", "", "",
    )


class TestSSERecovery:
    def test_disconnect_reconnects_then_delivers_terminal_event(self):
        terminal = _release_event()
        with patch.object(
            callback,
            "_read_sse_stream",
            side_effect=[("disconnect", "lost"), ("terminal", terminal)],
        ) as read_stream, patch.object(
            callback, "_check_run_status", return_value=None
        ) as check_status, patch.object(
            callback, "_deliver_to_session"
        ) as deliver, patch.object(
            callback, "_cleanup_listener"
        ), patch.object(callback.time, "sleep") as sleep:
            callback._listen_sse(*_listen_args())

        assert read_stream.call_count == 2
        check_status.assert_called_once()
        sleep.assert_called_once_with(5)
        deliver.assert_called_once_with(terminal, "run-release", "reviewer")

    def test_pre_reconnect_check_catches_completed_run(self):
        terminal = _release_event("finished while disconnected")
        with patch.object(
            callback, "_read_sse_stream", return_value=("disconnect", "lost")
        ) as read_stream, patch.object(
            callback, "_check_run_status", return_value=terminal
        ) as check_status, patch.object(
            callback, "_deliver_to_session"
        ) as deliver, patch.object(
            callback, "_cleanup_listener"
        ), patch.object(callback.time, "sleep") as sleep:
            callback._listen_sse(*_listen_args())

        read_stream.assert_called_once()
        check_status.assert_called_once()
        sleep.assert_not_called()
        deliver.assert_called_once_with(terminal, "run-release", "reviewer")

    def test_exhausted_reconnects_fall_back_to_polling(self):
        terminal = _release_event("completed via polling")
        with patch.object(
            callback, "_SSE_RECONNECT_ATTEMPTS", 3
        ), patch.object(
            callback, "_read_sse_stream", return_value=("disconnect", "lost")
        ) as read_stream, patch.object(
            callback, "_check_run_status", return_value=None
        ) as check_status, patch.object(
            callback, "_poll_run_status", return_value=terminal
        ) as poll_status, patch.object(
            callback, "_deliver_to_session"
        ) as deliver, patch.object(
            callback, "_cleanup_listener"
        ), patch.object(callback.time, "sleep") as sleep:
            callback._listen_sse(*_listen_args())

        assert read_stream.call_count == 3
        assert check_status.call_count == 3
        assert [item.args[0] for item in sleep.call_args_list] == [5, 10]
        poll_status.assert_called_once()
        deliver.assert_called_once_with(terminal, "run-release", "reviewer")

    def test_reconnect_404_after_disconnect_falls_back_to_polling(self):
        """A consumed core SSE queue disappears while its run remains active."""
        terminal = _release_event("completed after events endpoint disappeared")
        with patch.object(
            callback,
            "_read_sse_stream",
            side_effect=[
                ("disconnect", "wire dropped"),
                ("not_found", "events endpoint 404"),
            ],
        ) as read_stream, patch.object(
            callback, "_check_run_status", return_value=None
        ) as check_status, patch.object(
            callback, "_poll_run_status", return_value=terminal
        ) as poll_status, patch.object(
            callback, "_deliver_to_session"
        ) as deliver, patch.object(
            callback, "_cleanup_listener"
        ), patch.object(callback.time, "sleep"):
            callback._listen_sse(*_listen_args())

        assert read_stream.call_count == 2
        assert check_status.call_count == 2
        poll_status.assert_called_once()
        deliver.assert_called_once_with(terminal, "run-release", "reviewer")


class TestActivityAwarePolling:
    def test_rfc3339_updated_at_resets_activity_without_crashing(self):
        responses = [
            _FakeJSONResponse({
                "status": "running",
                "updated_at": "2026-07-26T12:34:56+00:00",
            }),
            _FakeJSONResponse({
                "status": "completed",
                "updated_at": "2026-07-26T12:34:57Z",
                "output": "done",
                "usage": {},
            }),
        ]
        with patch.object(
            callback, "urlopen", side_effect=responses
        ), patch.object(
            callback.time, "sleep"
        ), patch.object(callback, "_STALL_TIMEOUT", 10):
            result = callback._poll_run_status(
                "run-rfc3339", "reviewer", "http://target", "secret",
                "preview", "session", "session-key", "", "",
                start_time=1.0, poll_start=callback.time.time(),
            )

        assert result is not None
        assert result["status"] == "completed"
        assert result["summary"] == "done"

    def test_stall_fires_when_updated_at_stops_advancing(self):
        running = {"status": "running", "updated_at": 1.0}
        times = iter([1.0, 1.0, 1.0, 2.0])
        with patch.object(
            callback, "urlopen", return_value=_FakeJSONResponse(running)
        ), patch.object(
            callback.time, "time", side_effect=lambda: next(times, 2.0)
        ), patch.object(
            callback.time, "sleep"
        ), patch.object(callback, "_STALL_TIMEOUT", 0):
            result = callback._poll_run_status(
                "run-stall", "reviewer", "http://target", "secret",
                "preview", "session", "session-key", "", "",
                start_time=1.0, poll_start=1.0,
            )

        assert result is not None
        assert result["status"] == "failed"
        assert "stall" in result["error"]

    def test_updated_at_activity_allows_later_completion(self):
        responses = [
            _FakeJSONResponse({"status": "running", "updated_at": 1.0}),
            _FakeJSONResponse({"status": "running", "updated_at": 2.0}),
            _FakeJSONResponse({
                "status": "completed", "updated_at": 2.0,
                "output": "done", "usage": {},
            }),
        ]
        times = iter([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        with patch.object(
            callback, "urlopen", side_effect=responses
        ), patch.object(
            callback.time, "time", side_effect=lambda: next(times, 3.0)
        ), patch.object(
            callback.time, "sleep"
        ), patch.object(callback, "_STALL_TIMEOUT", 10):
            result = callback._poll_run_status(
                "run-active", "reviewer", "http://target", "secret",
                "preview", "session", "session-key", "", "",
                start_time=1.0, poll_start=1.0,
            )

        assert result is not None
        assert result["status"] == "completed"
        assert result["summary"] == "done"

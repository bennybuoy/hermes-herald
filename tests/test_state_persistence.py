"""Regression tests for Hermes Herald state persistence."""
import json
import logging
import os
import sys
import types
import tempfile
import threading
from datetime import datetime
import pytest

from hermes_herald import tools
from hermes_herald import callback


class TestStatePersistence:
    """Round-trip and concurrency tests for the state file."""

    def test_persist_and_load(self):
        """_persist_run writes, _load_state reads it back."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        tools._persist_run("test-run-1", "marie", "Hello test", model="glm-5.2")

        state = tools._load_state()
        assert len(state["runs"]) == 1
        run = state["runs"][0]
        assert run["run_id"] == "test-run-1"
        assert run["profile"] == "marie"
        assert run["model"] == "glm-5.2"
        assert run["status"] == "dispatched"

    def test_update_run_status(self):
        """_update_run_status merges terminal fields."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        tools._persist_run("test-run-2", "ada", "Test message", model="glm-5.2")
        tools._update_run_status(
            "test-run-2", "completed",
            output_preview="Done!", duration_seconds=3.14,
            usage={"total_tokens": 500}, model="glm-5.2",
        )

        state = tools._load_state()
        run = state["runs"][0]
        assert run["status"] == "completed"
        assert run["duration_seconds"] == 3.14
        assert run["output_preview"] == "Done!"
        assert run["usage"]["total_tokens"] == 500

    def test_concurrent_writes(self):
        """Multiple threads writing to state file don't lose entries."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        def write_run(i):
            tools._persist_run(f"run-{i}", "marie", f"message {i}", model="glm-5.2")

        threads = [threading.Thread(target=write_run, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        state = tools._load_state()
        assert len(state["runs"]) == 20, f"Expected 20 runs, got {len(state['runs'])}"

    def test_uses_os_replace(self):
        """_save_state uses os.replace, not os.rename."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        import unittest.mock
        with unittest.mock.patch.object(os, "replace") as mock_replace:
            tools._save_state({"runs": []})
            mock_replace.assert_called_once()

    def test_persist_run_uses_utc_timestamp(self):
        """dispatched_at is a UTC ISO-8601 timestamp matching the ledger."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        tools._persist_run("test-utc", "marie", "Hello", model="glm-5.2")
        run = tools._load_state()["runs"][0]
        stamp = run["dispatched_at"]
        assert stamp.endswith("+00:00"), stamp
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_update_run_status_uses_utc_timestamp(self):
        """completed_at is a UTC ISO-8601 timestamp matching the ledger."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        tools._persist_run("test-utc-done", "marie", "Hello", model="glm-5.2")
        tools._update_run_status("test-utc-done", "completed", output_preview="ok")
        run = tools._load_state()["runs"][0]
        stamp = run["completed_at"]
        assert stamp.endswith("+00:00"), stamp
        parsed = datetime.fromisoformat(stamp)
        assert parsed.utcoffset().total_seconds() == 0

    def test_persist_run_uses_passed_session_id_not_environ(self, monkeypatch):
        """_persist_run stores the caller-supplied session_id, never os.environ."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        monkeypatch.setenv("HERMES_SESSION_ID", "stale-env-session")
        tools._persist_run(
            "test-sid", "marie", "Hello", model="glm-5.2",
            session_id="live-captured-session",
        )
        run = tools._load_state()["runs"][0]
        assert run["session_id"] == "live-captured-session"

    def test_persist_run_does_not_read_environ_session_id(self, monkeypatch):
        """Omitting session_id records empty rather than falling back to env."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        monkeypatch.setenv("HERMES_SESSION_ID", "stale-env-session")
        tools._persist_run("test-sid-empty", "marie", "Hello", model="glm-5.2")
        run = tools._load_state()["runs"][0]
        assert run["session_id"] == ""

    def test_save_state_warns_and_prefers_non_terminal(self, caplog, monkeypatch):
        """Truncation logs a warning and keeps in-flight runs over completed ones."""
        monkeypatch.setattr(tools, "_MAX_STATE_ENTRIES", 2)
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        with caplog.at_level(logging.WARNING, logger="hermes_herald.tools"):
            tools._save_state({"runs": [
                {"run_id": "old-done", "status": "completed"},
                {"run_id": "mid-done", "status": "failed"},
                {"run_id": "live", "status": "dispatched"},
            ]})

        assert any("truncated" in rec.message.lower() for rec in caplog.records)
        ids = [run["run_id"] for run in tools._load_state()["runs"]]
        assert ids == ["mid-done", "live"]

    def test_trim_state_runs_drops_oldest_live_only_when_needed(self):
        """Non-terminal overflow still bounds the cache to the newest live runs."""
        runs = [
            {"run_id": "live-old", "status": "dispatched"},
            {"run_id": "live-mid", "status": "running"},
            {"run_id": "live-new", "status": "waiting_for_approval"},
        ]
        kept = tools._trim_state_runs(runs, 2)
        assert [run["run_id"] for run in kept] == ["live-mid", "live-new"]

    def test_trim_state_runs_evicts_stale_detached_not_fresh_chat(self):
        """Unmonitored delivery=none dispatches must not evict a fresh chat record."""
        stale = [
            {
                "run_id": f"detached-{i}",
                "status": "dispatched",
                "delivery": "none",
            }
            for i in range(200)
        ]
        chat = {
            "run_id": "chat-fresh",
            "profile": "marie",
            "session_id": "sess-fresh",
            "type": "chat",
            "status": "completed",
        }
        kept = tools._trim_state_runs(stale + [chat], 200)
        ids = [run["run_id"] for run in kept]
        assert len(kept) == 200
        assert "chat-fresh" in ids
        assert ids[0] == "detached-1"
        assert ids[-1] == "chat-fresh"
        assert "detached-0" not in ids

    def test_trim_state_runs_still_prefers_monitored_live_over_chat(self):
        """Callback-monitored dispatches still beat a completed chat record."""
        live = [
            {
                "run_id": f"live-{i}",
                "status": "dispatched",
                "delivery": "callback",
            }
            for i in range(200)
        ]
        chat = {
            "run_id": "chat-fresh",
            "type": "chat",
            "status": "completed",
        }
        kept = tools._trim_state_runs(live + [chat], 200)
        ids = [run["run_id"] for run in kept]
        assert len(kept) == 200
        assert "chat-fresh" not in ids
        assert ids[0] == "live-0"
        assert ids[-1] == "live-199"

    def test_trim_state_runs_keeps_polled_detached_running(self):
        """A delivery=none run polled to running stays live over evictable entries."""
        runs = [
            {"run_id": "old-chat", "type": "chat", "status": "completed"},
            {"run_id": "polled", "status": "running", "delivery": "none"},
            {"run_id": "stale", "status": "dispatched", "delivery": "none"},
        ]
        kept = tools._trim_state_runs(runs, 1)
        assert [run["run_id"] for run in kept] == ["polled"]


class TestCancelFlags:
    """Cancel flag suppression in callback._deliver_to_session."""

    def test_cancel_suppresses_delivery(self):
        """When cancel flag is set, _deliver_to_session does not deliver."""
        delivered = []

        class FakeQueue:
            def put(self, evt):
                delivered.append(evt)

        class FakeReg:
            completion_queue = FakeQueue()

        sys.modules["tools.process_registry"] = types.ModuleType("process_registry")
        sys.modules["tools.process_registry"].process_registry = FakeReg()

        run_id = "test-cancel-1"
        flag = threading.Event()
        callback._cancel_flags[run_id] = flag
        flag.set()

        callback._deliver_to_session({"type": "test"}, run_id, "marie")
        assert len(delivered) == 0, "Should have suppressed delivery"

    def test_normal_delivery_works(self):
        """When cancel flag is NOT set, delivery proceeds normally."""
        delivered = []

        class FakeQueue:
            def put(self, evt):
                delivered.append(evt)

        class FakeReg:
            completion_queue = FakeQueue()

        sys.modules["tools.process_registry"] = types.ModuleType("process_registry")
        sys.modules["tools.process_registry"].process_registry = FakeReg()

        run_id = "test-cancel-2"
        flag = threading.Event()
        callback._cancel_flags[run_id] = flag

        callback._deliver_to_session({"type": "test"}, run_id, "marie")
        assert len(delivered) == 1, "Should have delivered"


class TestSessionRecovery:
    """Session ID recovery from state file."""

    def test_recover_session_ids(self):
        """recover_session_ids reads chat entries from state file."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        with tools._state_lock:
            state = tools._load_state()
            state["runs"].append({
                "run_id": "chat-marie-123",
                "profile": "marie",
                "session_id": "sess-abc",
                "type": "chat",
                "status": "completed",
            })
            tools._save_state(state)

        n = callback.recover_session_ids()
        assert n >= 1, f"Expected at least 1 recovery, got {n}"
        sid = callback.get_profile_session_id("marie")

        assert sid == "sess-abc", f"Expected sess-abc, got {sid}"

    def test_stale_detached_runs_do_not_block_session_recovery(self):
        """200 unmonitored dispatches must not evict the latest chat session."""
        state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
        if os.path.exists(state_path):
            os.unlink(state_path)

        stale = [
            {
                "run_id": f"detached-{i}",
                "profile": "other",
                "status": "dispatched",
                "delivery": "none",
            }
            for i in range(200)
        ]
        chat = {
            "run_id": "chat-marie-fresh",
            "profile": "marie",
            "session_id": "sess-fresh",
            "type": "chat",
            "status": "completed",
        }
        with tools._state_lock:
            tools._save_state({"runs": stale + [chat]})

        n = callback.recover_session_ids()
        assert n >= 1, f"Expected at least 1 recovery, got {n}"
        assert callback.get_profile_session_id("marie") == "sess-fresh"
        ids = [run["run_id"] for run in tools._load_state()["runs"]]
        assert "chat-marie-fresh" in ids
        assert len(ids) == 200

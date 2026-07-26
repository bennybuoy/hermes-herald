"""Regression tests for Hermes Herald state persistence."""
import json
import os
import sys
import types
import tempfile
import threading
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
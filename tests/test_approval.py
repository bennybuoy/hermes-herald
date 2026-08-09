"""Tests for the approval relay + resolution feature (GitHub issue #7).

Covers:
  - approve_dispatch handler (deny-only / error cases / scope)
  - pending approval metadata persistence + clearing
  - stall timer suppression while awaiting approval
  - cancel clears pending approval metadata
  - check_dispatch / dispatch_status surface waiting_for_approval
  - recovery after restart
  - credential values never leak into tool output or the state file
"""
import json
import os
import sys
import types
import threading
from unittest.mock import patch, MagicMock

import pytest

from hermes_herald import tools
from hermes_herald import callback


# A deliberately credential-shaped API key + command. None of these must ever
# appear in tool output, the state file, or origin-session notifications.
_API_KEY = "AK_SECRET_VALUE_123xyz"
_SECRET_IN_COMMAND = "TOKEN_LEAK_CANARY_987"

PROFILE_CFG = {"url": "http://localhost:9999", "api_key": _API_KEY}


def _state_path() -> str:
    return os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")


@pytest.fixture(autouse=True)
def _isolate_state():
    """Reset in-memory dicts and the state file before each test."""
    with callback._pending_approvals_lock:
        callback._pending_approvals.clear()
        callback._pending_approval_queues.clear()
        callback._local_approval_responses.clear()
    with callback._listeners_lock:
        callback._listeners.clear()
    callback._cancel_flags.clear()
    with callback._session_ids_lock:
        callback._session_ids.clear()
    for p in (_state_path(),):
        if os.path.exists(p):
            os.unlink(p)
    yield
    with callback._pending_approvals_lock:
        callback._pending_approvals.clear()
        callback._pending_approval_queues.clear()
        callback._local_approval_responses.clear()
    callback._cancel_flags.clear()


@pytest.fixture(autouse=True)
def _grant_test_human_consent(monkeypatch):
    """Unit tests opt into the human gate; production defaults fail closed."""
    monkeypatch.setattr(
        tools,
        "_request_dispatch_approval_consent",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        callback,
        "capture_session_routing",
        lambda parent_agent=None: {
            "session_id": "origin-session",
            "session_key": "origin-key",
            "origin_ui_session_id": "origin-ui",
        },
    )


def _resolve_profile_ok(profile: str):
    """Patch target for tools._resolve_profile that always succeeds."""
    return (dict(PROFILE_CFG), None)


def _set_pending(run_id: str, profile: str = "marie") -> dict:
    """Populate in-memory + state-file pending approval metadata."""
    approval_data = {
        "run_id": run_id,
        "profile": profile,
        "command": f"rm -rf {profile}/data",
        "description": "destructive command needs approval",
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
    callback._set_pending_approval(run_id, approval_data)
    tools._persist_run(run_id, profile, "preview", model="glm-5.2")
    tools._update_pending_approval(run_id, approval_data)
    return approval_data


def _approval_id(run_id: str) -> str:
    return f"approval-{run_id}"


# ---------------------------------------------------------------------------
# approve_dispatch handler
# ---------------------------------------------------------------------------


class TestApproveDispatch:
    @pytest.mark.parametrize("choice", ["once", "session", "always"])
    def test_positive_choices_fail_closed_before_target_contact(self, choice):
        run_id = f"run-positive-{choice}"
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile") as resolve, \
             patch.object(tools, "_get_json") as get_status, \
             patch.object(tools, "_request_dispatch_approval_consent") as consent, \
             patch.object(tools, "_post_json") as post:
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id,
                "profile": "marie",
                "choice": choice,
                "approval_id": _approval_id(run_id),
            }))

        assert result["status"] == "error"
        assert "deny-only" in result["error"]
        resolve.assert_not_called()
        get_status.assert_not_called()
        consent.assert_not_called()
        post.assert_not_called()
        assert callback.get_pending_approval(run_id) is not None

    def test_deny_resolves_and_returns_count(self):
        run_id = "run-approve-1"
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}) as gj, \
             patch.object(tools, "_post_json", return_value={"resolved": 1, "status": "running"}) as pj:

            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": _approval_id(run_id),
            }))

        assert result["resolved"] == 1
        assert result["status"] == "running"
        assert result["choice"] == "deny"
        assert result["resolve_all"] is False
        # Posted to the approval endpoint with the right body
        posted_url = pj.call_args.args[0]
        assert posted_url.endswith(f"/v1/runs/{run_id}/approval")
        assert pj.call_args.args[2] == {"choice": "deny", "all": False}
        # Status was fetched first
        assert gj.call_count == 1
        # Pending metadata cleared after resolution
        assert callback.get_pending_approval(run_id) is None

    def test_choice_deny_sends_deny(self):
        run_id = "run-approve-deny"
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}), \
             patch.object(tools, "_post_json", return_value={"resolved": 1, "status": "running"}) as pj:

            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": _approval_id(run_id),
            }))

        assert result["choice"] == "deny"
        assert pj.call_args.args[2] == {"choice": "deny", "all": False}

    def test_resolve_all_deny_passes_all_flag(self):
        run_id = "run-approve-all"
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}), \
             patch.object(tools, "_post_json", return_value={"resolved": 3, "status": "running"}) as pj:

            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie",
                "choice": "deny", "resolve_all": True,
                "approval_id": _approval_id(run_id),
            }))

        assert result["resolved"] == 3
        assert pj.call_args.args[2] == {"choice": "deny", "all": True}

    def test_resolve_all_refuses_positive_choice_before_target_contact(self):
        run_id = "run-positive-all"
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile") as resolve, \
             patch.object(tools, "_get_json") as get_status, \
             patch.object(tools, "_post_json") as post:
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie",
                "choice": "session", "resolve_all": True,
                "approval_id": _approval_id(run_id),
            }))

        assert result["status"] == "error"
        assert "Positive remote approval" in result["error"]
        resolve.assert_not_called()
        get_status.assert_not_called()
        post.assert_not_called()

    def test_human_decline_sends_nothing_and_preserves_pending(self):
        run_id = "run-human-decline"
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}), \
             patch.object(tools, "_request_dispatch_approval_consent", return_value=False), \
             patch.object(tools, "_post_json") as pj:
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": _approval_id(run_id),
            }))

        assert result["status"] == "error"
        assert "Human confirmation was not granted" in result["error"]
        pj.assert_not_called()
        assert callback.get_pending_approval(run_id) is not None

    def test_fails_when_not_waiting_for_approval(self):
        run_id = "run-approve-notwaiting"
        # No pending metadata in memory; GET reports a non-waiting status.
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "running"}), \
             patch.object(tools, "_post_json") as pj:

            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": f"approval-{run_id}",
            }))

        assert result["status"] == "error"
        assert "not awaiting approval" in result["error"]
        pj.assert_not_called()

    def test_fails_for_wrong_profile_run_id_pair(self):
        run_id = "run-approve-scope"
        # Pending metadata says this run belongs to "marie".
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json") as gj, \
             patch.object(tools, "_post_json") as pj:

            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "ada", "choice": "deny",
                "approval_id": _approval_id(run_id),
            }))

        assert result["status"] == "error"
        assert "scoped to the exact profile/run_id pair" in result["error"]
        # Fails closed before any HTTP call to the target.
        gj.assert_not_called()
        pj.assert_not_called()
        # Pending metadata is preserved (nothing was resolved).
        assert callback.get_pending_approval(run_id) is not None

    def test_fails_when_run_not_found(self):
        run_id = "run-approve-missing"
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "not_found"}), \
             patch.object(tools, "_post_json") as pj:

            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": f"approval-{run_id}",
            }))

        assert result["status"] == "error"
        assert "not found" in result["error"]
        pj.assert_not_called()

    def test_409_clears_stale_pending_metadata(self):
        run_id = "run-approve-409"
        _set_pending(run_id, "marie")

        def boom(*a, **kw):
            raise RuntimeError("HTTP 409 from http://x: no active approval session")

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}), \
             patch.object(tools, "_post_json", side_effect=boom):

            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": _approval_id(run_id),
            }))

        assert result["status"] == "error"
        assert "No active approval session" in result["error"]
        assert callback.get_pending_approval(run_id) is None

    def test_invalid_choice_rejected(self):
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_post_json") as pj:
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": "r", "profile": "marie", "choice": "maybe",
            }))
        assert result["status"] == "error"
        pj.assert_not_called()

    def test_missing_choice_rejected(self):
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_post_json") as pj:
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": "r", "profile": "marie",
            }))
        assert result["status"] == "error"
        assert "auto-approves" in result["error"]
        pj.assert_not_called()

    def test_unresolved_profile_fails(self):
        with patch.object(tools, "_resolve_profile", return_value=({}, "Profile 'x' not found")):
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": "r", "profile": "x", "choice": "deny",
            }))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Pending metadata persistence + clearing
# ---------------------------------------------------------------------------


class TestPendingMetadataPersistence:
    def test_update_and_clear_pending_approval_in_state_file(self):
        run_id = "run-pending-state"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")
        approval_data = {
            "run_id": run_id, "profile": "marie", "command": "rm -rf x",
            "description": "destructive", "choices": ["once", "session", "always", "deny"],
            "timestamp": 1, "requested_at": 1.0,
        }
        tools._update_pending_approval(run_id, approval_data)

        state = tools._load_state()
        run = next(r for r in state["runs"] if r["run_id"] == run_id)
        assert run["status"] == "waiting_for_approval"
        assert run["pending_approval"] == approval_data

        tools._clear_pending_approval(run_id)
        state = tools._load_state()
        run = next(r for r in state["runs"] if r["run_id"] == run_id)
        assert "pending_approval" not in run

    def test_pending_cleared_after_resolution(self):
        run_id = "run-clear-after-resolve"
        _set_pending(run_id, "marie")
        assert callback.get_pending_approval(run_id) is not None

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}), \
             patch.object(tools, "_post_json", return_value={"resolved": 1, "status": "running"}):
            tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": _approval_id(run_id),
            })

        assert callback.get_pending_approval(run_id) is None
        state = tools._load_state()
        run = next(r for r in state["runs"] if r["run_id"] == run_id)
        assert "pending_approval" not in run


class TestApprovalOwnership:
    def test_wrong_session_or_nonce_never_posts_approval(self, monkeypatch):
        run_id = "run-owner-bound"
        _set_pending(run_id, "marie")
        monkeypatch.setattr(
            callback,
            "capture_session_routing",
            lambda parent_agent=None: {
                "session_id": "other-session",
                "session_key": "other-key",
                "origin_ui_session_id": "other-ui",
            },
        )
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json") as gj, \
             patch.object(tools, "_post_json") as pj:
            wrong_owner = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": _approval_id(run_id),
            }))
        assert wrong_owner["status"] == "error"
        assert "originating session" in wrong_owner["error"]
        gj.assert_not_called()
        pj.assert_not_called()

        monkeypatch.setattr(
            callback,
            "capture_session_routing",
            lambda parent_agent=None: {
                "session_id": "origin-session",
                "session_key": "origin-key",
                "origin_ui_session_id": "origin-ui",
            },
        )
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json") as gj, \
             patch.object(tools, "_post_json") as pj:
            wrong_nonce = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": "stale-approval-id",
            }))
        assert wrong_nonce["status"] == "error"
        assert "approval_id" in wrong_nonce["error"]
        gj.assert_not_called()
        pj.assert_not_called()

    def test_denying_visible_request_resolves_fifo_head_and_promotes_next(self):
        run_id = "run-fifo-bound"
        first = _set_pending(run_id, "marie")
        second = dict(first)
        second.update({
            "command": "COMMAND_B",
            "delivery_id": "approval-command-b",
            "requested_at": first["requested_at"] + 1,
        })
        assert callback._set_pending_approval(run_id, second) is False

        completion_queue = __import__("queue").Queue()
        module = types.ModuleType("tools.process_registry")
        setattr(
            module,
            "process_registry",
            types.SimpleNamespace(completion_queue=completion_queue),
        )
        sys.modules["tools.process_registry"] = module

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}), \
             patch.object(tools, "_post_json", return_value={"resolved": 1, "status": "running"}) as post:
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id,
                "profile": "marie",
                "choice": "deny",
                "approval_id": first["delivery_id"],
            }))

        assert result["status"] == "running"
        post.assert_called_once()
        assert callback.get_pending_approval(run_id)["delivery_id"] == second["delivery_id"]
        notice = completion_queue.get_nowait()
        assert notice["approval"]["command"] == "COMMAND_B"
        assert notice["approval"]["delivery_id"] == second["delivery_id"]
        # The later SSE approval.responded event is consumed without popping B.
        assert callback._consume_local_approval_response(run_id) is True
        assert callback.get_pending_approval(run_id)["delivery_id"] == second["delivery_id"]

        # Current core reports running even though its FIFO still contains B.
        # The locally verified promoted head remains resolvable.
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "running"}), \
             patch.object(tools, "_post_json", return_value={"resolved": 1, "status": "running"}) as second_post:
            second_result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id,
                "profile": "marie",
                "choice": "deny",
                "approval_id": second["delivery_id"],
            }))
        assert second_result["status"] == "running"
        second_post.assert_called_once()
        assert callback.get_pending_approval(run_id) is None


# ---------------------------------------------------------------------------
# Stall timer suppression while awaiting approval
# ---------------------------------------------------------------------------


class TestStallSuppression:
    def test_effective_timeout_is_approval_window_while_awaiting(self):
        run_id = "run-stall-1"
        try:
            callback._set_pending_approval(run_id, {"run_id": run_id, "profile": "marie"})
            assert callback._is_awaiting_approval(run_id) is True
            assert callback._effective_stall_timeout(run_id) == callback._APPROVAL_TIMEOUT
            assert callback._APPROVAL_TIMEOUT > callback._STALL_TIMEOUT
        finally:
            callback._clear_pending_approval_mem(run_id)

    def test_effective_timeout_is_normal_when_not_awaiting(self):
        run_id = "run-stall-2"
        assert callback._is_awaiting_approval(run_id) is False
        assert callback._effective_stall_timeout(run_id) == callback._STALL_TIMEOUT


# ---------------------------------------------------------------------------
# Cancellation clears pending approval
# ---------------------------------------------------------------------------


class TestCancelClearsPending:
    def test_cancel_clears_in_memory_and_state(self):
        run_id = "run-cancel-pending"
        _set_pending(run_id, "marie")
        callback._listeners[run_id] = threading.Thread(target=lambda: None)

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_post_empty", return_value={"status": "stopping"}):
            result = json.loads(tools.handle_cancel_dispatch({
                "run_id": run_id, "profile": "marie",
            }))

        assert result["status"] == "stopping"
        assert callback.get_pending_approval(run_id) is None
        state = tools._load_state()
        run = next(r for r in state["runs"] if r["run_id"] == run_id)
        assert "pending_approval" not in run


# ---------------------------------------------------------------------------
# check_dispatch / dispatch_status surface waiting_for_approval
# ---------------------------------------------------------------------------


class TestStatusSurfacing:
    def test_check_dispatch_includes_pending_approval(self):
        run_id = "run-check-pending"
        _set_pending(run_id, "marie")

        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval", "run_id": run_id}):
            result = json.loads(tools.handle_check_dispatch({
                "run_id": run_id, "profile": "marie",
            }))

        assert result["status"] == "waiting_for_approval"
        assert result["pending_approval"]["run_id"] == run_id
        assert result["pending_approval"]["command"] == "rm -rf marie/data"

    def test_dispatch_status_lists_awaiting_approval_separately(self):
        run_id = "run-status-awaiting"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")
        tools._update_pending_approval(run_id, {
            "run_id": run_id, "profile": "marie", "command": "x",
            "description": "d", "choices": ["once"], "timestamp": 1,
            "requested_at": 1.0,
        })
        # A normal running run that is NOT awaiting approval.
        tools._persist_run("run-status-running", "marie", "preview2", model="glm-5.2")

        result = json.loads(tools.handle_dispatch_status({}))

        awaiting_ids = [r["run_id"] for r in result["awaiting_approval"]]
        assert run_id in awaiting_ids
        assert "run-status-running" not in awaiting_ids


# ---------------------------------------------------------------------------
# Recovery after restart
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_recover_pending_approvals_from_state_file(self):
        run_id = "run-recover-1"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")
        approval_data = {
            "run_id": run_id, "profile": "marie", "command": "rm -rf x",
            "description": "destructive", "choices": ["once", "session", "always", "deny"],
            "timestamp": 1, "requested_at": 1.0,
        }
        tools._update_pending_approval(run_id, approval_data)
        # Simulate a restart: in-memory dict is empty.
        with callback._pending_approvals_lock:
            callback._pending_approvals.clear()

        n = callback.recover_pending_approvals()
        assert n >= 1
        recovered = callback.get_pending_approval(run_id)
        assert recovered is not None
        assert recovered["profile"] == "marie"
        assert recovered["command"] == "rm -rf x"

    def test_recover_restores_entire_fifo(self):
        run_id = "run-recover-fifo"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")
        first = {
            "run_id": run_id,
            "profile": "marie",
            "command": "COMMAND_A",
            "delivery_id": "approval-a",
        }
        second = {
            "run_id": run_id,
            "profile": "marie",
            "command": "COMMAND_B",
            "delivery_id": "approval-b",
        }
        tools._update_pending_approval(run_id, first, [first, second])
        with callback._pending_approvals_lock:
            callback._pending_approvals.clear()
            callback._pending_approval_queues.clear()

        assert callback.recover_pending_approvals() >= 2
        recovered = callback.get_pending_approval_queue(run_id)
        assert [item["delivery_id"] for item in recovered] == [
            "approval-a", "approval-b",
        ]
        promoted = callback._advance_pending_approval(run_id, "approval-a")
        assert promoted["delivery_id"] == "approval-b"
        assert promoted["fifo_promoted"] is True

    def test_recover_enables_scope_check_after_restart(self):
        run_id = "run-recover-scope"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")
        tools._update_pending_approval(run_id, {
            "run_id": run_id, "profile": "marie", "command": "x",
            "description": "d", "choices": ["once"], "timestamp": 1, "requested_at": 1.0,
        })
        with callback._pending_approvals_lock:
            callback._pending_approvals.clear()
        callback.recover_pending_approvals()

        # After recovery, a cross-profile approve attempt must fail closed.
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json") as gj, \
             patch.object(tools, "_post_json") as pj:
            result = json.loads(tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "ada", "choice": "deny",
                "approval_id": _approval_id(run_id),
            }))
        assert result["status"] == "error"
        gj.assert_not_called()
        pj.assert_not_called()


# ---------------------------------------------------------------------------
# Credential redaction — nothing leaks into output or state
# ---------------------------------------------------------------------------


class TestNoCredentialLeak:
    def _setup_fake_registry(self):
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

    def test_api_key_not_in_approval_notification(self):
        delivered = self._setup_fake_registry()
        # The command is already redacted by the target; we preserve it. But
        # the target API key must never appear in the notification.
        approval_data = {
            "run_id": "run-leak-1", "profile": "marie",
            "command": f"do stuff with {_SECRET_IN_COMMAND}",
            "description": "needs approval",
            "choices": ["once", "session", "always", "deny"],
            "timestamp": 1, "requested_at": 1.0,
            "delivery_id": "dispatch-approval-run-leak-1-test",
        }
        callback._set_pending_approval("run-leak-1", approval_data)
        callback._deliver_approval_required(approval_data, "sess", "key")
        assert len(delivered) == 1
        evt = delivered[0]
        blob = json.dumps(evt)
        assert _API_KEY not in blob
        # The redacted command IS relayed (it's already redacted by target).
        assert _SECRET_IN_COMMAND in evt["summary"]
        assert "[DISPATCH APPROVAL REQUIRED]" in evt["summary"]
        assert evt["status"] == "approval_required"

    def test_api_key_not_in_state_file(self):
        run_id = "run-leak-2"
        tools._persist_run(run_id, "marie", "preview", model="glm-5.2")
        approval_data = {
            "run_id": run_id, "profile": "marie", "command": "redacted cmd",
            "description": "d", "choices": ["once"], "timestamp": 1,
            "requested_at": 1.0,
        }
        tools._update_pending_approval(run_id, approval_data)
        with open(_state_path()) as f:
            raw = f.read()
        assert _API_KEY not in raw

    def test_api_key_not_in_approve_dispatch_output(self):
        run_id = "run-leak-3"
        _set_pending(run_id, "marie")
        with patch.object(tools, "_resolve_profile", side_effect=_resolve_profile_ok), \
             patch.object(tools, "_get_json", return_value={"status": "waiting_for_approval"}), \
             patch.object(tools, "_post_json", return_value={"resolved": 1, "status": "running"}):
            out = tools.handle_approve_dispatch({
                "run_id": run_id, "profile": "marie", "choice": "deny",
                "approval_id": _approval_id(run_id),
            })
        assert _API_KEY not in out

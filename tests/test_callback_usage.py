"""Remote token usage is not an inference-call count."""
from queue import Queue

import pytest

from hermes_herald import callback, tools


@pytest.mark.parametrize("usage", [{"input_tokens": 200, "output_tokens": 10, "total_tokens": 210}, {}])
def test_remote_callback_marks_unavailable_call_count_without_losing_usage(monkeypatch, usage):
    recorded = []
    monkeypatch.setattr(tools, "_update_run_status", lambda *args, **kwargs: recorded.append(kwargs))
    event = callback._build_delegation_event(
        "remote-run", "reviewer", "done", usage, "marker", "session", "session",
    )
    assert event["api_calls"] == "?"
    assert event["summary"] == "done"
    assert recorded[0]["usage"] == (usage or None)


def test_remote_approval_notice_also_has_unknown_call_count(monkeypatch):
    from tools.process_registry import process_registry

    queue = Queue()
    monkeypatch.setattr(process_registry, "completion_queue", queue)
    approval = {"run_id": "remote-approval", "profile": "reviewer", "delivery_id": "approval-1"}
    monkeypatch.setattr(callback, "_pending_approvals", {"remote-approval": approval})
    monkeypatch.setattr(callback, "_approval_notice_events", {})
    monkeypatch.setattr(callback, "_approval_notice_ids_by_run", {})
    callback._deliver_approval_required(approval, "session", "session")
    event = queue.get_nowait()
    assert event["status"] == "approval_required"
    assert event["api_calls"] == "?"

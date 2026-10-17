"""Bounded handle retention, error privacy, and unchanged legacy registration."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_herald import tools
from hermes_herald.host_lifecycle import HostLifecycleAdapter, MAX_TASKS


class Lifecycle:
    def __init__(self):
        self.launched = []
        self.reads = 0
        self.fail = False

    def launch(self, request):
        self.launched.append(request)
        return {"capability": "private-test-capability", "model": "synthetic", "provider": "synthetic"}

    def status(self, handle):
        self.reads += 1
        if self.fail:
            raise RuntimeError(repr(handle))
        return {"handle": handle, "state": "FAILED"}

    def result(self, handle):
        return {"handle": handle, "terminal_state": "FAILED", "ready": True,
                "summary": None, "error_classification": "SYNTHETIC_FAILURE",
                "error_message": "synthetic failure preserved", "usage_metadata": {"api_calls": 0}}


def invoke(adapter, args, **kwargs):
    raw = adapter(args, session_id="owner", **kwargs)
    assert "private-test-capability" not in raw
    assert "capability" not in raw
    assert "handle" not in json.loads(raw)
    return json.loads(raw)


def test_bounded_table_refuses_launch_without_evicting_results():
    service = Lifecycle()
    adapter = HostLifecycleAdapter(SimpleNamespace(subagent_lifecycle=service))
    tasks = [invoke(adapter, {"goal": "synthetic"})["task_id"] for _ in range(MAX_TASKS)]
    assert len(set(tasks)) == MAX_TASKS
    refused = invoke(adapter, {"goal": "must not construct"})
    assert refused["status"] == "error" and "capacity" in refused["error"]
    assert len(service.launched) == MAX_TASKS
    result = invoke(adapter, {"action": "result", "task_id": tasks[0]})
    assert result["ready"] is True
    assert result["error_message"] == "synthetic failure preserved"
    assert invoke(adapter, {"action": "result", "task_id": tasks[0]}) == result
    assert service.reads == 2


def test_rpc_errors_do_not_expose_handles_or_discard_task():
    service = Lifecycle()
    adapter = HostLifecycleAdapter(SimpleNamespace(subagent_lifecycle=service))
    launch = invoke(adapter, {"goal": "synthetic"})
    service.fail = True
    args = {"action": "result", "task_id": launch["task_id"]}
    error = invoke(adapter, args)
    assert error["status"] == "error"
    assert "owning active session" in error["error"]
    service.fail = False
    assert invoke(adapter, args)["error_message"] == "synthetic failure preserved"
    assert len(service.launched) == 1


@pytest.mark.parametrize("args", [None, [], {"goal": ""}, {"goal": 1},
    {"goal": "x", "context": None}, {"goal": "x", "provider": ""},
    {"goal": "x", "model": ""}, {"goal": "x" * 16001},
    {"action": "result"}, {"action": "wait"}, {"action": []},
    {"action": "steer", "task_id": "x", "text": ""},
    {"action": "status", "task_id": "x", "goal": "ignored"},
])
def test_malformed_or_irrelevant_options_never_launch(args):
    service = Lifecycle()
    adapter = HostLifecycleAdapter(SimpleNamespace(subagent_lifecycle=service))
    assert invoke(adapter, args)["status"] == "error"
    assert service.launched == []


def test_missing_session_fails_before_launch():
    service = Lifecycle()
    adapter = HostLifecycleAdapter(SimpleNamespace(subagent_lifecycle=service))
    assert json.loads(adapter({"goal": "x"}))["status"] == "error"
    assert service.launched == []


def test_in_process_registration_preserves_original_handler_and_schema(monkeypatch):
    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
    path = Path(tools.__file__).with_name("__init__.py")
    spec = importlib.util.spec_from_file_location("hermes_herald.legacy_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_herald"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    registered = {}
    ctx = SimpleNamespace(register_tool=lambda **kw: registered.update({kw["name"]: kw}),
                          register_skill=lambda *a: None)
    module.register(ctx)
    delegate = registered["delegate_subagent"]
    assert delegate["handler"] is tools.handle_delegate_subagent
    assert delegate["schema"] is tools.DELEGATE_SUBAGENT_SCHEMA
    assert "action" not in delegate["schema"]["parameters"]["properties"]

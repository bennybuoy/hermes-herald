"""Bounded handles, error privacy, native registration and fail-closed compatibility."""
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


def test_missing_facade_names_public_requirement_without_legacy_advice():
    adapter = HostLifecycleAdapter(SimpleNamespace())
    error = invoke(adapter, {"goal": "x"})
    assert error["status"] == "error"
    assert "ctx.subagent_lifecycle" in error["error"]
    assert "133762" not in error["error"]
    assert "use in_process" not in error["error"]


def test_missing_session_fails_before_launch():
    service = Lifecycle()
    adapter = HostLifecycleAdapter(SimpleNamespace(subagent_lifecycle=service))
    assert json.loads(adapter({"goal": "x"}))["status"] == "error"
    assert service.launched == []


def test_in_process_registration_binds_public_adapter_and_schema(monkeypatch):
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
    from hermes_herald.native_lifecycle import NativeLifecycleAdapter
    assert isinstance(delegate["handler"], NativeLifecycleAdapter)
    assert delegate["schema"] is tools.DELEGATE_SUBAGENT_SCHEMA
    assert "action" in delegate["schema"]["parameters"]["properties"]
    result = json.loads(delegate["handler"]({"goal": "missing facade"}, session_id="owner"))
    assert result["status"] == "error"
    assert "ctx.subagent_lifecycle" in result["error"]

@pytest.mark.parametrize("adapter_kind", ["host", "native"])
def test_denial_never_falls_back_or_relaunches(adapter_kind):
    from hermes_herald.native_lifecycle import NativeLifecycleAdapter
    class Denied:
        calls = 0
        def launch(self, request):
            self.calls += 1
            raise ValueError("private-test-capability must not escape")
    service = Denied()
    cls = HostLifecycleAdapter if adapter_kind == "host" else NativeLifecycleAdapter
    result = invoke(cls(SimpleNamespace(subagent_lifecycle=service)), {"goal": "denied"})
    assert result["status"] == "error"
    assert service.calls == 1


def test_pre_provider_host_refuses_even_an_unpinned_launch():
    class OldService:
        def launch(self, request):
            assert "provider" in request  # old host must not silently accept a reduced request
            raise ValueError("Unknown request fields: provider")
    result = invoke(HostLifecycleAdapter(SimpleNamespace(subagent_lifecycle=OldService())), {"goal": "old host"})
    assert result["status"] == "error"
    assert "per-launch provider support" in result["error"]


def test_native_stateless_session_refuses_before_launch(monkeypatch):
    from hermes_herald.native_lifecycle import NativeLifecycleAdapter
    service = Lifecycle()
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: False)
    result = invoke(NativeLifecycleAdapter(SimpleNamespace(subagent_lifecycle=service)), {"goal": "x"})
    assert result["status"] == "error"
    assert "cannot receive detached results" in result["error"]
    assert not service.launched


def test_accepted_launch_survives_state_and_observer_start_failures(monkeypatch):
    from hermes_herald.native_lifecycle import NativeLifecycleAdapter
    import threading
    service = Lifecycle()
    def fail(*a, **kw):
        raise RuntimeError("private-test-capability")
    monkeypatch.setattr(tools, "_save_state", fail)
    monkeypatch.setattr(threading.Thread, "start", fail)
    result = invoke(NativeLifecycleAdapter(SimpleNamespace(subagent_lifecycle=service)), {"goal": "x"})
    assert result["status"] == "dispatched"
    assert result["delivery"] == "poll"
    assert result["next_action"]["task_id"] == result["task_id"]
    assert "persistence_warning" in result
    assert len(service.launched) == 1
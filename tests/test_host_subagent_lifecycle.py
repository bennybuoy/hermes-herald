"""Actual Herald registration across the Hermes host RPC boundary; no inference."""
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from agent.subagent_lifecycle import bind_subagent_parent
from hermes_cli.plugin_host import PluginHost
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, LoadedPlugin
from tools.registry import registry

PLUGIN = Path(os.environ["HERMES_HERALD_PLUGIN_DIR"])


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    import tools.delegate_tool as delegate
    import agent.subagent_lifecycle as lifecycle
    import tools.delegate_tool_registry as child_registry

    built, routes, controls = [], [], []
    release = threading.Event()
    done = threading.Event()
    outcome = {"status": "completed", "summary": "synthetic final result", "api_calls": 0,
               "duration_seconds": 0}

    def credentials(config, parent):
        routes.append(dict(config))
        return dict(model=config.get("model") or "configured-model", provider="synthetic",
                    api_key="synthetic-not-a-credential", base_url="http://invalid",
                    api_mode="chat_completions")

    def build(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(_subagent_id=f"synthetic-{tmp_path.name}-{len(built)}",
                               _delegate_role="leaf", _delegate_depth=1,
                               model=kwargs["model"], provider="synthetic", session_id="child")

    def run(*args, **kwargs):
        assert release.wait(5), "synthetic child was not released"
        done.set()
        return dict(outcome)

    monkeypatch.setattr(delegate, "_resolve_delegation_credentials", credentials)
    monkeypatch.setattr(delegate, "_credential_overrides", lambda *a: {})
    monkeypatch.setattr(delegate, "_build_child_agent", build)
    monkeypatch.setattr(delegate, "_run_single_child", run)
    monkeypatch.setattr(delegate, "_load_config", lambda: {"model": "configured-model"})
    monkeypatch.setattr(lifecycle, "request_hard_interrupt",
                        lambda *a, **kw: controls.append(("cancel", kw)) or True)
    monkeypatch.setattr(child_registry, "steer_subagent",
                        lambda sid, text: controls.append(("steer", text)) or True)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({"plugins": {"isolation": "host"},
        "hermes_herald": {"state_file": str(home / "runs.json")}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(home / "managed-empty"))
    manager = PluginManager(scope_key=str(home))
    host = PluginHost(manager)
    manifest = PluginManifest(name="hermes-herald", source="user", path=str(PLUGIN))
    manager._plugins[manifest.name] = LoadedPlugin(manifest=manifest, enabled=True)
    ctx = PluginContext(manifest, manager)
    try:
        host.load(manifest, ctx, module_name="hermes_plugins.herald_lifecycle", entrypoint=False)
        assert host.alive and host.pid != os.getpid()
        parent = SimpleNamespace(session_id="owner", enabled_toolsets=["file"])
        yield SimpleNamespace(manager=manager, ctx=ctx, parent=parent, built=built,
                              routes=routes, controls=controls, release=release, done=done,
                              outcome=outcome)
    finally:
        release.set()
        manager.unload()
        process = host._proc
        host.shutdown()
        assert process is None or process.poll() == 0


def call(h, args, parent=None, session_id=None):
    parent = parent or h.parent
    with bind_subagent_parent(parent):
        raw = registry.dispatch("delegate_subagent", args, scope=h.manager.scope_key,
                                session_id=session_id or parent.session_id)
    assert "capability" not in raw
    return json.loads(raw)


def test_launch_returns_then_same_parent_can_poll_and_retrieve(hosted):
    h = hosted
    launch = call(h, {"goal": "synthetic task", "context": "explicit brief",
                      "model": "exact-model", "provider": "synthetic"})
    assert launch["status"] == "dispatched", launch
    assert launch["delivery"] == "poll"
    task = launch["task_id"]
    assert task.startswith("herald-host-")
    assert launch["next_action"] == {"action": "result", "task_id": task}
    assert "handle" not in launch and "subagent_id" not in launch
    # Separate invocations, not background RPC after the launch call returns.
    status = call(h, {"action": "status", "task_id": task})
    assert status["state"] in {"PENDING", "RUNNING"}
    pending = call(h, {"action": "result", "task_id": task})
    assert pending["ready"] is False
    assert pending["next_action"] == launch["next_action"]
    h.release.set()
    # Synchronize synthetic execution without adding a public wait action.
    assert h.done.wait(3)
    # done precedes the terminal snapshot; bounded foreground polling in the test only.
    import time
    deadline = time.monotonic() + 3
    while True:
        result = call(h, {"action": "result", "task_id": task})
        if result.get("ready") or time.monotonic() >= deadline:
            break
        threading.Event().wait(0.01)
    assert result["ready"] is True, result
    assert result["terminal_state"] == "SUCCEEDED"
    assert result["summary"] == "synthetic final result"
    assert result["usage_metadata"] == {"api_calls": 0}
    assert call(h, {"action": "result", "task_id": task}) == result
    assert call(h, {"action": "steer", "task_id": task, "text": "too late"})["accepted"] is False
    assert call(h, {"action": "cancel", "task_id": task})["already_terminal"] is True
    assert h.built[0]["parent_agent"] is h.parent
    assert h.built[0]["context"] == "explicit brief"
    assert h.built[0]["toolsets"] is None
    assert h.routes == [{"provider": "synthetic", "model": "exact-model"}]


@pytest.mark.parametrize("action", ["status", "result", "cancel", "steer"])
def test_foreign_parent_rejected_even_with_forged_session_kwarg(hosted, action):
    h = hosted
    launch = call(h, {"goal": "synthetic"})
    assert launch["status"] == "dispatched", launch
    args = {"action": action, "task_id": launch["task_id"]}
    if action == "steer":
        args["text"] = "do not cross sessions"
    foreign = SimpleNamespace(session_id="foreign", enabled_toolsets=["file"])
    for session in ("foreign", "owner"):
        result = call(h, args, parent=foreign, session_id=session)
        assert result["status"] == "error", result
        assert "summary" not in result
    assert h.controls == []


@pytest.mark.parametrize("options", [
    {"toolsets": []}, {"toolsets": ["file"]}, {"allowed_toolsets": []},
    {"inherit_toolsets": False}, {"inherit_toolsets": "false"},
    {"stall_timeout_seconds": 60}, {"interrupt_after_seconds": 60},
    {"timeout_seconds": 60}, {"hard_timeout_seconds": 60},
    {"inherit_context": True}, {"inherit_context": False},
    {"inherit_soul": True}, {"inherit_soul": False},
    {"reasoning_effort": "low"}, {"reasoning_effort": False},
    {"role": "orchestrator"}, {"working_directory": "/ignored"},
    {"blocked_tools": ["terminal"]}, {"unexpected_option": 1},
])
def test_unsupported_options_fail_before_child_construction(hosted, options):
    result = call(hosted, {"goal": "must not launch", **options})
    assert result["status"] == "error", result
    assert next(iter(options)) in result["error"]
    assert "in_process" in result["error"]
    assert hosted.built == []


def test_controls_and_default_delegation_route(hosted):
    h = hosted
    launch = call(h, {"action": "launch", "goal": "synthetic", "inherit_toolsets": True})
    assert launch["status"] == "dispatched", launch
    args = {"task_id": launch["task_id"]}
    assert call(h, {**args, "action": "steer", "text": "updated brief"})["accepted"] is True
    assert call(h, {**args, "action": "cancel", "reason": "stop synthetic"})["accepted"] is True
    assert h.controls[0] == ("steer", "updated brief")
    assert h.controls[1][0] == "cancel"
    assert h.routes == [{"model": "configured-model"}]


def test_registered_host_schema_is_polled_not_legacy(hosted):
    schema = registry.get_entry("delegate_subagent", scope=hosted.manager.scope_key).schema
    assert "poll" in schema["description"]
    assert "auto-delivered" not in schema["description"]
    props = schema["parameters"]["properties"]
    assert props["action"]["enum"] == ["launch", "status", "result", "cancel", "steer"]
    assert "provider" in props
    assert "toolsets" not in props and "reasoning_effort" not in props
    assert schema["parameters"]["additionalProperties"] is False


def test_failure_result_survives_tool_return(hosted):
    h = hosted
    h.outcome.update(status="error", summary=None, error="synthetic child failure")
    launch = call(h, {"goal": "synthetic failing task"})
    assert launch["status"] == "dispatched", launch
    h.release.set()
    assert h.done.wait(3)
    import time
    deadline = time.monotonic() + 3
    args = {"action": "result", "task_id": launch["task_id"]}
    while True:
        result = call(h, args)
        if result.get("ready") or time.monotonic() >= deadline:
            break
        threading.Event().wait(0.01)
    assert result["ready"] is True, result
    assert result["terminal_state"] == "FAILED"
    assert result["error_message"] == "synthetic child failure"
    assert result["error_classification"] == "ERROR"
    assert call(h, args) == result


def test_launch_rejects_mismatched_or_absent_active_parent(hosted):
    h = hosted
    result = call(h, {"goal": "must not construct"}, session_id="not-owner")
    assert result["status"] == "error", result
    raw = registry.dispatch("delegate_subagent", {"goal": "must not construct"},
                            scope=h.manager.scope_key, session_id="owner")
    assert json.loads(raw)["status"] == "error"
    assert h.built == []

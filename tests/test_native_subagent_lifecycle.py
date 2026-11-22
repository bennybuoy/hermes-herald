"""Registered native lifecycle integration; controlled execution, no network."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

lifecycle_api = pytest.importorskip("agent.subagent_lifecycle")
if "provider" not in lifecycle_api.SubagentLaunchRequest.__dataclass_fields__:
    pytest.skip("Host lacks per-launch provider lifecycle support; refusal covered separately", allow_module_level=True)

from hermes_herald import tools


def test_native_registration_launches_through_public_facade(monkeypatch, tmp_path):
    from agent.subagent_lifecycle import bind_subagent_parent
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry
    import tools.delegate_tool as delegate
    from tools.process_registry import process_registry

    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = Path(tools.__file__).parent
    spec = importlib.util.spec_from_file_location("hermes_herald.native_entry", plugin / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_herald"
    spec.loader.exec_module(module)
    manager = PluginManager(scope_key=str(tmp_path))
    ctx = PluginContext(PluginManifest(name="hermes-herald", source="user", path=str(plugin)), manager)
    release = threading.Event()
    built = []

    def build(**kw):
        built.append(kw)
        return SimpleNamespace(_subagent_id=f"native-{tmp_path.name}", _delegate_depth=1,
                               _delegate_role="leaf", model=kw["model"], provider="synthetic")

    def run(*a, **kw):
        assert release.wait(5)
        return {"status": "completed", "summary": "controlled child output", "api_calls": 3}

    monkeypatch.setattr(delegate, "_build_child_agent", build)
    monkeypatch.setattr(delegate, "_run_single_child", run)
    try:
        import tools.delegate_tool_config as delegate_config
    except ImportError:
        delegate_config = None
    for mod in (delegate_config, delegate):
        if mod is None:
            continue
        if hasattr(mod, "_resolve_delegation_credentials"):
            monkeypatch.setattr(mod, "_resolve_delegation_credentials", lambda *a: {
                "model": "configured", "provider": None, "base_url": None, "api_key": None, "api_mode": None})
        if hasattr(mod, "_load_config"):
            monkeypatch.setattr(mod, "_load_config", lambda: {})
    parent = SimpleNamespace(session_id="owner", enabled_toolsets=["file"])
    try:
        module.register(ctx)
        with bind_subagent_parent(parent):
            response = json.loads(registry.dispatch("delegate_subagent", {"goal": "bounded task", "model": "exact"},
                                                  scope=manager.scope_key, session_id="owner"))
        assert response["status"] == "dispatched", response
        assert response["delivery"] == "callback"
        assert built[0]["parent_agent"] is parent
        assert built[0]["model"] == "exact"
        release.set()
        event = process_registry.completion_queue.get(timeout=5)
        assert event["delegation_id"] == response["task_id"]
        assert event["parent_session_id"] == "owner"
        assert event["summary"] == "controlled child output"
        assert event["api_calls"] == 3
        assert process_registry.completion_queue.empty()
    finally:
        release.set()
        manager.unload()

def test_concurrent_children_and_observers_keep_profile_and_session_scope(monkeypatch, tmp_path):
    import queue
    from agent.subagent_lifecycle import bind_subagent_parent
    from agent.secret_scope import set_secret_scope, reset_secret_scope, get_secret
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override, get_hermes_home
    from gateway.session_context import set_session_vars, clear_session_vars
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_herald.native_lifecycle import NativeLifecycleAdapter
    from tools.process_registry import process_registry
    import tools.delegate_tool as delegate

    release = threading.Event()
    notifications = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", notifications)
    built, observed = [], []
    def build(**kw):
        built.append(kw)
        return SimpleNamespace(_subagent_id=f"scoped-{tmp_path.name}-{len(built)}",
                               _delegate_role="leaf", _delegate_depth=1, model="scoped", provider="synthetic")
    def run(index, goal, child, parent):
        assert release.wait(5)
        observed.append((parent.session_id, str(get_hermes_home()), get_secret("SYNTHETIC_SECRET")))
        return {"status": "completed", "summary": parent.session_id}
    monkeypatch.setattr(delegate, "_build_child_agent", build)
    monkeypatch.setattr(delegate, "_run_single_child", run)
    try:
        import tools.delegate_tool_config as delegate_config
    except ImportError:
        delegate_config = None
    for mod in (delegate_config, delegate):
        if mod is None:
            continue
        if hasattr(mod, "_resolve_delegation_credentials"):
            monkeypatch.setattr(mod, "_resolve_delegation_credentials", lambda *a: {
                "model": "scoped", "provider": None, "base_url": None, "api_key": None, "api_mode": None})
        if hasattr(mod, "_load_config"):
            monkeypatch.setattr(mod, "_load_config", lambda: {})
    parents = [SimpleNamespace(session_id=f"owner-{name}", enabled_toolsets=[]) for name in ("a", "b")]
    managers = []
    expected = []
    tasks = {}
    try:
        for parent in parents:
            name = parent.session_id[-1]
            home = tmp_path / name
            home.mkdir()
            (home / "config.yaml").write_text("{}")
            ht = set_hermes_home_override(home)
            st = set_secret_scope({"SYNTHETIC_SECRET": f"secret-{name}"})
            rt = set_session_vars(source="gateway", session_id=parent.session_id, session_key=f"chat-{name}")
            try:
                manager = PluginManager(scope_key=str(home))
                managers.append(manager)
                ctx = PluginContext(PluginManifest(name="hermes-herald"), manager)
                with bind_subagent_parent(parent):
                    response = json.loads(NativeLifecycleAdapter(ctx)({"goal": "scope"}, session_id=parent.session_id))
                assert response["status"] == "dispatched", response
                tasks[parent.session_id] = response["task_id"]
                expected.append((parent.session_id, str(home), f"secret-{name}"))
            finally:
                clear_session_vars(rt)
                reset_secret_scope(st)
                reset_hermes_home_override(ht)
        release.set()
        events = [notifications.get(timeout=5) for _ in parents]
        assert sorted(observed) == sorted(expected)
        assert {e["parent_session_id"] for e in events} == set(tasks)
        for event in events:
            owner = event["parent_session_id"]
            assert event["delegation_id"] == tasks[owner]
            assert event["summary"] == owner
            state = json.loads((tmp_path / owner[-1] / "hermes-herald-runs.json").read_text())
            assert [r["run_id"] for r in state["runs"]] == [tasks[owner]]
            assert state["runs"][0]["status"] == "completed"
        assert notifications.empty()
    finally:
        release.set()
        for manager in managers:
            manager.unload()

"""Schema/handler contract tests for delegate_subagent."""

import threading
import time
from types import SimpleNamespace

import pytest

from hermes_herald import tools


def test_delegate_subagent_schema_documents_conditional_return_contract():
    description = tools.DELEGATE_SUBAGENT_SCHEMA["description"]
    properties = tools.DELEGATE_SUBAGENT_SCHEMA["parameters"]["properties"]

    # Async-capable sessions: immediate task metadata + auto-delivery.
    assert "task_id" in description
    assert "auto-delivered" in description
    # Supported API runs without detached delivery: blocking terminal payload.
    assert "block this tool call" in description
    assert "duration_seconds} payload in-turn" in description
    # Fail-closed exclusions stay documented.
    assert "/v1/chat/completions" in description
    assert "fail closed" in description
    # Parameter contract unchanged.
    assert properties["inherit_soul"]["type"] == "boolean"
    assert properties["inherit_soul"]["default"] is False
    assert "full SOUL.md" in properties["inherit_soul"]["description"]
    assert properties["inherit_context"]["type"] == "boolean"
    assert properties["inherit_context"]["default"] is False
    assert "user/assistant" in properties["inherit_context"]["description"]
    assert properties["inherit_toolsets"]["type"] == "boolean"
    assert properties["inherit_toolsets"]["default"] is True
    assert "model-only" in properties["inherit_toolsets"]["description"]
    assert "hard_timeout_seconds" not in properties
    assert properties["interrupt_after_seconds"]["minimum"] == 30
    assert "cooperative" in properties["interrupt_after_seconds"]["description"]


def test_async_capable_session_returns_dispatched_and_queues_completion(monkeypatch):
    import tools.delegate_tool as delegate_tool
    import tools.process_registry as process_registry

    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)

    class FakeQueue:
        def __init__(self):
            self.events = []

        def put(self, evt):
            self.events.append(evt)

    queue = FakeQueue()
    monkeypatch.setattr(
        process_registry.process_registry, "completion_queue", queue
    )

    parent = SimpleNamespace(model="parent-model", _session_messages=[])
    child = SimpleNamespace(tool_progress_callback=None)
    monkeypatch.setattr(
        delegate_tool, "_build_child_agent", lambda **kwargs: child
    )

    def fake_run_single_child(**kwargs):
        assert kwargs.get("child") is child
        return {"status": "completed", "summary": "async child summary"}

    monkeypatch.setattr(delegate_tool, "_run_single_child", fake_run_single_child)

    monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
    monkeypatch.setattr(tools, "_save_state", lambda state: None)
    updates = []
    finished = threading.Event()

    def fake_update_status(*args, **kwargs):
        updates.append((args, kwargs))
        finished.set()

    monkeypatch.setattr(tools, "_update_run_status", fake_update_status)
    monkeypatch.setattr(
        tools,
        "_capture_subagent_routing",
        lambda parent_agent=None: {
            "session_id": "context-session",
            "session_key": "context-key",
            "origin_ui_session_id": "ui-session-9",
        },
    )

    result = tools.json.loads(tools.handle_delegate_subagent(
        {"goal": "audit that"}, parent_agent=parent,
    ))

    # Immediate return carries task metadata, not the child's terminal summary.
    assert result["status"] == "dispatched"
    assert result["task_id"].startswith("subagent-")
    assert "summary" not in result
    assert "will be delivered" in result["message"].lower()

    # The final result is auto-delivered as a new queued message.
    assert finished.wait(timeout=5)
    assert [evt["type"] for evt in queue.events] == ["async_delegation"]
    evt = queue.events[0]
    assert evt["delegation_id"] == result["task_id"]
    assert evt["status"] == "completed"
    assert evt["summary"] == "async child summary"
    assert evt["session_id"] == "context-session"
    assert updates and updates[0][1].get("status") == "completed"


def test_in_turn_child_runs_on_caller_thread_when_async_delivery_off(monkeypatch):
    import tools.delegate_tool as delegate_tool
    import tools.process_registry as process_registry

    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: False)

    class FakeQueue:
        def __init__(self):
            self.events = []

        def put(self, evt):
            self.events.append(evt)

    queue = FakeQueue()
    monkeypatch.setattr(
        process_registry.process_registry, "completion_queue", queue
    )

    parent = SimpleNamespace(model="parent-model", _session_messages=[])
    child = SimpleNamespace(tool_progress_callback=None)
    monkeypatch.setattr(
        delegate_tool, "_build_child_agent", lambda **kwargs: child
    )

    def fake_run_single_child(**kwargs):
        assert kwargs.get("child") is child
        return {"status": "completed", "summary": "child summary text"}

    monkeypatch.setattr(delegate_tool, "_run_single_child", fake_run_single_child)

    monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
    monkeypatch.setattr(tools, "_save_state", lambda state: None)
    updates = []
    monkeypatch.setattr(
        tools,
        "_update_run_status",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tools,
        "_capture_subagent_routing",
        lambda parent_agent=None: {
            "session_id": "context-session",
            "session_key": "context-key",
            "origin_ui_session_id": "ui-session-9",
        },
    )

    result = tools.json.loads(tools.handle_delegate_subagent(
        {"goal": "audit this"}, parent_agent=parent,
    ))

    assert result["status"] == "completed"
    assert result["summary"] == "child summary text"
    assert result["task_id"].startswith("subagent-")
    # No detached delivery: nothing may be queued behind the HTTP turn.
    assert queue.events == []
    assert updates and updates[0][1].get("status") == "completed"


def test_in_turn_child_fails_closed_without_parent(monkeypatch):
    import gateway.session_context as session_context
    import hermes_cli.plugins as plugins

    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: False)
    monkeypatch.setattr(
        session_context, "get_session_env", lambda key, default="": default
    )
    monkeypatch.setattr(
        plugins, "get_plugin_manager", lambda: SimpleNamespace(_cli_ref=None)
    )

    result = tools.json.loads(tools.handle_delegate_subagent({"goal": "x"}))
    assert result["status"] == "error"
    assert "parent agent context" in result["error"]


def test_delivery_capability_probe_fails_closed_on_core_error(monkeypatch):
    import gateway.session_context as session_context

    def broken_probe():
        raise RuntimeError("broken context")

    monkeypatch.setattr(session_context, "async_delivery_supported", broken_probe)
    assert tools._async_delivery_supported() is False


def test_parent_agent_resolves_from_exact_tui_session(monkeypatch):
    import gateway.session_context as session_context
    import hermes_cli.plugins as plugins
    import tui_gateway.server as tui_server

    expected = SimpleNamespace(session_id="durable-session")
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": (
            "tui-session-1" if key == "HERMES_UI_SESSION_ID" else default
        ),
    )
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(_cli_ref=None),
    )
    monkeypatch.setattr(
        tui_server,
        "_sessions",
        {"tui-session-1": {"agent": expected}},
    )

    assert tools._resolve_parent_agent(None) is expected


def test_parent_agent_resolves_from_exact_api_run_id(monkeypatch):
    import gateway.session_context as session_context
    import hermes_cli.plugins as plugins
    import gateway.config as gateway_config
    import gateway.run as gateway_run
    import tui_gateway.server as tui_server

    expected = SimpleNamespace(session_id="durable-session")
    unrelated = SimpleNamespace(session_id="unrelated")

    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": (
            "api_server" if key == "HERMES_SESSION_PLATFORM" else default
        ),
    )
    monkeypatch.setattr(
        plugins, "get_plugin_manager", lambda: SimpleNamespace(_cli_ref=None)
    )
    monkeypatch.setattr(
        tui_server,
        "_sessions",
        {"tui-session-1": {"agent": SimpleNamespace(session_id="other")}},
    )

    class FakeAdapter:
        _active_run_agents = {
            "run-abc": expected,
            "run-other": unrelated,
        }

    class FakeRunner:
        adapters = {gateway_config.Platform.API_SERVER: FakeAdapter()}

    monkeypatch.setattr(
        gateway_run, "_gateway_runner_ref", lambda: FakeRunner()
    )

    # Exact run_id in the adapter registry wins; the TUI path is skipped.
    assert tools._resolve_parent_agent(None, "run-abc") is expected
    # Unknown keys fail closed without scanning other runs.
    assert tools._resolve_parent_agent(None, "run-missing") is None
    assert tools._resolve_parent_agent(None, "run-other") is unrelated
    # An empty session_id resolves to no parent.
    assert tools._resolve_parent_agent(None, "") is None
    assert tools._resolve_parent_agent(None, "   ") is None


def test_parent_agent_api_lookup_fails_closed_without_runner_or_adapter(monkeypatch):
    import gateway.session_context as session_context
    import hermes_cli.plugins as plugins
    import gateway.config as gateway_config
    import gateway.run as gateway_run
    import tui_gateway.server as tui_server

    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": (
            "api_server" if key == "HERMES_SESSION_PLATFORM" else default
        ),
    )
    monkeypatch.setattr(
        plugins, "get_plugin_manager", lambda: SimpleNamespace(_cli_ref=None)
    )
    monkeypatch.setattr(tui_server, "_sessions", {})

    class EmptyAdapter:
        _active_run_agents = {"run-abc": SimpleNamespace()}

    live_runner = SimpleNamespace(
        adapters={gateway_config.Platform.API_SERVER: EmptyAdapter()}
    )

    # No runner (weakref dead) → no parent.
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: None)
    assert tools._resolve_parent_agent(None, "run-abc") is None

    # Runner without the API-server adapter → no parent.
    monkeypatch.setattr(
        gateway_run, "_gateway_runner_ref", lambda: SimpleNamespace(adapters={})
    )
    assert tools._resolve_parent_agent(None, "run-abc") is None

    # Adapter present but the exact key is missing → no parent, no scan.
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: live_runner)
    assert tools._resolve_parent_agent(None, "not-registered") is None


def test_parent_agent_resolution_is_exact_and_fails_closed(monkeypatch):
    import gateway.session_context as session_context
    import hermes_cli.plugins as plugins
    import tui_gateway.server as tui_server

    unrelated = SimpleNamespace(session_id="other")
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": (
            "missing-tui-session" if key == "HERMES_UI_SESSION_ID" else default
        ),
    )
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(_cli_ref=None),
    )
    monkeypatch.setattr(
        tui_server,
        "_sessions",
        {"unrelated-session": {"agent": unrelated}},
    )

    assert tools._resolve_parent_agent(None) is None
    explicit = SimpleNamespace(session_id="explicit")
    assert tools._resolve_parent_agent(explicit) is explicit


def test_parent_agent_resolves_from_exact_durable_session_id(monkeypatch):
    import gateway.session_context as session_context
    import hermes_cli.plugins as plugins
    import tui_gateway.server as tui_server

    expected = SimpleNamespace(session_id="durable-session")
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": default,
    )
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(_cli_ref=None),
    )
    monkeypatch.setattr(
        tui_server,
        "_sessions",
        {
            "ui-session-1": {
                "agent": expected,
                "session_key": "durable-session",
            }
        },
    )

    assert tools._resolve_parent_agent(None, "durable-session") is expected


def test_parent_agent_durable_session_resolution_rejects_ambiguity(monkeypatch):
    import gateway.session_context as session_context
    import hermes_cli.plugins as plugins
    import tui_gateway.server as tui_server

    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": default,
    )
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(_cli_ref=None),
    )
    monkeypatch.setattr(
        tui_server,
        "_sessions",
        {
            "ui-session-1": {
                "agent": SimpleNamespace(session_id="durable-session"),
                "session_key": "durable-session",
            },
            "ui-session-2": {
                "agent": SimpleNamespace(session_id="durable-session"),
                "session_key": "durable-session",
            },
        },
    )

    assert tools._resolve_parent_agent(None, "durable-session") is None


def test_subagent_routing_uses_task_local_capture_not_stale_environment(monkeypatch):
    from hermes_herald import callback

    monkeypatch.setenv("HERMES_SESSION_ID", "stale-process-session")
    monkeypatch.setenv("HERMES_SESSION_KEY", "stale-process-key")
    monkeypatch.setattr(
        callback,
        "capture_session_routing",
        lambda parent_agent=None: {
            "session_id": "context-session",
            "session_key": "context-key",
            "origin_ui_session_id": "ui-session-9",
        },
    )

    routing = tools._capture_subagent_routing(SimpleNamespace(session_id="parent"))

    assert routing == {
        "session_id": "context-session",
        "session_key": "context-key",
        "origin_ui_session_id": "ui-session-9",
    }


def test_timeout_policy_parses_cooperative_interrupt_name():
    stall, interrupt = tools._parse_subagent_timeout_policy(
        {"stall_timeout_seconds": 45, "interrupt_after_seconds": 90},
        core_timeout_seconds=None,
    )
    assert (stall, interrupt) == (45.0, 90.0)
    with pytest.raises(ValueError, match="interrupt_after_seconds"):
        tools._parse_subagent_timeout_policy(
            {"interrupt_after_seconds": 29},
            core_timeout_seconds=None,
        )


def test_interrupt_threshold_is_cooperative_and_reports_interrupt_kind():
    release = threading.Event()
    interrupted = threading.Event()
    child = SimpleNamespace(
        tool_progress_callback=None,
        interrupt=interrupted.set,
    )

    started = time.monotonic()
    with pytest.raises(tools._SubagentPolicyTimeout) as caught:
        tools._run_child_with_timeout_policy(
            child=child,
            run_child=lambda: release.wait(2),
            stall_timeout_seconds=2,
            interrupt_after_seconds=0.05,
            poll_interval_seconds=0.01,
        )
    elapsed = time.monotonic() - started
    release.set()

    assert caught.value.kind == "interrupt"
    assert interrupted.is_set()
    assert elapsed < 0.5
    error, kind = tools._describe_subagent_error(caught.value)
    assert kind == "interrupt"
    assert "cooperative interrupt threshold" in error


def test_timeout_policy_worker_inherits_caller_contextvars(monkeypatch, tmp_path):
    import hermes_constants
    import gateway.session_context as session_context

    # Keep the fallback deterministic: no ambient session identity in os.environ.
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

    expected_home = str(tmp_path / "profile-home-a")
    observed = {}

    def run_child():
        observed["home"] = hermes_constants.get_hermes_home_override()
        observed["platform"] = session_context.get_session_env(
            "HERMES_SESSION_PLATFORM"
        )
        observed["session_id"] = session_context.get_session_env(
            "HERMES_SESSION_ID"
        )
        return {"status": "completed", "summary": "ok"}

    child = SimpleNamespace(tool_progress_callback=None)
    home_token = hermes_constants.set_hermes_home_override(expected_home)
    session_context.set_session_vars(
        platform="api_server", session_id="run-ctx-a", async_delivery=False,
    )
    try:
        result = tools._run_child_with_timeout_policy(
            child=child,
            run_child=run_child,
            stall_timeout_seconds=600,
            interrupt_after_seconds=None,
            poll_interval_seconds=0.01,
        )
    finally:
        hermes_constants.reset_hermes_home_override(home_token)
        session_context.reset_session_vars()

    assert result == {"status": "completed", "summary": "ok"}
    assert observed["home"] == expected_home
    assert observed["platform"] == "api_server"
    assert observed["session_id"] == "run-ctx-a"


def test_timeout_policy_worker_keeps_concurrent_contexts_distinct(tmp_path):
    import hermes_constants
    import gateway.session_context as session_context

    barrier = threading.Barrier(2)
    results = {}

    def scenario(index):
        expected_home = str(tmp_path / f"profile-home-{index}")
        expected_session = f"run-ctx-{index}"
        observed = {}

        def run_child():
            observed["home"] = hermes_constants.get_hermes_home_override()
            observed["session_id"] = session_context.get_session_env(
                "HERMES_SESSION_ID"
            )
            time.sleep(0.05)  # keep the two workers' windows overlapped
            return {"status": "completed", "summary": f"child-{index}"}

        child = SimpleNamespace(tool_progress_callback=None)
        home_token = hermes_constants.set_hermes_home_override(expected_home)
        session_context.set_session_vars(session_id=expected_session)
        try:
            barrier.wait(timeout=5)
            result = tools._run_child_with_timeout_policy(
                child=child,
                run_child=run_child,
                stall_timeout_seconds=600,
                interrupt_after_seconds=None,
                poll_interval_seconds=0.01,
            )
        finally:
            hermes_constants.reset_hermes_home_override(home_token)
            session_context.reset_session_vars()
        results[index] = (observed, result)

    threads = [threading.Thread(target=scenario, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert set(results) == {0, 1}
    for index, (observed, result) in results.items():
        assert result == {"status": "completed", "summary": f"child-{index}"}
        assert observed["home"] == str(tmp_path / f"profile-home-{index}")
        assert observed["session_id"] == f"run-ctx-{index}"


def test_subagent_api_call_count_uses_result_and_live_child_activity():
    child = SimpleNamespace(
        get_activity_summary=lambda: {"api_call_count": 46}
    )

    assert tools._subagent_api_call_count({"api_calls": 2}, child) == 46
    assert tools._subagent_api_call_count(None, child) == 46
    assert tools._subagent_api_call_count({"api_calls": 2}, None) == 2
    assert tools._subagent_api_call_count(None, None) == 0


def test_soul_inheritance_is_opt_in_and_invalidates_cached_prompt():
    child = SimpleNamespace(load_soul_identity=True, _cached_system_prompt="stale")

    tools._apply_soul_inheritance(child, False)
    assert child.load_soul_identity is False
    assert child._cached_system_prompt is None

    child._cached_system_prompt = "stale-again"
    tools._apply_soul_inheritance(child, True)
    assert child.load_soul_identity is True
    assert child._cached_system_prompt is None


def test_reasoning_effort_parses_levels_none_and_rejects_unknown():
    assert tools._parse_subagent_reasoning_effort(None) is None
    assert tools._parse_subagent_reasoning_effort("  LOW ") == {
        "enabled": True, "effort": "low",
    }
    assert tools._parse_subagent_reasoning_effort("none") == {"enabled": False}
    for level in ("minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
        assert tools._parse_subagent_reasoning_effort(level) == {
            "enabled": True, "effort": level,
        }
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        tools._parse_subagent_reasoning_effort("maximum")
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        tools._parse_subagent_reasoning_effort(True)
    # YAML-false-style argument disables thinking rather than coercing away.
    assert tools._parse_subagent_reasoning_effort(False) == {"enabled": False}


def test_reasoning_effort_applies_after_build_and_noop_when_omitted():
    child = SimpleNamespace(reasoning_config={"enabled": True, "effort": "high"})
    tools._apply_reasoning_effort(child, None)
    assert child.reasoning_config == {"enabled": True, "effort": "high"}

    tools._apply_reasoning_effort(child, {"enabled": True, "effort": "minimal"})
    assert child.reasoning_config == {"enabled": True, "effort": "minimal"}

    tools._apply_reasoning_effort(child, {"enabled": False})
    assert child.reasoning_config == {"enabled": False}
    parsed = {"enabled": True, "effort": "low"}
    tools._apply_reasoning_effort(child, parsed)
    parsed["effort"] = "ultra"
    assert child.reasoning_config == {"enabled": True, "effort": "low"}


def test_subagent_schema_exposes_reasoning_effort_contract():
    properties = tools.DELEGATE_SUBAGENT_SCHEMA["parameters"]["properties"]
    assert properties["reasoning_effort"]["enum"] == [
        "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "none",
    ]
    assert "delegation.reasoning_effort" in properties["reasoning_effort"]["description"]
    description = tools.DELEGATE_SUBAGENT_SCHEMA["description"]
    assert "reasoning_effort" in description


def test_parent_context_inheritance_is_opt_in_bounded_and_excludes_tools():
    parent = SimpleNamespace(_session_messages=[
        {"role": "system", "content": "hidden system"},
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "secret"}]},
        {"role": "tool", "content": "sensitive tool output"},
        {"role": "user", "content": "follow-up"},
    ])

    assert tools._compose_subagent_context(parent, "explicit facts", False) == "explicit facts"
    inherited = tools._compose_subagent_context(parent, "explicit facts", True)

    assert "Parent user: original question" in inherited
    assert "Parent assistant: first answer" in inherited
    assert "Parent user: follow-up" in inherited
    assert "Explicit task context:\nexplicit facts" in inherited
    assert "hidden system" not in inherited
    assert "sensitive tool output" not in inherited
    assert "secret" not in inherited

    oversized = SimpleNamespace(_session_messages=[
        {"role": "user", "content": str(index) + "x" * 1000}
        for index in range(30)
    ])
    bounded = tools._compose_subagent_context(oversized, None, True)
    assert len(bounded) <= tools._INHERITED_CONTEXT_CHAR_LIMIT + 100
    assert "Parent user: 29" in bounded
    assert "Parent user: 0" not in bounded


def test_toolset_inheritance_can_be_disabled_or_explicitly_emptied():
    assert tools._resolve_subagent_toolsets(None, True) is None
    assert tools._resolve_subagent_toolsets(None, False) == [tools._NO_TOOLSETS_SENTINEL]
    assert tools._resolve_subagent_toolsets([], True) == [tools._NO_TOOLSETS_SENTINEL]
    assert tools._resolve_subagent_toolsets(["web", "file"], False) == ["web", "file"]

    with pytest.raises(ValueError, match="toolsets"):
        tools._resolve_subagent_toolsets("web", True)


def test_post_build_toolset_policy_removes_core_preserved_mcp(monkeypatch):
    child = SimpleNamespace(
        enabled_toolsets=["mcp-demo"],
        disabled_toolsets=[],
        tools=[{"function": {"name": "mcp_secret"}}],
        valid_tool_names={"mcp_secret"},
        _cached_system_prompt="stale",
    )
    rebuilds = []

    import model_tools

    def rebuild(*, enabled_toolsets, disabled_toolsets, quiet_mode):
        rebuilds.append((enabled_toolsets, disabled_toolsets, quiet_mode))
        return []

    monkeypatch.setattr(model_tools, "get_tool_definitions", rebuild)

    tools._enforce_subagent_toolset_policy(
        child,
        [tools._NO_TOOLSETS_SENTINEL],
    )

    assert child.enabled_toolsets == []
    assert "mcp-demo" in child.disabled_toolsets
    assert child.tools == []
    assert child.valid_tool_names == set()
    assert child._cached_system_prompt is None
    assert rebuilds == [([], ["mcp-demo"], True)]

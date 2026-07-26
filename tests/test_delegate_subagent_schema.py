"""Schema/handler contract tests for delegate_subagent."""

import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_herald import tools


def test_delegate_subagent_schema_matches_async_handler():
    description = tools.DELEGATE_SUBAGENT_SCHEMA["description"]
    properties = tools.DELEGATE_SUBAGENT_SCHEMA["parameters"]["properties"]
    source = inspect.getsource(tools.handle_delegate_subagent)

    assert "asynchronously" in description
    assert "task_id" in description
    assert "auto-delivered" in description
    assert "Runs synchronously" not in description
    assert "Only the final summary is returned" not in description
    assert "final result (summary or error)" in description
    assert "final summary" not in description
    assert "in-process rather than durable" in description
    assert "hard wall-clock timeout" not in description
    assert "cooperative interruption" in description
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

    assert "thread.start()" in source
    assert '"task_id": task_id' in source
    assert '"status": "dispatched"' in source


def test_stateless_session_is_rejected_before_background_work(monkeypatch):
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: False)
    result = tools.json.loads(tools.handle_delegate_subagent(
        {"goal": "audit this"},
        parent_agent=SimpleNamespace(),
    ))
    assert result["status"] == "error"
    assert "cannot receive detached results" in result["error"]


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


def test_soul_inheritance_is_opt_in_and_invalidates_cached_prompt():
    child = SimpleNamespace(load_soul_identity=True, _cached_system_prompt="stale")

    tools._apply_soul_inheritance(child, False)
    assert child.load_soul_identity is False
    assert child._cached_system_prompt is None

    child._cached_system_prompt = "stale-again"
    tools._apply_soul_inheritance(child, True)
    assert child.load_soul_identity is True
    assert child._cached_system_prompt is None


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

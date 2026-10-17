"""Schema/handler contract tests for delegate_subagent."""

from types import SimpleNamespace

import pytest

from hermes_herald import tools


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
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


def test_parent_agent_resolves_explicit_and_fails_closed_when_unbound():
    assert tools._resolve_parent_agent(None) is None
    explicit = SimpleNamespace(session_id="explicit")
    assert tools._resolve_parent_agent(explicit) is explicit


def test_parent_agent_resolves_from_public_turn_binding():
    lifecycle = pytest.importorskip("agent.subagent_lifecycle")
    expected = SimpleNamespace(session_id="durable-session")
    with lifecycle.bind_subagent_parent(expected):
        assert tools._resolve_parent_agent(None) is expected
        assert tools._resolve_parent_agent(None, "durable-session") is expected
        assert tools._resolve_parent_agent(None, "other-session") is None
    assert tools._resolve_parent_agent(None) is None


def test_parent_agent_fails_closed_when_lifecycle_module_unavailable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "agent.subagent_lifecycle", None)
    assert tools._resolve_parent_agent(None) is None
    explicit = SimpleNamespace(session_id="explicit")
    assert tools._resolve_parent_agent(explicit) is explicit


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
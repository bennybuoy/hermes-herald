"""Regression tests for the dispatch approval-consent import path.

``request_elicitation_consent`` is imported from its **defining** module,
``tools.approval_prompt``. An earlier revision imported it through
``tools.approval`` — a revert-scheduled compatibility re-export — and that
alias import made the plugin unloadable once the compat layer was removed
(2026-09-14): the plugin manager refuses to load a plugin whose AST still
reaches a removed path, so the whole toolset silently disappeared.

These tests call the REAL gate with simulated host generations via
``sys.modules`` surgery (the plugin imports ``tools.*`` lazily at call time):

  - current host:     ``tools.approval_prompt`` present -> gate works
  - broken host:      defining module absent -> must fail CLOSED, never raise
  - legacy alias:     only the compat re-export present -> must NOT be
                      consulted (depending on it is the loadability bug)
"""
import sys
import types

import pytest

from hermes_herald import tools

# Captured at import time: the autouse consent fixture patches the module
# attribute per test, so the real callable must be bound before that.
_REAL_CONSENT = tools._request_dispatch_approval_consent


def _call_real_gate() -> bool:
    return _REAL_CONSENT(
        profile="atlas",
        run_id="run-123",
        choice="deny",
        resolve_all=False,
        approval_data={"command": "echo hi", "description": "test"},
    )


def _accept(message, description, *, timeout_seconds=None, surface=""):
    """Stand-in consent surface: a human said yes."""
    return "accept"


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.request_elicitation_consent = _accept
    return mod


def test_current_host_resolves_the_defining_module(monkeypatch):
    """Current host: the defining module resolves and the gate works."""
    monkeypatch.setitem(sys.modules, "tools.approval_prompt", _fake_module("tools.approval_prompt"))
    monkeypatch.setitem(sys.modules, "tools.approval", _fake_module("tools.approval"))
    assert _call_real_gate() is True


def test_missing_defining_module_fails_closed(monkeypatch):
    """Broken host: the defining module is unimportable. The gate must fail
    CLOSED (False), not raise — approve_dispatch then reports the denial was
    not confirmed instead of crashing."""
    monkeypatch.setitem(sys.modules, "tools.approval_prompt", None)
    monkeypatch.setitem(sys.modules, "tools.approval", _fake_module("tools.approval"))
    assert _call_real_gate() is False


def test_legacy_reexport_is_not_consulted(monkeypatch):
    """The compat re-export must not act as a fallback. Importing it is what
    makes the plugin unloadable after the compat layer is removed, so with only
    ``tools.approval`` available the gate fails closed rather than depending on
    the alias."""
    monkeypatch.setitem(sys.modules, "tools.approval_prompt", None)
    monkeypatch.setitem(sys.modules, "tools.approval", _fake_module("tools.approval"))
    assert _call_real_gate() is False

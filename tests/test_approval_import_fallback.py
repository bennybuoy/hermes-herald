"""Regression tests for the _request_dispatch_approval_consent import fallback.

Codex review of PR #18 (P1) flagged that importing
``request_elicitation_consent`` unconditionally from ``tools.approval_prompt``
breaks the human consent gate on pre-Sep-2026-split Hermes hosts (including the
CI-pinned revision), where the symbol only exists on ``tools.approval``. The
suite's autouse fixture patches ``tools._request_dispatch_approval_consent``,
so no existing test exercised the real import path.

These tests call the REAL gate with simulated host generations via
``sys.modules`` surgery (the plugin imports ``tools.*`` lazily at call time):

  - pre-split host:  ``tools.approval_prompt`` absent (``None`` entry)
  - current host:    ``tools.approval_prompt`` present, old module absent
  - broken host:     both absent — must still fail CLOSED, never raise
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


def test_pre_split_host_falls_back_to_legacy_module(monkeypatch):
    """Pre-split host (CI-pinned 212e8417 generation): no approval_prompt
    module exists. Without the fallback the gate returns False (ImportError
    hits the outer handler and fails closed) — remote denials would silently
    stop working."""
    monkeypatch.setitem(sys.modules, "tools.approval_prompt", None)
    monkeypatch.setitem(sys.modules, "tools.approval", _fake_module("tools.approval"))
    assert _call_real_gate() is True


def test_current_host_uses_new_module_without_legacy(monkeypatch):
    """Current post-split host: the new module resolves; the legacy module is
    not needed (simulated absent) and must never be consulted."""
    monkeypatch.setitem(sys.modules, "tools.approval_prompt", _fake_module("tools.approval_prompt"))
    monkeypatch.setitem(sys.modules, "tools.approval", None)
    assert _call_real_gate() is True


def test_both_modules_missing_fails_closed(monkeypatch):
    """Broken host: neither module importable. The gate must fail CLOSED
    (False), not raise — approve_dispatch then reports the denial was not
    confirmed instead of crashing."""
    monkeypatch.setitem(sys.modules, "tools.approval_prompt", None)
    monkeypatch.setitem(sys.modules, "tools.approval", None)
    assert _call_real_gate() is False
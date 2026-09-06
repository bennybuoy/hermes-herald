"""Tests for llm_direct + remote reasoning_effort plumbing."""

import json
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import pytest

from hermes_herald import config, tools


# ---------------------------------------------------------------------------
# llm_direct
# ---------------------------------------------------------------------------

def _enable_llm_direct(monkeypatch, endpoints=None, default="test"):
    section = {
        "enabled": True,
        "default_endpoint": default,
        "endpoints": endpoints
        or {
            "test": {
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "test-key",
                "default_model": "model-a",
            },
        },
    }
    monkeypatch.setattr(config, "_load_config", lambda: {"llm_direct": section})


_MESSAGES = [{"role": "user", "content": "hello"}]


def test_llm_direct_disabled_fails_closed(monkeypatch):
    monkeypatch.setattr(config, "_load_config", lambda: {})
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "enabled: true" in result["error"]


def test_llm_direct_builds_full_request(monkeypatch):
    _enable_llm_direct(monkeypatch)
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "model": "model-a",
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["auth"] = req.headers.get("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    with mock_patch("urllib.request.urlopen", fake_urlopen):
        result = json.loads(tools.handle_llm_direct({
            "messages": _MESSAGES,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 64,
            "seed": 7,
            "stop": ["END"],
            "reasoning_effort": "low",
            "extra_body": {"top_k": 40},
        }))

    assert "error" not in result, result
    assert result["text"] == "hi"
    assert result["model"] == "model-a"
    body = captured["body"]
    assert body["model"] == "model-a"  # default_model used
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 64
    assert body["seed"] == 7
    assert body["stop"] == ["END"]
    assert body["reasoning_effort"] == "low"
    assert body["reasoning"] == {"enabled": True, "effort": "low"}
    assert body["top_k"] == 40
    assert "extra_body" not in body  # flattened, never sent as a key
    assert captured["auth"] == "Bearer test-key"


def test_llm_direct_reasoning_none_maps_to_disabled(monkeypatch):
    _enable_llm_direct(monkeypatch)
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }).encode()

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return FakeResponse()

    with mock_patch("urllib.request.urlopen", fake_urlopen):
        result = json.loads(tools.handle_llm_direct({
            "messages": _MESSAGES, "reasoning_effort": "none",
        }))

    assert "error" not in result, result
    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["reasoning"] == {"enabled": False}


def test_llm_direct_extra_body_rejects_underscore_keys(monkeypatch):
    _enable_llm_direct(monkeypatch)
    result = json.loads(tools.handle_llm_direct({
        "messages": _MESSAGES,
        "extra_body": {"_internal": "nope"},
    }))
    assert result["status"] == "error"
    assert "starting with '_'" in result["error"]


def test_llm_direct_model_allowlist_enforced(monkeypatch):
    _enable_llm_direct(monkeypatch, endpoints={
        "test": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "k",
            "allowed_models": ["model-a"],
        },
    })
    result = json.loads(tools.handle_llm_direct({
        "messages": _MESSAGES, "model": "model-b",
    }))
    assert result["status"] == "error"
    assert "allowed_models" in result["error"]


def test_llm_direct_endpoint_must_be_configured(monkeypatch):
    _enable_llm_direct(monkeypatch)
    result = json.loads(tools.handle_llm_direct({
        "messages": _MESSAGES, "endpoint": "unknown",
    }))
    assert result["status"] == "error"
    assert "not configured" in result["error"]


def test_llm_direct_param_validation(monkeypatch):
    _enable_llm_direct(monkeypatch)
    for bad, field in [
        ({"messages": _MESSAGES, "temperature": 3.0}, "temperature"),
        ({"messages": _MESSAGES, "top_p": 1.5}, "top_p"),
        ({"messages": _MESSAGES, "max_tokens": 0}, "max_tokens"),
        ({"messages": _MESSAGES, "seed": True}, "seed"),
        ({"messages": _MESSAGES, "stop": ["a", "b", "c", "d", "e"]}, "stop"),
        ({"messages": _MESSAGES, "reasoning_effort": "maximum"}, "reasoning_effort"),
        ({"messages": _MESSAGES, "timeout_seconds": 900}, "timeout_seconds"),
    ]:
        result = json.loads(tools.handle_llm_direct(bad))
        assert result["status"] == "error", f"{field} not validated: {bad}"
        assert field in result["error"]


def test_llm_direct_http_error_surfaces_provider_detail(monkeypatch):
    _enable_llm_direct(monkeypatch)
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 429, "rate limited",
            hdrs=None, fp=None,
        )

    with mock_patch("urllib.request.urlopen", fake_urlopen):
        result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "HTTP 429" in result["error"]


# ---------------------------------------------------------------------------
# Remote reasoning plumbing (dispatch_agent / dispatch_chat)
# ---------------------------------------------------------------------------

def test_remote_reasoning_model_options_shape():
    assert tools._remote_reasoning_model_options(None) is None
    assert tools._remote_reasoning_model_options({"enabled": False}) == {
        "reasoning": {"enabled": False},
    }
    assert tools._remote_reasoning_model_options(
        {"enabled": True, "effort": "high"}
    ) == {"reasoning": {"enabled": True, "effort": "high"}}


def test_dispatch_agent_schema_and_dispatch_chat_schema_share_reasoning_enum():
    for schema in (tools.DISPATCH_AGENT_SCHEMA, tools.DISPATCH_CHAT_SCHEMA):
        prop = schema["parameters"]["properties"]["reasoning_effort"]
        assert prop["enum"] == [
            "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "none",
        ]


def test_dispatch_agent_rejects_bad_reasoning_before_network(monkeypatch):
    # No network: handler must fail on enum validation before any request.
    result = json.loads(tools.handle_dispatch_agent({
        "profile": "whatever", "message": "task", "reasoning_effort": "bogus",
    }))
    assert result["status"] == "error"
    assert "reasoning_effort must be one of" in result["error"]


def test_dispatch_chat_rejects_bad_reasoning_before_network(monkeypatch):
    result = json.loads(tools.handle_dispatch_chat({
        "profile": "whatever", "message": "task", "reasoning_effort": 7,
    }))
    assert result["status"] == "error"
    assert "reasoning_effort must be one of" in result["error"]


def test_dispatch_agent_sends_model_options_reasoning(monkeypatch):
    captured = {}

    monkeypatch.setattr(config, "_load_config", lambda: {
        "profiles": {
            "remote": {"url": "http://127.0.0.1:9", "api_key": "k"},
        },
    })
    monkeypatch.setattr(tools, "_resolve_profile", lambda profile, operation="dispatch": (
        {"url": "http://127.0.0.1:9", "api_key": "k"}, None,
    ))
    monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
    monkeypatch.setattr(tools, "_verify_run_model_route", lambda *a, **k: ({}, None))
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env", lambda key, default="": default)

    def fake_post_json(url, api_key, body, timeout=30.0):
        captured["url"] = url
        captured["body"] = body
        return {"run_id": "run_x"}

    monkeypatch.setattr(tools, "_post_json", fake_post_json)
    monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: captured.update(
        ledger=kw))
    monkeypatch.setattr(tools, "_persist_run", lambda *a, **kw: None)

    result = json.loads(tools.handle_dispatch_agent({
        "profile": "remote",
        "message": "do the thing",
        "reasoning_effort": "high",
    }))

    assert result.get("run_id") == "run_x", result
    assert captured["body"]["model_options"] == {
        "reasoning": {"enabled": True, "effort": "high"},
    }
    assert captured["ledger"]["reasoning"] == "high"


def test_dispatch_agent_omitted_reasoning_sends_no_model_options(monkeypatch):
    captured = {}

    monkeypatch.setattr(tools, "_resolve_profile", lambda profile, operation="dispatch": (
        {"url": "http://127.0.0.1:9", "api_key": "k"}, None,
    ))
    monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
    monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env", lambda key, default="": default)

    def fake_post_json(url, api_key, body, timeout=30.0):
        captured["body"] = body
        return {"run_id": "run_y"}

    monkeypatch.setattr(tools, "_post_json", fake_post_json)
    monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: None)
    monkeypatch.setattr(tools, "_persist_run", lambda *a, **kw: None)

    result = json.loads(tools.handle_dispatch_agent({
        "profile": "remote", "message": "task",
    }))
    assert result.get("run_id") == "run_y"
    assert "model_options" not in captured["body"]
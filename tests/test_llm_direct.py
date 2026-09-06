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
    monkeypatch.setenv("HERMES_HERALD_TEST_DIRECT_KEY", "test-key")
    section = {
        "enabled": True,
        "default_endpoint": default,
        "endpoints": endpoints
        or {
            "test": {
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "${HERMES_HERALD_TEST_DIRECT_KEY}",
                "default_model": "model-a",
            },
        },
    }
    monkeypatch.setattr(config, "_load_config", lambda: {"llm_direct": section})


def _forbid_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("llm_direct must fail closed before any network call")
    monkeypatch.setattr(tools, "urlopen", boom)


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

    with mock_patch.object(tools, "urlopen", fake_urlopen):
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

    with mock_patch.object(tools, "urlopen", fake_urlopen):
        result = json.loads(tools.handle_llm_direct({
            "messages": _MESSAGES, "reasoning_effort": "none",
        }))

    assert "error" not in result, result
    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["reasoning"] == {"enabled": False}


def test_llm_direct_extra_body_rejects_underscore_keys(monkeypatch):
    _enable_llm_direct(monkeypatch)
    _forbid_network(monkeypatch)
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
            "api_key": "${HERMES_HERALD_TEST_DIRECT_KEY}",
            "allowed_models": ["model-a"],
        },
    })
    _forbid_network(monkeypatch)
    result = json.loads(tools.handle_llm_direct({
        "messages": _MESSAGES, "model": "model-b",
    }))
    assert result["status"] == "error"
    assert "allowed_models" in result["error"]


def test_llm_direct_endpoint_must_be_configured(monkeypatch):
    _enable_llm_direct(monkeypatch)
    _forbid_network(monkeypatch)
    result = json.loads(tools.handle_llm_direct({
        "messages": _MESSAGES, "endpoint": "unknown",
    }))
    assert result["status"] == "error"
    assert "not configured" in result["error"]


def test_llm_direct_param_validation(monkeypatch):
    _enable_llm_direct(monkeypatch)
    _forbid_network(monkeypatch)
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

    with mock_patch.object(tools, "urlopen", fake_urlopen):
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


def test_llm_direct_extra_body_cannot_override_model_or_temperature(monkeypatch):
    _enable_llm_direct(monkeypatch, endpoints={
        "test": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "${HERMES_HERALD_TEST_DIRECT_KEY}",
            "default_model": "model-a",
            "allowed_models": ["model-a"],
        },
    })
    _forbid_network(monkeypatch)
    for extra in ({"model": "FORBIDDEN"}, {"temperature": 900}):
        result = json.loads(tools.handle_llm_direct({
            "messages": _MESSAGES, "extra_body": extra,
        }))
        assert result["status"] == "error"
        assert "reserved fields" in result["error"]


def test_llm_direct_literal_api_key_rejected_before_network(monkeypatch):
    monkeypatch.setattr(config, "_load_config", lambda: {
        "llm_direct": {
            "enabled": True,
            "default_endpoint": "test",
            "endpoints": {
                "test": {
                    "base_url": "http://127.0.0.1:9/v1",
                    "api_key": "literal-secret",
                    "default_model": "model-a",
                },
            },
        },
    })
    _forbid_network(monkeypatch)
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "${ENV_VAR}" in result["error"]
    assert "literal-secret" not in result["error"]


def test_llm_direct_missing_env_rejected_before_network(monkeypatch):
    monkeypatch.delenv("MISSING_DIRECT_KEY", raising=False)
    monkeypatch.setattr(config, "_load_config", lambda: {
        "llm_direct": {
            "enabled": True,
            "default_endpoint": "test",
            "endpoints": {
                "test": {
                    "base_url": "http://127.0.0.1:9/v1",
                    "api_key": "${MISSING_DIRECT_KEY}",
                    "default_model": "model-a",
                },
            },
        },
    })
    _forbid_network(monkeypatch)
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "unset or empty" in result["error"]


def test_llm_direct_scheme_only_url_rejected_before_network(monkeypatch):
    monkeypatch.setenv("HERMES_HERALD_TEST_DIRECT_KEY", "test-key")
    monkeypatch.setattr(config, "_load_config", lambda: {
        "llm_direct": {
            "enabled": True,
            "default_endpoint": "test",
            "endpoints": {
                "test": {
                    "base_url": "http://",
                    "api_key": "${HERMES_HERALD_TEST_DIRECT_KEY}",
                    "default_model": "model-a",
                },
            },
        },
    })
    _forbid_network(monkeypatch)
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "hostname" in result["error"]


def test_llm_direct_url_embedded_credentials_rejected(monkeypatch):
    monkeypatch.setenv("HERMES_HERALD_TEST_DIRECT_KEY", "test-key")
    monkeypatch.setattr(config, "_load_config", lambda: {
        "llm_direct": {
            "enabled": True,
            "default_endpoint": "test",
            "endpoints": {
                "test": {
                    "base_url": "https://user:supersecret@127.0.0.1:9/v1",
                    "api_key": "${HERMES_HERALD_TEST_DIRECT_KEY}",
                    "default_model": "model-a",
                },
            },
        },
    })
    _forbid_network(monkeypatch)
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "embed credentials" in result["error"]
    assert "supersecret" not in result["error"]


def test_llm_direct_malformed_section_does_not_enable(monkeypatch):
    monkeypatch.setattr(config, "_load_config", lambda: {"llm_direct": "yes"})
    _forbid_network(monkeypatch)
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "enabled: true" in result["error"]


def test_llm_direct_redirect_handler_refuses_follow():
    from urllib.request import Request
    handler = tools._NoRedirectHandler()
    req = Request("http://127.0.0.1:9/v1/chat/completions")
    assert handler.redirect_request(
        req, fp=None, code=302, msg="Found",
        headers={"Location": "http://evil.example/steal"},
        newurl="http://evil.example/steal",
    ) is None


def test_llm_direct_uses_module_urlopen_not_urllib_default(monkeypatch):
    _enable_llm_direct(monkeypatch)
    calls = []

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
        calls.append(req.full_url)
        return FakeResponse()

    monkeypatch.setattr(tools, "urlopen", fake_urlopen)
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert "error" not in result, result
    assert calls == ["http://127.0.0.1:9/v1/chat/completions"]


def test_llm_direct_redacts_credential_from_http_error(monkeypatch):
    import io
    import urllib.error
    _enable_llm_direct(monkeypatch)

    def fake_urlopen(req, timeout=None):
        body = io.BytesIO(b'{"error":"invalid token test-key"}')
        raise urllib.error.HTTPError(
            req.full_url, 401, "unauthorized", hdrs=None, fp=body,
        )

    monkeypatch.setattr(tools, "urlopen", fake_urlopen)
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert result["status"] == "error"
    assert "HTTP 401" in result["error"]
    assert "test-key" not in result["error"]
    assert "[redacted]" in result["error"]


def test_llm_direct_redacts_credential_from_success_text(monkeypatch):
    _enable_llm_direct(monkeypatch)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {"content": "echo test-key please"},
                    "finish_reason": "stop",
                }],
            }).encode()

    monkeypatch.setattr(tools, "urlopen", lambda *a, **k: FakeResponse())
    result = json.loads(tools.handle_llm_direct({"messages": _MESSAGES}))
    assert "error" not in result, result
    assert "test-key" not in result["text"]
    assert "[redacted]" in result["text"]


def test_dispatch_chat_sends_model_options_reasoning(monkeypatch):
    from hermes_herald import callback
    captured = {}
    monkeypatch.setattr(
        tools, "_resolve_profile",
        lambda profile, operation=None: ({"url": "http://127.0.0.1:9", "api_key": "k"}, None),
    )
    monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
    monkeypatch.setattr(tools, "_verify_run_model_route", lambda *a, **k: ({}, None))

    def fake_stream(url, api_key, body, **kwargs):
        captured["body"] = body
        return {
            "session_id": "session-1",
            "reply": "ack",
            "model": "hermes-agent",
            "usage": {},
        }

    monkeypatch.setattr(tools, "_post_streaming_chat", fake_stream)
    monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
    monkeypatch.setattr(tools, "_save_state", lambda state: None)
    monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: captured.update(ledger=kw))
    monkeypatch.setattr(callback, "get_profile_session_id", lambda profile: "session-1")
    monkeypatch.setattr(callback, "capture_session_routing", lambda parent=None: {})
    result = json.loads(tools.handle_dispatch_chat({
        "profile": "remote", "message": "hi", "reasoning_effort": "high",
    }))
    assert result.get("status") == "completed", result
    assert captured["body"]["model_options"] == {
        "reasoning": {"enabled": True, "effort": "high"},
    }
    assert captured["ledger"]["reasoning"] == "high"
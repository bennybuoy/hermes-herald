"""Behavioral contracts for Herald's host-backed ``llm_call`` tool."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_herald import tools


def _usage(**overrides):
    values = {
        "input_tokens": 4,
        "output_tokens": 3,
        "total_tokens": 7,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _host_result(text="host result", provider="openai-codex", model="gpt-5.6-sol"):
    return SimpleNamespace(
        text=text,
        provider=provider,
        model=model,
        usage=_usage(),
    )


def _configured_inventory():
    return {
        "configured_default": {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
        },
        "providers": [
            {"provider": "openai-codex", "models": ["gpt-5.6-sol"]},
            {"provider": "ollama-cloud", "models": ["glm-5.2"]},
        ],
    }


@pytest.fixture
def host_llm():
    facade = MagicMock()
    facade.complete.return_value = _host_result()
    return facade


@pytest.fixture(autouse=True)
def configured_routes(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_configured_local_model_inventory",
        lambda **kwargs: _configured_inventory(),
    )


def _call(args, host_llm):
    return json.loads(tools.handle_llm_call(args, _llm=host_llm))


def test_exact_advertised_pair_uses_host_plugin_llm_facade(host_llm):
    host_llm.complete.return_value = _host_result(
        text="Hello from GLM",
        provider="ollama-cloud",
        model="glm-5.2",
    )

    result = _call({
        "messages": [{"role": "user", "content": "Say hello"}],
        "provider": "ollama-cloud",
        "model": "glm-5.2",
    }, host_llm)

    host_llm.complete.assert_called_once_with(
        messages=[{"role": "user", "content": "Say hello"}],
        provider="ollama-cloud",
        model="glm-5.2",
        temperature=None,
        max_tokens=None,
        timeout=120.0,
        purpose="hermes-herald.llm_call",
    )
    assert result == {
        "text": "Hello from GLM",
        "provider": "ollama-cloud",
        "requested_provider": "ollama-cloud",
        "requested_model": "glm-5.2",
        "model": "glm-5.2",
        "usage": {
            "input_tokens": 4,
            "output_tokens": 3,
            "total_tokens": 7,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": None,
        },
    }


def test_omitted_route_uses_host_configured_default(host_llm):
    _call({"messages": [{"role": "user", "content": "Hi"}]}, host_llm)

    assert host_llm.complete.call_args.kwargs["provider"] is None
    assert host_llm.complete.call_args.kwargs["model"] is None


def test_unconfigured_pair_is_rejected_before_host_call(host_llm):
    result = _call({
        "messages": [{"role": "user", "content": "Hi"}],
        "provider": "openrouter",
        "model": "guess/model",
    }, host_llm)

    assert result["status"] == "error"
    assert "not configured for this profile" in result["error"]
    assert "list_profile_models" in result["error"]
    host_llm.complete.assert_not_called()


@pytest.mark.parametrize(
    "route",
    [
        {"provider": "ollama-cloud"},
        {"model": "glm-5.2"},
    ],
)
def test_provider_and_model_must_come_from_same_inventory_pair(route, host_llm):
    result = _call({
        "messages": [{"role": "user", "content": "Hi"}],
        **route,
    }, host_llm)

    assert result["status"] == "error"
    assert "Pass both" in result["error"]
    host_llm.complete.assert_not_called()


def test_host_policy_error_is_reported_without_retry(host_llm):
    host_llm.complete.side_effect = PermissionError(
        "Plugin 'hermes-herald' is not trusted for cross-provider routing"
    )

    result = _call({
        "messages": [{"role": "user", "content": "Hi"}],
        "provider": "ollama-cloud",
        "model": "glm-5.2",
    }, host_llm)

    assert result["status"] == "error"
    assert "not trusted" in result["error"]
    host_llm.complete.assert_called_once()


def test_missing_host_facade_fails_closed():
    result = json.loads(tools.handle_llm_call({
        "messages": [{"role": "user", "content": "Hi"}],
    }))

    assert result["status"] == "error"
    assert "host LLM service is unavailable" in result["error"]


def test_system_prompt_and_parameters_are_forwarded(host_llm):
    _call({
        "messages": [{"role": "user", "content": "Hi"}],
        "system_prompt": "Be concise.",
        "temperature": 0.25,
        "max_tokens": 80,
    }, host_llm)

    call = host_llm.complete.call_args.kwargs
    assert call["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    assert call["temperature"] == 0.25
    assert call["max_tokens"] == 80


def test_json_mode_instructs_and_validates_object(host_llm):
    host_llm.complete.return_value = _host_result(text='```json\n{"ok": true}\n```')

    result = _call({
        "messages": [{"role": "user", "content": "Return JSON"}],
        "json_mode": True,
    }, host_llm)

    assert result["text"] == '{"ok":true}'
    sent = host_llm.complete.call_args.kwargs["messages"]
    assert sent[0]["role"] == "system"
    assert "valid JSON object" in sent[0]["content"]


@pytest.mark.parametrize("text", ["not JSON", "[1,2,3]"])
def test_json_mode_rejects_non_object_output(host_llm, text):
    host_llm.complete.return_value = _host_result(text=text)

    result = _call({
        "messages": [{"role": "user", "content": "Return JSON"}],
        "json_mode": True,
    }, host_llm)

    assert result["status"] == "error"
    assert "json_mode" in result["error"]


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"messages": []},
        {"messages": "not-a-list"},
        {"messages": [{"role": "invalid", "content": "Hi"}]},
        {"messages": [{"role": "user", "content": 123}]},
        {"messages": [{"role": "user", "content": "Hi"}], "temperature": 3},
        {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 0},
    ],
)
def test_invalid_inputs_do_not_reach_host(args, host_llm):
    result = _call(args, host_llm)

    assert result["status"] == "error"
    host_llm.complete.assert_not_called()


def test_schema_teaches_discovery_before_exact_pair_execution():
    description = tools.LLM_CALL_SCHEMA["description"]
    properties = tools.LLM_CALL_SCHEMA["parameters"]["properties"]
    assert "list_profile_models" in description
    assert "provider/model pair" in description
    assert properties["messages"]["minItems"] == 1
    assert "must be supplied together" in properties["model"]["description"]
    assert "must be supplied together" in properties["provider"]["description"]
    assert "response_format" not in properties["json_mode"]["description"]

"""Compatibility tests against Hermes' real public plugin LLM facade."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.plugin_llm import _TrustPolicy, make_plugin_llm_for_test
from hermes_herald import tools


def _openai_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, role="assistant"),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
        ),
    )


def _inventory():
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


def test_handler_executes_exact_inventory_pair_through_real_plugin_llm(monkeypatch):
    captured = {}

    def caller(**kwargs):
        captured.update(kwargs)
        return "ollama-cloud", "glm-5.2", _openai_response("real facade result")

    policy = _TrustPolicy(
        plugin_id="hermes-herald",
        allow_provider_override=True,
        allow_any_provider=True,
        allow_model_override=True,
        allow_any_model=True,
    )
    llm = make_plugin_llm_for_test(
        plugin_id="hermes-herald",
        policy=policy,
        sync_caller=caller,
    )
    monkeypatch.setattr(tools, "_configured_local_model_inventory", lambda **kwargs: _inventory())

    result = json.loads(tools.handle_llm_call({
        "messages": [{"role": "user", "content": "ping"}],
        "provider": "ollama-cloud",
        "model": "glm-5.2",
    }, _llm=llm))

    assert result["text"] == "real facade result"
    assert result["provider"] == "ollama-cloud"
    assert result["model"] == "glm-5.2"
    assert result["usage"]["total_tokens"] == 6
    assert captured["provider_override"] == "ollama-cloud"
    assert captured["model_override"] == "glm-5.2"


def test_real_plugin_llm_trust_gate_blocks_override_before_transport(monkeypatch):
    calls = []
    llm = make_plugin_llm_for_test(
        plugin_id="hermes-herald",
        policy=_TrustPolicy(plugin_id="hermes-herald"),
        sync_caller=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(tools, "_configured_local_model_inventory", lambda **kwargs: _inventory())

    result = json.loads(tools.handle_llm_call({
        "messages": [{"role": "user", "content": "ping"}],
        "provider": "ollama-cloud",
        "model": "glm-5.2",
    }, _llm=llm))

    assert result["status"] == "error"
    assert "cannot override the provider" in result["error"]
    assert calls == []

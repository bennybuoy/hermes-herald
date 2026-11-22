"""Regression tests for configured local model-route discovery."""
import json
import os
import sys
import types
import tempfile
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from hermes_herald import tools


class TestLocalModelRouteDiscovery:
    """The caller can discover only routes declared by its active profile."""

    def test_schema_allows_local_discovery_without_a_profile_name(self):
        assert "profile" not in tools.LIST_PROFILE_MODELS_SCHEMA["parameters"].get(
            "required", []
        )

    def test_local_inventory_uses_core_explicit_only_contract(self, monkeypatch):
        import hermes_cli.inventory as inventory

        context = SimpleNamespace(
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            current_base_url="http://ollama-cloud.example/v1",
            user_providers={
                "backup": {
                    "base_url": "http://backup.example/v1",
                    "models": ["backup-model"],
                }
            },
            custom_providers=[{
                "name": "llamaherd",
                "base_url": "http://llamaherd.example/v1",
                "models": {"glm-5.2": {}},
            }],
            excluded_providers=[],
        )
        monkeypatch.setattr(inventory, "load_picker_context", lambda: context)
        import hermes_cli.model_switch as model_switch

        monkeypatch.setattr(
            model_switch,
            "switch_model",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("discovery must report configured slugs without resolving them")
            ),
        )
        captured = {}

        def explicit_payload(cfg_ctx, *args, **kwargs):
            captured.update(kwargs)
            captured["ctx"] = cfg_ctx
            return {
                "providers": [
                    {"slug": "ollama-cloud", "models": ["glm-5.2"]},
                    {"slug": "backup", "models": ["backup-model"]},
                    {"slug": "custom:llamaherd", "models": ["glm-5.2"]},
                    {"slug": "custom:openai-2", "models": ["model-b"]},
                    {"slug": "moa", "models": ["default"]},
                ]
            }

        monkeypatch.setattr(inventory, "build_models_payload", explicit_payload)

        result = json.loads(tools.handle_list_profile_models({}))

        assert result["scope"] == "local"
        assert result["configured_default"] == {
            "provider": "ollama-cloud",
            "model": "glm-5.2",
        }
        assert result["available_routes"] == [
            {"provider": "backup", "model": "backup-model"},
            {"provider": "custom:llamaherd", "model": "glm-5.2"},
            {"provider": "custom:openai-2", "model": "model-b"},
            {"provider": "ollama-cloud", "model": "glm-5.2"},
        ]
        assert result["route_count"] == 4
        assert "do not pass a bare model name" in result["contract"]
        assert captured["explicit_only"] is True
        assert captured["include_unconfigured"] is False

        # Live bound parent overrides persisted disk defaults without reading
        # private auxiliary_client helpers.
        live_parent = SimpleNamespace(
            provider="backup",
            model="backup-model",
            base_url="http://backup.example/v1",
        )
        live_result = json.loads(
            tools.handle_list_profile_models({}, parent_agent=live_parent)
        )
        assert live_result["configured_default"] == {
            "provider": "backup",
            "model": "backup-model",
        }
        assert captured["ctx"].current_provider == "backup"
        assert captured["ctx"].current_base_url == "http://backup.example/v1"

    def test_named_profile_preserves_remote_authenticated_alias_contract(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile: ({
                "url": "https://reviewer.example",
                "api_key": "remote-secret",
            }, None),
        )
        request = {}

        def remote_models(url, api_key, timeout):
            request.update(url=url, api_key=api_key, timeout=timeout)
            return {
                "data": [
                    {"id": "glm-5.2", "root": "glm-5.2"},
                    {
                        "id": "review-fast",
                        "root": "glm-5.2",
                        "parent": "glm-5.2",
                        "provider": "nous",
                    },
                ]
            }

        monkeypatch.setattr(tools, "_get_json", remote_models)

        raw = tools.handle_list_profile_models({"profile": "reviewer"})
        result = json.loads(raw)

        assert request == {
            "url": "https://reviewer.example/v1/models",
            "api_key": "remote-secret",
            "timeout": 10.0,
        }
        assert result["profile"] == "reviewer"
        assert result["advertised_primary"] == {
            "model": "glm-5.2",
            "dispatchable_as_override": False,
            "is_runtime_evidence": False,
        }
        assert result["pass_as_model"] == ["review-fast"]
        assert result["dispatchable_models"] == [{
            "alias": "review-fast",
            "pass_as": "review-fast",
            "resolved_model": "glm-5.2",
            "provider": "nous",
            "provider_pinned": True,
        }]
        assert result["dispatchable_model_count"] == 1
        assert "pass provider=" in result["contract"]
        assert "remote-secret" not in raw

        queried = json.loads(tools.handle_list_profile_models({
            "profile": "reviewer", "query": "review",
        }))
        assert queried["query"] == "review"
        assert queried["match_pass_as"] == ["review-fast"]
        assert queried["pass_as_model"] == ["review-fast"]

        missed = json.loads(tools.handle_list_profile_models({
            "profile": "reviewer", "query": "astra",
        }))
        assert missed["match_count"] == 0
        assert missed["match_pass_as"] == []
        assert missed["pass_as_model"] == ["review-fast"]
"""Regression tests for local and delegated model resolution."""
import json
import os
import sys
import types
import tempfile
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from hermes_herald import tools


class TestModelResolution:
    """Test _resolve_model_creds with mocked switch_model."""

    def test_empty_model_returns_none(self):
        """Empty model name returns all-None creds (inherits parent)."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"

        creds = tools._resolve_model_creds("", parent)
        assert creds["model"] is None
        assert creds["provider"] is None

    def test_same_provider_inherits(self):
        """Same provider as parent inherits credentials."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"
        parent.base_url = "http://localhost:8399/v1"
        parent.api_key = "test-key"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.new_model = "glm-5.2"
        mock_result.target_provider = "ollama-cloud"

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            creds = tools._resolve_model_creds("glm-5.2", parent)

        assert creds["model"] == "glm-5.2"
        assert creds["provider"] is None

    def test_different_provider_overrides(self):
        """Different provider gets fresh credentials."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"
        parent.base_url = "http://localhost:8399/v1"
        parent.api_key = "test-key"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.new_model = "claude-sonnet-4"
        mock_result.target_provider = "anthropic"
        mock_result.base_url = "https://api.anthropic.com"
        mock_result.api_key = "sk-ant-xxx"
        mock_result.api_mode = "anthropic"

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            creds = tools._resolve_model_creds("claude-sonnet-4", parent)

        assert creds["model"] == "claude-sonnet-4"
        assert creds["provider"] == "anthropic"
        assert creds["base_url"] == "https://api.anthropic.com"
        assert creds["api_key"] == "sk-ant-xxx"

    def test_resolution_failure_raises(self):
        """Failed model resolution raises ValueError."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Unknown model 'nonexistent'"

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            with pytest.raises(ValueError, match="Unknown model"):
                tools._resolve_model_creds("nonexistent", parent)


class TestLocalModelRouteDiscovery:
    """The caller can discover only routes declared by its active profile."""

    def test_schema_allows_local_discovery_without_a_profile_name(self):
        assert "profile" not in tools.LIST_PROFILE_MODELS_SCHEMA["parameters"].get(
            "required", []
        )

    def test_local_inventory_excludes_ambient_openrouter_and_nous(self, monkeypatch):
        import agent.auxiliary_client as auxiliary_client
        import hermes_cli.inventory as inventory

        context = SimpleNamespace(
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            current_base_url="http://ollama-cloud.example/v1",
            user_providers={},
            custom_providers=[{
                "name": "llamaherd",
                "base_url": "http://llamaherd.example/v1",
                "models": {"glm-5.2": {}},
            }],
            excluded_providers=[],
        )
        monkeypatch.setattr(inventory, "load_picker_context", lambda: context)
        monkeypatch.setattr(
            auxiliary_client, "_read_main_provider", lambda: "ollama-cloud"
        )
        monkeypatch.setattr(auxiliary_client, "_read_main_model", lambda: "glm-5.2")
        monkeypatch.setattr(
            auxiliary_client,
            "_read_main_base_url",
            lambda: "http://ollama-cloud.example/v1",
        )
        monkeypatch.setattr(
            inventory,
            "build_models_payload",
            lambda *args, **kwargs: {
                "providers": [
                    {"slug": "ollama-cloud", "models": ["glm-5.2"]},
                    {"slug": "custom:llamaherd", "models": ["glm-5.2"]},
                    {"slug": "openrouter", "models": ["z-ai/glm-5.2"]},
                    {"slug": "nous", "models": ["z-ai/glm-5.2"]},
                ]
            },
        )

        result = json.loads(tools.handle_list_profile_models({}))

        assert result["scope"] == "local"
        assert result["configured_default"] == {
            "provider": "ollama-cloud",
            "model": "glm-5.2",
        }
        assert result["available_routes"] == [
            {"provider": "custom:llamaherd", "model": "glm-5.2"},
            {"provider": "ollama-cloud", "model": "glm-5.2"},
        ]
        assert result["route_count"] == 2
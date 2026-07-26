"""Regression tests for model resolution in delegate_subagent."""
import os
import sys
import types
import tempfile
import pytest
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
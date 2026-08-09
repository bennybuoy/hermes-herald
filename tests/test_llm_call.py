"""Tests for the llm_call tool — response extraction, error handling, edge cases."""
import json
import os
import sys
import types
import tempfile
import pytest
from unittest.mock import patch, MagicMock, ANY

from hermes_herald import tools


@pytest.fixture(autouse=True)
def _declared_default_llm_route(monkeypatch):
    """Most tests exercise response handling, not route discovery."""
    from types import SimpleNamespace
    import agent.auxiliary_client as auxiliary_client
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(auxiliary_client, "_read_main_provider", lambda: "openai-codex")
    monkeypatch.setattr(auxiliary_client, "_read_main_model", lambda: "gpt-5.6-sol")
    monkeypatch.setattr(
        auxiliary_client,
        "_read_main_base_url",
        lambda: "https://chatgpt.com/backend-api/codex",
    )
    monkeypatch.setattr(auxiliary_client, "_read_main_api_key", lambda: "test-token")
    monkeypatch.setattr(
        tools,
        "_configured_local_model_inventory",
        lambda **kwargs: {
            "configured_default": {
                "provider": kwargs.get("current_provider") or "openai-codex",
                "model": kwargs.get("current_model") or "gpt-5.6-sol",
            },
            "providers": [{
                "provider": kwargs.get("current_provider") or "openai-codex",
                "models": [kwargs.get("current_model") or "gpt-5.6-sol"],
            }],
            "user_providers": {},
            "custom_providers": [],
        },
    )
    monkeypatch.setattr(
        model_switch,
        "switch_model",
        lambda raw_input, explicit_provider, **kwargs: SimpleNamespace(
            success=True,
            new_model=raw_input,
            target_provider=explicit_provider,
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-token",
            api_mode="codex_responses",
            error_message="",
        ),
    )


class FakeChoice:
    """Simulates an OpenAI API response choice."""
    def __init__(self, text="", message_content=None):
        if message_content is not None:
            self.message = MagicMock()
            self.message.content = message_content
        else:
            self.message = MagicMock()
            self.message.content = text


class FakeResponse:
    """Simulates a call_llm response with various shapes."""
    def __init__(self, text="", model="gpt-4o", choices=None,
                 content=None, usage=None, is_dict=False):
        self.model = model
        if choices is not None:
            self.choices = choices
        if content is not None:
            self.content = content
        if usage is not None:
            self.usage = usage


class FakeUsage:
    """Simulates a usage object with model_dump support."""
    def __init__(self, prompt=10, completion=20, total=30):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total

    def model_dump(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class FakeUsageV1:
    """Simulates a Pydantic v1 usage object (uses .dict() not .model_dump())."""
    def __init__(self, prompt=5, completion=10, total=15):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total

    def dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class FakeUsageBare:
    """Simulates a usage object with only attribute access (no model_dump/dict)."""
    def __init__(self, prompt=1, completion=2, total=3):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


def _install_route_resolution_fakes(
    monkeypatch,
    *,
    current_provider="openai-codex",
    current_model="gpt-5.6-sol",
    resolved_provider="openai-codex",
    resolved_model="gpt-5.6-sol",
    available_models=None,
    resolved_base_url="https://chatgpt.com/backend-api/codex",
    resolved_api_key="test-token",
):
    """Install deterministic current-runtime, resolver, and inventory fakes."""
    from types import SimpleNamespace
    import agent.auxiliary_client as auxiliary_client
    import hermes_cli.inventory as inventory
    import hermes_cli.model_switch as model_switch

    current_base_url = "https://chatgpt.com/backend-api/codex"
    for name, value in (
        ("_read_main_provider", current_provider),
        ("_read_main_model", current_model),
        ("_read_main_base_url", current_base_url),
        ("_read_main_api_key", ""),
    ):
        monkeypatch.setattr(
            auxiliary_client,
            name,
            MagicMock(return_value=value),
            raising=False,
        )

    switch = MagicMock(return_value=SimpleNamespace(
        success=True,
        new_model=resolved_model,
        target_provider=resolved_provider,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        api_mode="chat_completions",
        error_message="",
    ))
    monkeypatch.setattr(model_switch, "switch_model", switch)

    picker_context = SimpleNamespace(
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        user_providers={},
        custom_providers=[],
        with_overrides=lambda **kwargs: picker_context,
    )
    monkeypatch.setattr(inventory, "load_picker_context", lambda: picker_context)
    monkeypatch.setattr(
        inventory,
        "build_models_payload",
        lambda *args, **kwargs: {
            "providers": [
                {
                    "slug": current_provider,
                    "models": [current_model],
                },
                {
                    "slug": resolved_provider,
                    "models": available_models or [
                        "gpt-5.6-sol", "gpt-5.6-terra"
                    ],
                },
            ]
        },
    )
    monkeypatch.setattr(
        tools,
        "_configured_local_model_inventory",
        lambda **kwargs: {
            "configured_default": {
                "provider": current_provider,
                "model": current_model,
            },
            "providers": [{
                "provider": current_provider,
                "models": (
                    available_models
                    if resolved_provider == current_provider and available_models
                    else [current_model]
                ),
            }],
            "user_providers": {},
            "custom_providers": [],
        },
    )
    return switch


class TestLlmCallResponseExtraction:
    """Test handle_llm_call response extraction from various response shapes."""

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_openai_choices_response(self, mock_call_llm):
        """OpenAI-compatible response with choices[0].message.content."""
        mock_call_llm.return_value = FakeResponse(
            text="Hello world",
            model="gpt-4o",
            choices=[FakeChoice(message_content="Hello world")],
            usage=FakeUsage(prompt=10, completion=20, total=30),
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Say hello"}],
        }))

        assert result["text"] == "Hello world"
        assert result["model"] == "gpt-4o"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_reasoning_only_openai_response(self, mock_call_llm):
        response = FakeResponse(
            model="deepseek-r1",
            choices=[FakeChoice(message_content="")],
        )
        response.choices[0].message.reasoning_content = "Reasoning-only answer"
        mock_call_llm.return_value = response

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Solve this"}],
        }))

        assert result["text"] == "Reasoning-only answer"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_reasoning_fallback_for_older_hermes_core(self, mock_call_llm, monkeypatch):
        import agent.auxiliary_client as auxiliary_client

        monkeypatch.delattr(auxiliary_client, "extract_content_or_reasoning")
        response = FakeResponse(
            model="deepseek-r1",
            choices=[FakeChoice(message_content="")],
        )
        response.choices[0].message.reasoning_content = "compatibility answer"
        mock_call_llm.return_value = response

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Solve this"}],
        }))

        assert result["text"] == "compatibility answer"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_anthropic_content_response(self, mock_call_llm):
        """Anthropic-style response with top-level content string."""
        mock_call_llm.return_value = FakeResponse(
            content="Hello from Claude",
            model="claude-sonnet-4",
            usage=FakeUsage(prompt=15, completion=25, total=40),
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "Hello from Claude"
        assert result["model"] == "claude-sonnet-4"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_anthropic_content_list(self, mock_call_llm):
        """Anthropic-style response with content as list of text blocks."""
        block1 = MagicMock()
        block1.text = "Hello"
        block2 = MagicMock()
        block2.text = "world"
        mock_call_llm.return_value = FakeResponse(
            content=[block1, block2],
            model="claude-sonnet-4",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "Hello world"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_anthropic_content_list_skips_non_text_blocks(self, mock_call_llm):
        block1 = MagicMock()
        block1.text = None
        block2 = MagicMock()
        block2.text = "world"
        mock_call_llm.return_value = FakeResponse(content=[block1, block2])

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "world"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_dict_response_fallback(self, mock_call_llm):
        """Response as a plain dict (fallback path)."""
        mock_call_llm.return_value = {
            "content": "Hello from dict",
            "model": "gpt-4o-mini",
        }

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "Hello from dict"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_dict_response_extracts_model_and_usage(self, mock_call_llm):
        mock_call_llm.return_value = {
            "content": "Hello from dict",
            "model": "dict-model",
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        }

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["model"] == "dict-model"
        assert result["usage"]["total_tokens"] == 9

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_dict_response_text_fallback(self, mock_call_llm):
        """Dict response with 'text' key instead of 'content'."""
        mock_call_llm.return_value = {
            "text": "Hello from text key",
            "model": "gpt-4o-mini",
        }

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "Hello from text key"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_dict_openai_message_response(self, mock_call_llm):
        mock_call_llm.return_value = {
            "choices": [{"message": {"content": "Hello from choices"}}],
        }

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "Hello from choices"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_dict_reasoning_only_response(self, mock_call_llm):
        mock_call_llm.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "dict reasoning",
                    "reasoning_details": [{"summary": "detail summary"}],
                }
            }],
        }

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "dict reasoning\n\ndetail summary"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_dict_reasoning_wrappers_fall_back_to_structured_reasoning(self, mock_call_llm):
        mock_call_llm.return_value = {
            "choices": [{
                "message": {
                    "content": "<think>private chain</think>",
                    "reasoning_content": "visible fallback",
                }
            }],
        }

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "visible fallback"
        assert "private chain" not in result["text"]

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_dict_openai_text_response(self, mock_call_llm):
        mock_call_llm.return_value = {"choices": [{"text": "legacy text"}]}

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "legacy text"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_unknown_dict_shape_fails_closed(self, mock_call_llm):
        marker = object()
        mock_call_llm.return_value = {"unexpected": marker}

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "empty or malformed response" in result["error"]
        assert "No alternate provider was tried" in result["error"]

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_empty_choices(self, mock_call_llm):
        """Response with empty choices list."""
        mock_call_llm.return_value = FakeResponse(
            choices=[],
            model="gpt-4o",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "empty or malformed response" in result["error"]

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_str_response_fallback(self, mock_call_llm):
        """Response as a plain string (last-resort fallback)."""
        mock_call_llm.return_value = "raw string response"

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "raw string response"


class TestLlmCallUsageExtraction:
    """Test usage extraction from various usage object shapes."""

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_usage_model_dump(self, mock_call_llm):
        """Usage with model_dump() (Pydantic v2)."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
            usage=FakeUsage(prompt=10, completion=20, total=30),
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_usage_dict_method(self, mock_call_llm):
        """Usage with .dict() (Pydantic v1)."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
            usage=FakeUsageV1(prompt=5, completion=10, total=15),
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["completion_tokens"] == 10
        assert result["usage"]["total_tokens"] == 15

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_usage_bare_attributes(self, mock_call_llm):
        """Usage with only attribute access (no model_dump/dict)."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
            usage=FakeUsageBare(prompt=1, completion=2, total=3),
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["usage"]["prompt_tokens"] == 1
        assert result["usage"]["completion_tokens"] == 2
        assert result["usage"]["total_tokens"] == 3

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_usage_dict_direct(self, mock_call_llm):
        """Usage as a plain dict."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["total_tokens"] == 300

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_no_usage(self, mock_call_llm):
        """Response with no usage attribute."""
        # Use a SimpleNamespace to avoid MagicMock's auto-attribute creation
        from types import SimpleNamespace
        resp = SimpleNamespace()
        resp.choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]
        resp.model = "gpt-4o"
        # No usage attribute at all
        mock_call_llm.return_value = resp

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["usage"] == {}


class TestLlmCallProviderProvenance:
    """Test that routing intent is not mislabeled as the serving provider."""

    @patch("hermes_herald.tools._create_strict_llm_stream")
    @patch("agent.auxiliary_client._read_main_provider")
    def test_configured_provider_is_separate_from_actual(self, mock_read_main, mock_call_llm):
        mock_read_main.return_value = "ollama-cloud"
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="deepseek-v4-flash",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["provider"] == ""
        assert result["requested_provider"] == ""
        assert result["configured_provider"] == "ollama-cloud"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    @patch("agent.auxiliary_client._read_main_provider")
    def test_requested_provider_is_not_claimed_as_actual(self, mock_read_main, mock_call_llm):
        mock_read_main.return_value = "openai-codex"
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="claude-sonnet-4",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "provider": "openai-codex",
        }))

        assert result["provider"] == ""
        assert result["requested_provider"] == "openai-codex"
        assert result["configured_provider"] == "openai-codex"
        assert mock_read_main.call_count == 2

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_explicit_response_provider_is_reported_as_actual(self, mock_call_llm):
        response = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )
        setattr(response, "provider", "openai")
        mock_call_llm.return_value = response

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "provider": "openai-codex",
        }))

        assert result["provider"] == "openai"
        assert result["requested_provider"] == "openai-codex"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    @patch("agent.auxiliary_client._read_main_provider")
    def test_configured_provider_lookup_failure_fails_closed(self, mock_read_main, mock_call_llm):
        mock_read_main.side_effect = ImportError("not in Hermes session")
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "No inference request was sent" in result["error"]
        mock_call_llm.assert_not_called()


class TestLlmCallModelOverride:
    """Test model resolution from response vs override."""

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_model_from_response(self, mock_call_llm):
        """Model is read from response.model when no override."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o-2024-08-06",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["model"] == "gpt-4o-2024-08-06"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_model_override_wins(self, mock_call_llm):
        """Explicit model override is used even when response.model differs."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o-2024-08-06",
        )

        with patch.object(
            tools,
            "_resolve_llm_call_route",
            return_value=tools._LlmCallRoute(
                provider="anthropic",
                model="claude-sonnet-4",
                base_url="https://api.anthropic.com",
                api_key="test-token",
                api_mode="chat_completions",
            ),
        ):
            result = json.loads(tools.handle_llm_call({
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "claude-sonnet-4",
            }))

        # response.model still wins (it's the actual model that served the request)
        assert result["model"] == "gpt-4o-2024-08-06"


class TestLlmCallSystemPrompt:
    """Test system prompt handling."""

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_system_prompt_prepended(self, mock_call_llm):
        """System prompt is prepended as a system-role message."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )

        tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "system_prompt": "You are a helpful assistant.",
        })

        # Verify call_llm received the system message first
        call_args = mock_call_llm.call_args[1]
        assert call_args["messages"][0] == {
            "role": "system", "content": "You are a helpful assistant."
        }
        assert call_args["messages"][1] == {"role": "user", "content": "Hi"}

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_empty_system_prompt_is_still_prepended(self, mock_call_llm):
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )

        tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "system_prompt": "",
        })

        call_args = mock_call_llm.call_args[1]
        assert call_args["messages"][0] == {"role": "system", "content": ""}

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_no_system_prompt(self, mock_call_llm):
        """Without system_prompt, messages pass through unchanged."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )

        tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_args = mock_call_llm.call_args[1]
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0] == {"role": "user", "content": "Hi"}


class TestLlmCallJsonMode:
    """Test cross-provider JSON-mode construction and validation."""

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_json_mode_adds_instruction_extra_body_and_validates(self, mock_call_llm):
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content='{"ok": true}')],
            model="gpt-4o",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Return JSON"}],
            "json_mode": True,
        }))

        call_args = mock_call_llm.call_args[1]
        assert call_args["extra_body"] == {
            "response_format": {"type": "json_object"}
        }
        assert call_args["messages"][0]["role"] == "system"
        assert "valid JSON object" in call_args["messages"][0]["content"]
        assert result["text"] == '{"ok":true}'

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_json_mode_combines_existing_system_message_without_mutation(self, mock_call_llm):
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content='```json\n{"ok": true}\n```')],
        )
        messages = [
            {"role": "system", "content": "Keep keys short."},
            {"role": "user", "content": "Return JSON"},
        ]

        result = json.loads(tools.handle_llm_call({
            "messages": messages,
            "json_mode": True,
        }))

        sent_system = mock_call_llm.call_args[1]["messages"][0]["content"]
        assert sent_system.startswith("Keep keys short.\n\n")
        assert "valid JSON object" in sent_system
        assert messages[0]["content"] == "Keep keys short."
        assert result["text"] == '{"ok":true}'

    @pytest.mark.parametrize("model_text", ["not json", "[1, 2, 3]"])
    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_json_mode_rejects_non_object_output(self, mock_call_llm, model_text):
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content=model_text)],
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Return JSON"}],
            "json_mode": True,
        }))

        assert result["status"] == "error"
        assert "json_mode" in result["error"]

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_no_json_mode_no_extra_body(self, mock_call_llm):
        """json_mode=False (default) does not set extra_body."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )

        tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_args = mock_call_llm.call_args[1]
        assert call_args["extra_body"] is None


class TestLlmCallErrorHandling:
    """Test error handling in handle_llm_call."""

    def test_empty_messages(self):
        """Empty messages list returns error."""
        result = json.loads(tools.handle_llm_call({"messages": []}))
        assert result["status"] == "error"
        assert "messages" in result["error"]

    def test_missing_messages(self):
        """Missing messages key returns error."""
        result = json.loads(tools.handle_llm_call({}))
        assert result["status"] == "error"
        assert "messages" in result["error"]

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_call_llm_failure(self, mock_call_llm):
        """call_llm raising an exception returns a clean error."""
        mock_call_llm.side_effect = RuntimeError("API timeout")

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "RuntimeError" in result["error"]
        assert "API timeout" in result["error"]

    def test_import_error_handling(self):
        """ImportError for call_llm gives a clear message."""
        # Simulate ImportError by making the import fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "agent":
                raise ImportError("No module named 'agent'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            result = json.loads(tools.handle_llm_call({
                "messages": [{"role": "user", "content": "Hi"}],
            }))
        finally:
            builtins.__import__ = original_import

        assert result["status"] == "error"
        assert "Hermes agent session" in result["error"]


class TestLlmCallInputValidation:
    """Test handler validation for malformed direct calls."""

    @pytest.mark.parametrize("messages", ["not-a-list", ["not-an-object"]])
    def test_messages_must_be_a_list_of_objects(self, messages):
        result = json.loads(tools.handle_llm_call({"messages": messages}))
        assert result["status"] == "error"

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "invalid", "content": "Hi"},
            {"role": "user", "content": 123},
        ],
    )
    def test_message_fields_are_validated(self, message):
        result = json.loads(tools.handle_llm_call({"messages": [message]}))
        assert result["status"] == "error"

    def test_system_prompt_and_system_message_are_mutually_exclusive(self):
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "system", "content": "First"}],
            "system_prompt": "Second",
        }))
        assert result["status"] == "error"
        assert "either" in result["error"]

    def test_empty_system_prompt_and_system_message_are_mutually_exclusive(self):
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "system", "content": "First"}],
            "system_prompt": "",
        }))
        assert result["status"] == "error"
        assert "either" in result["error"]

    def test_system_prompt_must_be_a_string(self):
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "system_prompt": 123,
        }))
        assert result["status"] == "error"
        assert "system_prompt" in result["error"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("model", 123), ("provider", ["openrouter"]), ("json_mode", "true")],
    )
    def test_optional_parameter_types_are_validated(self, field, value):
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            field: value,
        }))
        assert result["status"] == "error"
        assert field in result["error"]

    def test_schema_rejects_empty_messages(self):
        messages_schema = tools.LLM_CALL_SCHEMA["parameters"]["properties"]["messages"]
        assert messages_schema["minItems"] == 1

    def test_schema_directs_discovery_and_promises_no_provider_fallback(self):
        description = tools.LLM_CALL_SCHEMA["description"]
        assert "list_profile_models" in description
        assert "never falls back to another provider" in description

    @pytest.mark.parametrize("temperature", [-0.1, 2.1, True, "cold"])
    def test_temperature_range_is_validated(self, temperature):
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": temperature,
        }))
        assert result["status"] == "error"

    @pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
    def test_max_tokens_is_a_positive_integer(self, max_tokens):
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": max_tokens,
        }))
        assert result["status"] == "error"


class TestLlmCallParameterPassthrough:
    """Test that parameters are correctly passed to call_llm."""

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_model_override_uses_active_provider_model_resolver(
        self, mock_call_llm, monkeypatch
    ):
        """Human model names resolve to an exact wire slug before inference."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-5.6-sol",
        )
        switch = _install_route_resolution_fakes(monkeypatch)

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt5.6sol",
        }))

        route = mock_call_llm.call_args.args[1]
        assert route.provider == "openai-codex"
        assert route.model == "gpt-5.6-sol"
        assert result["requested_model"] == "gpt5.6sol"
        assert result["resolved_provider"] == "openai-codex"
        assert result["resolved_model"] == "gpt-5.6-sol"
        assert switch.call_args.kwargs["explicit_provider"] == "openai-codex"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_incomplete_executable_route_fails_before_inference(
        self, mock_call_llm, monkeypatch
    ):
        from types import SimpleNamespace
        import hermes_cli.model_switch as model_switch

        _install_route_resolution_fakes(
            monkeypatch,
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            resolved_provider="ollama-cloud",
            resolved_model="glm-5.2",
            available_models=["glm-5.2"],
        )
        monkeypatch.setattr(
            model_switch,
            "switch_model",
            lambda **kwargs: SimpleNamespace(
                success=True,
                new_model="glm-5.2",
                target_provider="ollama-cloud",
                base_url="https://ollama.example/v1",
                api_key="",
                api_mode="chat_completions",
                error_message="",
            ),
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "api_key missing" in result["error"]
        assert "No inference request was sent" in result["error"]
        mock_call_llm.assert_not_called()

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_no_overrides_pin_the_profile_default_route(
        self, mock_call_llm, monkeypatch
    ):
        """An omitted route must never enter auxiliary_client's auto chain."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="glm-5.2",
        )
        _install_route_resolution_fakes(
            monkeypatch,
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            resolved_provider="ollama-cloud",
            resolved_model="glm-5.2",
            available_models=["glm-5.2", "qwen3.5:397b"],
        )
        import agent.auxiliary_client as auxiliary_client
        monkeypatch.setattr(
            auxiliary_client, "_read_main_provider", MagicMock(return_value="ollama-cloud")
        )
        monkeypatch.setattr(
            auxiliary_client, "_read_main_model", MagicMock(return_value="glm-5.2")
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["resolved_provider"] == "ollama-cloud"
        assert result["resolved_model"] == "glm-5.2"
        route = mock_call_llm.call_args.args[1]
        assert route.provider == "ollama-cloud"
        assert route.model == "glm-5.2"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_llm_call_uses_no_fallback_streaming_path(
        self, mock_call_llm, monkeypatch
    ):
        """Strict routing must bypass auxiliary_client's provider fallbacks."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="glm-5.2",
        )
        _install_route_resolution_fakes(
            monkeypatch,
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            resolved_provider="ollama-cloud",
            resolved_model="glm-5.2",
            available_models=["glm-5.2"],
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "glm-5.2",
        }))

        assert result["text"] == "ok"
        mock_call_llm.assert_called_once()

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_stream_chunks_are_aggregated_and_closed(
        self, mock_call_llm, monkeypatch
    ):
        from types import SimpleNamespace

        class ClosingStream:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                yield SimpleNamespace(
                    model="glm-5.2",
                    usage=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(
                        content="HERALD-", reasoning=None,
                    ))],
                )
                yield SimpleNamespace(
                    model="glm-5.2",
                    usage=FakeUsage(prompt=2, completion=3, total=5),
                    choices=[SimpleNamespace(delta=SimpleNamespace(
                        content="OK", reasoning=None,
                    ))],
                )

            def close(self):
                self.closed = True

        stream = ClosingStream()
        mock_call_llm.return_value = stream
        _install_route_resolution_fakes(
            monkeypatch,
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            resolved_provider="ollama-cloud",
            resolved_model="glm-5.2",
            available_models=["glm-5.2"],
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["text"] == "HERALD-OK"
        assert result["model"] == "glm-5.2"
        assert result["usage"]["total_tokens"] == 5
        assert stream.closed is True

    def test_reasoning_only_stream_and_empty_stream_contract(self):
        from types import SimpleNamespace

        reasoning = iter([SimpleNamespace(
            model="reasoning-model",
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(
                content=None, reasoning_content="reasoned answer",
            ))],
        )])
        response = tools._aggregate_llm_call_stream(
            reasoning,
            model="requested-model",
        )
        assert response["model"] == "reasoning-model"
        assert response["choices"][0]["message"]["content"] == ""
        assert response["choices"][0]["message"]["reasoning"] == "reasoned answer"

        with pytest.raises(RuntimeError, match="empty or malformed stream"):
            tools._aggregate_llm_call_stream(iter(()), model="requested-model")

    def test_stream_aggregation_enforces_wall_clock_timeout(self):
        import threading

        class BlockingStream:
            def __init__(self):
                self.released = threading.Event()
                self.closed = False

            def __iter__(self):
                self.released.wait()
                return
                yield

            def close(self):
                self.closed = True
                self.released.set()

        stream = BlockingStream()
        with pytest.raises(TimeoutError, match="timed out"):
            tools._aggregate_llm_call_stream(
                stream,
                model="requested-model",
                total_ceiling=0.01,
            )
        assert stream.closed is True

    def test_blocking_stream_close_cannot_extend_wall_clock_timeout(self):
        import threading
        import time

        class BlockingCloseStream:
            def __iter__(self):
                threading.Event().wait()
                yield  # pragma: no cover

            def close(self):
                threading.Event().wait()

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            tools._aggregate_llm_call_stream(
                BlockingCloseStream(),
                model="requested-model",
                total_ceiling=0.01,
            )
        assert time.monotonic() - started < 0.1

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_selected_route_failure_is_noisy_and_not_retried(
        self, mock_call_llm, monkeypatch
    ):
        """A failed strict stream surfaces its error after one inference call."""
        def broken_stream():
            raise RuntimeError("selected Ollama route unavailable")
            yield  # pragma: no cover - makes this a stream iterator

        mock_call_llm.return_value = broken_stream()
        _install_route_resolution_fakes(
            monkeypatch,
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            resolved_provider="ollama-cloud",
            resolved_model="glm-5.2",
            available_models=["glm-5.2"],
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "selected Ollama route unavailable" in result["error"]
        mock_call_llm.assert_called_once()

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_selected_route_error_redacts_api_key(self, mock_call_llm):
        mock_call_llm.side_effect = RuntimeError(
            "401 reflected Authorization: Bearer test-token"
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "test-token" not in result["error"]
        assert "[REDACTED]" in result["error"]

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_selected_route_error_redacts_short_api_key(self, mock_call_llm, monkeypatch):
        _install_route_resolution_fakes(
            monkeypatch,
            resolved_api_key="abc",
        )
        mock_call_llm.side_effect = RuntimeError(
            "401 reflected Authorization: Bearer abc"
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        }))

        assert result["status"] == "error"
        assert "abc" not in result["error"]
        assert "[REDACTED]" in result["error"]

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_unconfigured_openrouter_route_is_rejected_before_inference(
        self, mock_call_llm, monkeypatch
    ):
        """Ambient OpenRouter credentials are not a profile route declaration."""
        _install_route_resolution_fakes(
            monkeypatch,
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            resolved_provider="openrouter",
            resolved_model="z-ai/glm-5.2",
            available_models=["z-ai/glm-5.2"],
            resolved_base_url="https://openrouter.ai/api/v1",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "provider": "openrouter",
            "model": "glm-5.2",
        }))

        assert result["status"] == "error"
        assert "not configured for this profile" in result["error"]
        assert "ollama-cloud" in result["error"]
        mock_call_llm.assert_not_called()

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_alternate_provider_requires_an_explicit_model(
        self, mock_call_llm, monkeypatch
    ):
        _install_route_resolution_fakes(monkeypatch)

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "provider": "anthropic",
        }))

        assert result["status"] == "error"
        assert "requires an explicit model" in result["error"]
        mock_call_llm.assert_not_called()

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_family_model_resolves_to_active_advertised_variant(
        self, mock_call_llm, monkeypatch
    ):
        """A bare family name reuses the active advertised family variant."""
        _install_route_resolution_fakes(
            monkeypatch,
            resolved_model="gpt-5.6",
        )

        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-5.6-sol",
        )
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-5.6",
        }))

        assert result["text"] == "ok"
        route = mock_call_llm.call_args.args[1]
        assert route.provider == "openai-codex"
        assert route.model == "gpt-5.6-sol"

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_openai_provider_alias_cannot_silently_select_openrouter(
        self, mock_call_llm, monkeypatch
    ):
        """The ambiguous OpenAI alias must not turn a Codex request into OR."""
        _install_route_resolution_fakes(
            monkeypatch,
            resolved_provider="openrouter",
            resolved_model="openai/gpt-5.6-sol",
            available_models=["openai/gpt-5.6-sol"],
            resolved_base_url="https://openrouter.ai/api/v1",
        )

        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="wrong route")],
            model="openai/gpt-5.6-sol",
        )
        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "provider": "openai",
            "model": "gpt5.6sol",
        }))

        assert result["status"] == "error"
        assert "OpenRouter" in result["error"]
        assert "openai-codex" in result["error"]
        mock_call_llm.assert_not_called()

    def test_openai_provider_alias_without_model_is_rejected_before_inference(
        self
    ):
        import agent.auxiliary_client as auxiliary_client

        with patch.object(auxiliary_client, "call_llm") as mock_call_llm:
            mock_call_llm.return_value = FakeResponse(
                choices=[FakeChoice(message_content="wrong route")],
                model="gpt-5.6-sol",
            )
            result = json.loads(tools.handle_llm_call({
                "messages": [{"role": "user", "content": "Hi"}],
                "provider": "openai",
            }))

        assert result["status"] == "error"
        assert "openai-codex" in result["error"]
        assert "openrouter" in result["error"]
        mock_call_llm.assert_not_called()

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_unadvertised_model_is_rejected_before_inference(
        self, mock_call_llm, monkeypatch
    ):
        _install_route_resolution_fakes(
            monkeypatch,
            resolved_model="gpt-5.7",
        )

        result = json.loads(tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-5.7",
        }))

        assert result["status"] == "error"
        assert "not advertised" in result["error"]
        assert "gpt-5.6-sol" in result["error"]
        mock_call_llm.assert_not_called()

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_temperature_passthrough(self, mock_call_llm):
        """Temperature is passed through to call_llm."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )

        tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "max_tokens": 100,
        })

        call_args = mock_call_llm.call_args[1]
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 100

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_provider_and_model_passthrough(self, mock_call_llm):
        """Resolved provider and model are passed to call_llm."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="claude-sonnet-4",
        )

        with patch.object(
            tools,
            "_resolve_llm_call_route",
            return_value=tools._LlmCallRoute(
                provider="anthropic",
                model="claude-sonnet-4",
                base_url="https://api.anthropic.com",
                api_key="test-token",
                api_mode="chat_completions",
            ),
        ) as resolve_route:
            tools.handle_llm_call({
                "messages": [{"role": "user", "content": "Hi"}],
                "provider": "anthropic",
                "model": "claude-sonnet-4",
            })

        route = mock_call_llm.call_args.args[1]
        assert route.provider == "anthropic"
        assert route.model == "claude-sonnet-4"
        assert route.base_url == "https://api.anthropic.com"
        assert route.api_key == "test-token"
        assert route.api_mode == "chat_completions"
        resolve_route.assert_called_once_with("claude-sonnet-4", "anthropic")

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_provider_and_model_are_normalized(self, mock_call_llm):
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="resolved-model",
        )

        with patch.object(
            tools,
            "_resolve_llm_call_route",
            return_value=tools._LlmCallRoute(
                provider="openai-codex",
                model="requested-model",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="test-token",
                api_mode="codex_responses",
            ),
        ) as resolve_route:
            result = json.loads(tools.handle_llm_call({
                "messages": [{"role": "user", "content": "Hi"}],
                "provider": "   ",
                "model": "  requested-model  ",
            }))

        route = mock_call_llm.call_args.args[1]
        assert route.provider == "openai-codex"
        assert route.model == "requested-model"
        assert result["requested_provider"] == ""
        resolve_route.assert_called_once_with("requested-model", "")

    @patch("hermes_herald.tools._create_strict_llm_stream")
    def test_timeout_passthrough(self, mock_call_llm):
        """Timeout is always 120s."""
        mock_call_llm.return_value = FakeResponse(
            choices=[FakeChoice(message_content="ok")],
            model="gpt-4o",
        )

        tools.handle_llm_call({
            "messages": [{"role": "user", "content": "Hi"}],
        })

        call_args = mock_call_llm.call_args[1]
        assert call_args["timeout"] == 120.0

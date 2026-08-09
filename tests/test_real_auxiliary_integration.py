"""Real-import compatibility test against the installed Hermes runtime."""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


HERMES_CORE = Path(os.environ.get(
    "HERMES_SOURCE_DIR",
    os.path.expanduser("~/.hermes/hermes-agent"),
))
PLUGIN_DIR = Path(os.environ.get(
    "HERMES_HERALD_PLUGIN_DIR",
    Path(__file__).resolve().parent.parent,
))


def _resolve_hermes_python() -> Path:
    """Find an interpreter that has the real Hermes dependencies installed."""
    explicit = os.environ.get("HERMES_RUNTIME_PYTHON")
    if explicit:
        return Path(os.path.expanduser(explicit))

    candidates = (
        HERMES_CORE / ".venv" / "bin" / "python",
        HERMES_CORE / "venv" / "bin" / "python",
        HERMES_CORE / ".venv" / "Scripts" / "python.exe",
        HERMES_CORE / "venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    )
    return next((candidate for candidate in candidates if candidate.is_file()), Path(sys.executable))


HERMES_PYTHON = _resolve_hermes_python()


def test_streaming_route_bundle_does_not_request_auto_client(monkeypatch):
    """A supplied strict route must fail instead of entering core auto discovery."""
    from agent import auxiliary_client as aux

    requested = []

    def unavailable(provider, model=None, **kwargs):
        requested.append(provider)
        return None, model

    monkeypatch.setattr(aux, "_get_cached_client", unavailable)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        aux.call_llm(
            task=None,
            provider="custom",
            model="strict-model",
            base_url="https://strict.invalid/v1",
            api_key="test-token",
            api_mode="chat_completions",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

    assert requested == ["custom"]


def test_stream_iterator_failure_bypasses_core_fallback_helpers(monkeypatch):
    """Failures after stream creation propagate without payment/provider fallback."""
    from types import SimpleNamespace
    from agent import auxiliary_client as aux

    def broken_stream():
        raise ConnectionError("selected route disconnected")
        yield

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: broken_stream())
        )
    )
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda provider, model=None, **kwargs: (client, model),
    )
    monkeypatch.setattr(
        aux,
        "_try_payment_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("payment fallback must not run")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_try_configured_fallback_chain",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("configured fallback must not run")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_try_main_agent_model_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("main-agent fallback must not run")
        ),
    )

    stream = aux.call_llm(
        task=None,
        provider="custom",
        model="strict-model",
        base_url="https://strict.invalid/v1",
        api_key="test-token",
        api_mode="chat_completions",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    with pytest.raises(ConnectionError, match="selected route disconnected"):
        list(stream)


@pytest.mark.parametrize(
    "message",
    [
        "401 unauthorized",
        "402 insufficient credits",
        "429 rate limit",
        "model is incompatible with route",
    ],
)
def test_stream_creation_failures_bypass_core_fallback_helpers(
    monkeypatch, message
):
    """Classified setup failures unwind before core's fallback chain."""
    from types import SimpleNamespace
    from agent import auxiliary_client as aux

    def fail_create(**kwargs):
        raise RuntimeError(message)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fail_create)
        )
    )
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda provider, model=None, **kwargs: (client, model),
    )
    monkeypatch.setattr(
        aux,
        "_try_payment_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider fallback must not run")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_try_configured_fallback_chain",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("configured fallback must not run")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_try_main_agent_model_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("main-agent fallback must not run")
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        aux.call_llm(
            task=None,
            provider="custom",
            model="strict-model",
            base_url="https://strict.invalid/v1",
            api_key="test-token",
            api_mode="chat_completions",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )


def test_plugin_strict_creation_bypasses_call_llm_and_requests_usage(monkeypatch):
    """Herald dispatches directly on the selected client, before core fallback code."""
    from types import SimpleNamespace
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    captured = {}
    client_requests = []

    def create(**kwargs):
        captured.update(kwargs)
        return iter(())

    client = SimpleNamespace(
        base_url="https://strict.example/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        aux,
        "call_llm",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("strict creation must not enter call_llm")
        ),
    )
    def get_client(provider, model=None, **kwargs):
        client_requests.append((provider, kwargs))
        return client, model

    monkeypatch.setattr(aux, "_get_cached_client", get_client)
    monkeypatch.setattr(
        aux,
        "_build_call_kwargs",
        lambda provider, model, messages, **kwargs: {
            "model": model,
            "messages": messages,
        },
    )

    stream = tools._create_strict_llm_stream(
        aux,
        tools._LlmCallRoute(
            provider="custom:strict",
            model="strict-model",
            base_url="https://strict.example/v1",
            api_key="test-token",
            api_mode="chat_completions",
        ),
        messages=[{"role": "user", "content": "Hi"}],
        temperature=None,
        max_tokens=None,
        extra_body=None,
        timeout=120.0,
    )

    assert list(stream) == []
    assert client_requests[0][0] == "custom"
    assert client_requests[0][1]["base_url"] == "https://strict.example/v1"
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    ("selected_endpoint", "client_endpoint"),
    [
        ("https://selected.invalid/v1", "https://openrouter.ai/api/v1"),
        (
            "https://selected.invalid/v1?tenant=selected",
            "https://selected.invalid/v1?tenant=other",
        ),
    ],
)
def test_plugin_strict_creation_rejects_endpoint_mutation(
    monkeypatch, selected_endpoint, client_endpoint
):
    """Client resolution may not replace the preflighted endpoint."""
    from types import SimpleNamespace
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    escaped_client = SimpleNamespace(
        base_url=client_endpoint,
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter(()))
        ),
    )
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda provider, model=None, **kwargs: (escaped_client, model),
    )
    monkeypatch.setattr(aux, "_build_call_kwargs", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="changed endpoint"):
        tools._create_strict_llm_stream(
            aux,
            tools._LlmCallRoute(
                provider="custom:auto",
                model="strict-model",
                base_url=selected_endpoint,
                api_key="test-token",
                api_mode="chat_completions",
            ),
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=None,
            extra_body=None,
            timeout=120.0,
        )


def test_plugin_strict_creation_preserves_custom_anthropic_endpoint(monkeypatch):
    """Named custom Anthropic routes keep their raw /anthropic endpoint."""
    from types import SimpleNamespace
    from agent import anthropic_adapter
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    built = {}
    captured = {}
    real_client = object()

    def build_anthropic_client(api_key, base_url, timeout=None):
        built.update(api_key=api_key, base_url=base_url, timeout=timeout)
        return real_client

    def wrapped_client(real, model, api_key, base_url, is_oauth=False):
        assert real is real_client
        built.update(
            model=model,
            wrapper_api_key=api_key,
            wrapper_base_url=base_url,
            is_oauth=is_oauth,
        )
        def close():
            built["closed"] = True
            time.sleep(3.0)

        return SimpleNamespace(
            base_url=base_url,
            close=close,
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: captured.update(kwargs) or iter(())
                )
            ),
        )

    monkeypatch.setattr(anthropic_adapter, "build_anthropic_client", build_anthropic_client)
    monkeypatch.setattr(aux, "AnthropicAuxiliaryClient", wrapped_client)
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("custom Anthropic route must not enter generic routing")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_build_call_kwargs",
        lambda provider, model, messages, **kwargs: {
            "model": model,
            "messages": messages,
        },
    )

    endpoint = "https://anthropic-custom.selected.example/anthropic"
    started = time.monotonic()
    stream = tools._create_strict_llm_stream(
        aux,
        tools._LlmCallRoute(
            provider="custom:anthropic-wrapper",
            model="claude-test",
            base_url=endpoint,
            api_key="test-token",
            api_mode="anthropic_messages",
        ),
        messages=[{"role": "user", "content": "Hi"}],
        temperature=None,
        max_tokens=16,
        extra_body=None,
        timeout=0.01,
    )

    assert list(stream) == []
    assert time.monotonic() - started < 2.0
    assert built["base_url"] == endpoint
    assert built["wrapper_base_url"] == endpoint
    assert built["api_key"] == "test-token"
    assert built["wrapper_api_key"] == "test-token"
    assert built["timeout"] == 0.01
    assert built["closed"] is True
    assert captured["stream"] is True


def test_custom_anthropic_request_cannot_outlive_total_deadline(monkeypatch):
    """A progressively active native Anthropic call still has a hard deadline."""
    from types import SimpleNamespace
    from agent import anthropic_adapter
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    endpoint = "https://anthropic-custom.selected.example/anthropic"

    monkeypatch.setattr(
        anthropic_adapter,
        "build_anthropic_client",
        lambda api_key, base_url, timeout=None: object(),
    )

    def wrapped_client(real, model, api_key, base_url, is_oauth=False):
        def create(**kwargs):
            time.sleep(3.0)
            return "late-success"

        return SimpleNamespace(
            base_url=base_url,
            close=lambda: None,
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

    monkeypatch.setattr(aux, "AnthropicAuxiliaryClient", wrapped_client)
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("custom Anthropic route must not enter generic routing")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_build_call_kwargs",
        lambda provider, model, messages, **kwargs: {
            "model": model,
            "messages": messages,
        },
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="strict deadline"):
        tools._create_strict_llm_stream(
            aux,
            tools._LlmCallRoute(
                provider="custom:anthropic-wrapper",
                model="claude-test",
                base_url=endpoint,
                api_key="test-token",
                api_mode="anthropic_messages",
            ),
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=16,
            extra_body=None,
            timeout=0.02,
        )

    assert time.monotonic() - started < 2.0


def test_custom_anthropic_late_queue_wakeup_cannot_accept_late_success(monkeypatch):
    """A scheduler wake-up race cannot turn a post-deadline result into success."""
    from types import SimpleNamespace
    from agent import anthropic_adapter
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    endpoint = "https://anthropic-custom.selected.example/anthropic"

    class LateDeliveringQueue:
        def __init__(self, maxsize=0):
            self.item = None

        def put(self, item):
            self.item = item

        def get(self, timeout=None):
            while self.item is None:
                time.sleep(0.001)
            return self.item

    monkeypatch.setattr(tools.queue, "Queue", LateDeliveringQueue)
    monkeypatch.setattr(
        anthropic_adapter,
        "build_anthropic_client",
        lambda api_key, base_url, timeout=None: object(),
    )

    def wrapped_client(real, model, api_key, base_url, is_oauth=False):
        def create(**kwargs):
            time.sleep(0.03)
            return "late-success"

        return SimpleNamespace(
            base_url=base_url,
            close=lambda: None,
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

    monkeypatch.setattr(aux, "AnthropicAuxiliaryClient", wrapped_client)
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("custom Anthropic route must not enter generic routing")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_build_call_kwargs",
        lambda provider, model, messages, **kwargs: {
            "model": model,
            "messages": messages,
        },
    )

    with pytest.raises(TimeoutError, match="strict deadline"):
        tools._create_strict_llm_stream(
            aux,
            tools._LlmCallRoute(
                provider="custom:anthropic-wrapper",
                model="claude-test",
                base_url=endpoint,
                api_key="test-token",
                api_mode="anthropic_messages",
            ),
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=16,
            extra_body=None,
            timeout=0.02,
        )


def test_custom_anthropic_request_error_survives_bounded_raising_close(monkeypatch):
    """Owned-client cleanup cannot mask or prolong the provider failure."""
    from types import SimpleNamespace
    from agent import anthropic_adapter
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    endpoint = "https://anthropic-custom.selected.example/anthropic"

    monkeypatch.setattr(
        anthropic_adapter,
        "build_anthropic_client",
        lambda api_key, base_url, timeout=None: object(),
    )

    def wrapped_client(real, model, api_key, base_url, is_oauth=False):
        def create(**kwargs):
            raise RuntimeError("provider-failure")

        def close():
            time.sleep(3.0)
            raise RuntimeError("close-failure")

        return SimpleNamespace(
            base_url=base_url,
            close=close,
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

    monkeypatch.setattr(aux, "AnthropicAuxiliaryClient", wrapped_client)
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("custom Anthropic route must not enter generic routing")
        ),
    )
    monkeypatch.setattr(
        aux,
        "_build_call_kwargs",
        lambda provider, model, messages, **kwargs: {
            "model": model,
            "messages": messages,
        },
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="provider-failure"):
        tools._create_strict_llm_stream(
            aux,
            tools._LlmCallRoute(
                provider="custom:anthropic-wrapper",
                model="claude-test",
                base_url=endpoint,
                api_key="test-token",
                api_mode="anthropic_messages",
            ),
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=16,
            extra_body=None,
            timeout=0.02,
        )

    assert time.monotonic() - started < 2.0


def test_plugin_strict_creation_rejects_unverifiable_endpoint(monkeypatch):
    """Matching malformed endpoint strings are not treated as verified URLs."""
    from types import SimpleNamespace
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    calls = []
    client = SimpleNamespace(
        base_url="not-a-valid-http-endpoint",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs))
        ),
    )
    monkeypatch.setattr(
        aux,
        "_get_cached_client",
        lambda provider, model=None, **kwargs: (client, model),
    )
    monkeypatch.setattr(aux, "_build_call_kwargs", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="cannot be verified"):
        tools._create_strict_llm_stream(
            aux,
            tools._LlmCallRoute(
                provider="custom:invalid",
                model="strict-model",
                base_url="not-a-valid-http-endpoint",
                api_key="test-token",
                api_mode="chat_completions",
            ),
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=None,
            extra_body=None,
            timeout=120.0,
        )

    assert calls == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com:/v1",
        "https://example.com:0/v1",
        "https://-/v1",
    ],
)
def test_strict_endpoint_identity_rejects_malformed_authority(endpoint):
    """Ambiguous ports and malformed DNS labels are not executable endpoints."""
    from hermes_herald import tools  # type: ignore[import-not-found]

    assert tools._strict_base_url_identity(endpoint) is None


def test_real_auxiliary_import_and_reasoning_extraction(tmp_path):
    """Load real auxiliary_client and exercise the plugin's production imports."""
    script = textwrap.dedent(
        f"""
        import importlib.util
        import inspect
        import json
        import os
        import sys
        import types
        from types import SimpleNamespace

        sys.path.insert(0, {str(HERMES_CORE)!r})
        os.environ["HERMES_HOME"] = {str(tmp_path)!r}

        from agent import auxiliary_client as aux

        signature = inspect.signature(aux.call_llm)
        assert "messages" in signature.parameters
        assert "extra_body" in signature.parameters
        assert callable(aux.extract_content_or_reasoning)

        merged = aux._build_call_kwargs(
            "openai", "test-model", [{{"role": "user", "content": "Hi"}}],
            extra_body={{
                "reasoning": {{"enabled": True}},
                "response_format": {{"type": "json_object"}},
            }},
        )
        assert merged["extra_body"]["reasoning"]["enabled"] is True
        assert merged["extra_body"]["response_format"]["type"] == "json_object"

        # Token caps are intentionally provider-dependent in Hermes core.
        openrouter_kwargs = aux._build_call_kwargs(
            "openrouter", "test-model",
            [{{"role": "user", "content": "Hi"}}],
            max_tokens=123,
        )
        assert "max_tokens" not in openrouter_kwargs
        assert "max_completion_tokens" not in openrouter_kwargs
        anthropic_kwargs = aux._build_call_kwargs(
            "custom", "test-model",
            [{{"role": "user", "content": "Hi"}}],
            max_tokens=123,
            base_url="https://example.invalid/anthropic",
        )
        assert anthropic_kwargs["max_tokens"] == 123

        package = types.ModuleType("hermes_herald_real")
        package.__path__ = [{str(PLUGIN_DIR)!r}]
        sys.modules["hermes_herald_real"] = package
        for name in ("config", "tools"):
            spec = importlib.util.spec_from_file_location(
                f"hermes_herald_real.{{name}}", {str(PLUGIN_DIR)!r} + f"/{{name}}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"hermes_herald_real.{{name}}"] = module
            spec.loader.exec_module(module)

        tools = sys.modules["hermes_herald_real.tools"]
        captured = {{}}
        message = SimpleNamespace(
            content=None,
            reasoning=None,
            reasoning_content='{{"answer":"reasoning-only output"}}',
            reasoning_details=None,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            model="reasoning-model",
            usage=None,
        )

        original_create = tools._create_strict_llm_stream
        original_resolve = tools._resolve_llm_call_route
        def fake_create(auxiliary, route, **kwargs):
            captured.update(
                provider=route.provider,
                model=route.model,
                base_url=route.base_url,
                api_key=route.api_key,
                api_mode=route.api_mode,
            )
            captured.update(kwargs)
            return response
        tools._create_strict_llm_stream = fake_create
        tools._resolve_llm_call_route = lambda provider, model: tools._LlmCallRoute(
            provider="requested-provider",
            model="requested-model",
            base_url="https://example.invalid/v1",
            api_key="test-token",
            api_mode="chat_completions",
        )
        try:
            result = json.loads(tools.handle_llm_call({{
                "messages": [{{"role": "user", "content": "Solve"}}],
                "provider": "requested-provider",
                "json_mode": True,
            }}))
        finally:
            tools._create_strict_llm_stream = original_create
            tools._resolve_llm_call_route = original_resolve

        assert captured["provider"] == "requested-provider"
        assert captured["model"] == "requested-model"
        assert captured["base_url"] == "https://example.invalid/v1"
        assert captured["api_key"] == "test-token"
        assert captured["api_mode"] == "chat_completions"
        assert captured["extra_body"] == {{
            "response_format": {{"type": "json_object"}}
        }}
        assert captured["messages"][0]["role"] == "system"
        assert "valid JSON object" in captured["messages"][0]["content"]
        assert result["text"] == '{{"answer":"reasoning-only output"}}'
        assert result["provider"] == ""
        assert result["requested_provider"] == "requested-provider"
        assert result["configured_provider"] == ""
        print(json.dumps({{"status": "ok", "model": result["model"]}}))
        """
    )

    env = os.environ.copy()
    completed = subprocess.run(
        [str(HERMES_PYTHON), "-c", script],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert json.loads(completed.stdout)["status"] == "ok"

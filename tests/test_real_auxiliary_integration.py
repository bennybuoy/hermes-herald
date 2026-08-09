"""Real-import compatibility test against the installed Hermes runtime."""

import json
import os
import subprocess
import sys
import textwrap
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


def test_plugin_strict_creation_rejects_endpoint_mutation(monkeypatch):
    """Client resolution may not replace the preflighted endpoint."""
    from types import SimpleNamespace
    from agent import auxiliary_client as aux
    from hermes_herald import tools  # type: ignore[import-not-found]

    escaped_client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
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
                base_url="https://selected.invalid/v1",
                api_key="test-token",
                api_mode="chat_completions",
            ),
            messages=[{"role": "user", "content": "Hi"}],
            temperature=None,
            max_tokens=None,
            extra_body=None,
            timeout=120.0,
        )


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

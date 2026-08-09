"""Real-import compatibility test against the installed Hermes runtime."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


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

        original_call = aux.call_llm
        original_resolve = tools._resolve_llm_call_route
        def fake_call_llm(**kwargs):
            captured.update(kwargs)
            return response
        aux.call_llm = fake_call_llm
        tools._resolve_llm_call_route = lambda provider, model: (
            "requested-provider", "requested-model"
        )
        try:
            result = json.loads(tools.handle_llm_call({{
                "messages": [{{"role": "user", "content": "Solve"}}],
                "provider": "requested-provider",
                "json_mode": True,
            }}))
        finally:
            aux.call_llm = original_call
            tools._resolve_llm_call_route = original_resolve

        assert captured["task"] is None
        assert captured["provider"] == "requested-provider"
        assert captured["model"] == "requested-model"
        assert captured["stream"] is True
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

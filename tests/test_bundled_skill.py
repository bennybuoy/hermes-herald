"""Release-contract tests for Hermes Herald's bundled operating skill."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import yaml


PLUGIN_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PLUGIN_DIR / "plugin.yaml"
SKILL_PATH = PLUGIN_DIR / "skills" / "agent-dispatch" / "SKILL.md"


def _load_plugin_entry():
    spec = importlib.util.spec_from_file_location(
        "hermes_herald.plugin_entry",
        PLUGIN_DIR / "__init__.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest():
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_plugin_registers_all_tools_and_bundled_skill():
    plugin = _load_plugin_entry()
    ctx = MagicMock()
    manifest = _load_manifest()

    plugin.register(ctx)

    assert ctx.register_tool.call_count == len(manifest["provides_tools"])
    ctx.register_skill.assert_called_once_with("agent-dispatch", SKILL_PATH)


def test_registered_llm_call_handler_receives_host_llm_facade(monkeypatch):
    plugin = _load_plugin_entry()
    ctx = MagicMock()
    ctx.llm.complete.return_value = SimpleNamespace(
        text="host result",
        provider="openai-codex",
        model="gpt-5.6-sol",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=None,
        ),
    )
    monkeypatch.setattr(
        sys.modules[plugin.handle_llm_call.__module__],
        "_configured_local_model_inventory",
        lambda **kwargs: {},
    )

    plugin.register(ctx)
    llm_registration = next(
        call.kwargs
        for call in ctx.register_tool.call_args_list
        if call.kwargs["name"] == "llm_call"
    )
    result = json.loads(llm_registration["handler"]({
        "messages": [{"role": "user", "content": "ping"}],
    }))

    assert result["text"] == "host result"
    ctx.llm.complete.assert_called_once()


def test_bundled_skill_matches_v1_contract():
    text = SKILL_PATH.read_text(encoding="utf-8")
    manifest = _load_manifest()

    assert f"version: {manifest['version']}" in text
    assert "hermes_herald:" in text
    assert f"{len(manifest['provides_tools'])} tools" in text
    assert "interrupt_after_seconds" in text
    assert "inherit_soul=true" in text
    assert 'skill_view("hermes-herald:agent-dispatch")' in text

    stale_claims = (
        "hard_timeout_seconds",
        "deprecated alias `timeout`",
        "agent_dispatch.profiles",
        "provides ten tools",
        "all ten schemas",
        "per-call `model` override applies only to that call",
        "tests/pytest.ini",
    )
    for claim in stale_claims:
        assert claim not in text


def test_readme_documents_multiplex_scope_and_matching_release():
    readme = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
    version = _load_manifest()["version"]
    assert "### Multiplex support" in readme
    section = readme.split("### Multiplex support", 1)[1].split("\n---", 1)[0]
    assert "multiplexed gateway" in section
    assert "A→B→A" in section
    assert "PluginManager" in section
    assert "shared ledger" in section
    assert "Multiplex limitation (known, unfixed)" not in readme
    assert f"badge/version-{version}-" in readme
    assert f'alt="Version {version}"' in readme

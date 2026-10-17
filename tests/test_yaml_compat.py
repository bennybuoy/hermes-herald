"""YAML configuration loading compatibility without PyYAML."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hermes_herald.yaml_compat import safe_load

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_config_profiles_load_without_pyyaml(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "hermes_herald:\n"
        "  origin_name: origin-test\n"
        "  profiles:\n"
        "    target-test:\n"
        "      url: http://127.0.0.1:9999\n"
        "      capabilities: [dispatch]\n",
        encoding="utf-8",
    )
    module = __import__("hermes_herald.config", fromlist=["list_profiles"])
    assert module.list_profiles() == ["target-test"]
    assert module.get_active_profile_name() == "origin-test"


def test_bundled_manifest_parses_with_host_yaml():
    manifest = safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["name"] == "hermes-herald"
    assert manifest["provides_tools"]


def test_package_import_works_without_plugin_dir_on_sys_path():
    script = """
import importlib.util, sys, types
plugin_dir = sys.argv[1]
sys.path = [p for p in sys.path if p != plugin_dir]
agent = types.ModuleType('agent'); agent.__path__ = []
secret = types.ModuleType('agent.secret_scope'); secret.get_secret = lambda *a, **k: None
constants = types.ModuleType('hermes_constants'); constants.get_hermes_home = lambda: None
sys.modules.update({'agent': agent, 'agent.secret_scope': secret, 'hermes_constants': constants})
spec = importlib.util.spec_from_file_location('hermes_herald', plugin_dir + '/__init__.py', submodule_search_locations=[plugin_dir])
pkg = importlib.util.module_from_spec(spec); sys.modules['hermes_herald'] = pkg; spec.loader.exec_module(pkg)
import hermes_herald.config
"""
    result = subprocess.run([sys.executable, "-c", script, str(PLUGIN_DIR)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", ["host", "legacy"])
def test_yaml_compat_selects_host_or_legacy_loader(monkeypatch, source):
    import sys
    import types

    expected = {"source": source}
    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=lambda _: expected))
    if source == "host":
        host = types.ModuleType("hermes_yaml")
        host.safe_load = lambda _: expected
        monkeypatch.setitem(sys.modules, "hermes_yaml", host)
    else:
        monkeypatch.setitem(sys.modules, "hermes_yaml", None)
    assert safe_load("ignored") == expected


def test_yaml_compat_preserves_parse_errors():
    with pytest.raises(Exception):
        safe_load("key: [unterminated")

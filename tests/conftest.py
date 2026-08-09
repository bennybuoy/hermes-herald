"""Pytest configuration — sets up sys.path and hermes_herald package context."""
import os
import sys
import types
import tempfile
import importlib.util

# Add plugin dir and Hermes source to sys.path
PLUGIN_DIR = os.environ.get(
    "HERMES_HERALD_PLUGIN_DIR",
    os.path.join(os.path.dirname(__file__), ".."),
)
HERMES_SRC = os.environ.get(
    "HERMES_SOURCE_DIR",
    os.path.expanduser("~/.hermes/hermes-agent"),
)
sys.path.insert(0, PLUGIN_DIR)
sys.path.insert(0, HERMES_SRC)

# Set up hermes_herald package context so relative imports work.
# We register the package and manually load each submodule by file path
# so `from hermes_herald import tools` works in tests.
pkg = types.ModuleType("hermes_herald")
pkg.__path__ = [PLUGIN_DIR]
sys.modules["hermes_herald"] = pkg

# Manually load submodules by file path
for name in ("config", "ledger", "tools", "callback"):
    filepath = os.path.join(PLUGIN_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"hermes_herald.{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"hermes_herald.{name}"] = mod
    spec.loader.exec_module(mod)

# Ensure config exists
os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
_cfg_path = os.path.join(os.environ["HERMES_HOME"], "config.yaml")
if not os.path.exists(_cfg_path):
    _state_path = os.path.join(os.environ["HERMES_HOME"], "hermes-herald-runs.json")
    with open(_cfg_path, "w") as f:
        f.write(f"hermes_herald:\n  state_file: {_state_path}\n")
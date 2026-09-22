"""Profile-home and credential isolation contracts for multiplexed origins."""

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path

import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_herald import config


@contextmanager
def home_scope(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def write_home(home, origin):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"hermes_herald:\n  origin_name: {origin}\n"
        f"  profiles:\n    target-{origin}:\n"
        "      url: https://target.invalid\n"
        "      api_key: ${HERALD_TEST_KEY}\n"
        "      capabilities: [dispatch, chat]\n"
        "  llm_direct:\n    enabled: true\n    endpoints:\n      test:\n"
        "        base_url: https://inference.invalid/v1\n"
        "        api_key: ${HERALD_TEST_KEY}\n"
    )
    return home


@pytest.fixture(autouse=True)
def isolated_config():
    importlib.reload(config)
    with home_scope(None):
        yield
    importlib.reload(config)


@pytest.mark.parametrize("override", [False, True], ids=["standalone", "scoped"])
def test_config_uses_active_home(tmp_path, monkeypatch, override):
    launch = write_home(tmp_path / "launch", "launch")
    served = write_home(tmp_path / "served", "served")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    with home_scope(served if override else None):
        assert config.get_active_profile_name() == ("served" if override else "launch")


def test_config_cache_is_home_keyed_and_resettable(tmp_path, monkeypatch):
    a = write_home(tmp_path / "a", "a")
    b = write_home(tmp_path / "b", "b")
    monkeypatch.setenv("HERMES_HOME", str(a))
    for home, expected in [(a, "a"), (b, "b"), (a, "a")]:
        with home_scope(home):
            assert config.list_profiles() == [f"target-{expected}"]
    write_home(a, "edited")
    with home_scope(a):
        assert config.list_profiles() == ["target-a"]
    config._config_cache.clear()
    for home, expected in [(b, "b"), (a, "edited"), (b, "b")]:
        with home_scope(home):
            assert config.list_profiles() == [f"target-{expected}"]


def test_config_cache_canonicalizes_home_aliases(tmp_path, monkeypatch):
    home = write_home(tmp_path / "home", "original")
    alias = tmp_path / "alias"
    alias.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with home_scope(home):
        assert config.list_profiles() == ["target-original"]
    write_home(home, "edited")
    with home_scope(alias):
        assert config.list_profiles() == ["target-original"]
    config._config_cache.clear()
    with home_scope(alias):
        assert config.list_profiles() == ["target-edited"]


@contextmanager
def credential_scope(secrets, *, multiplex):
    previous = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(secrets)
    secret_scope.set_multiplex_active(multiplex)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous)


def credential_reader(kind):
    if kind == "profile":
        return lambda: config.get_profile_config("target-test")["api_key"]
    return lambda: config.get_endpoint_config("test")["api_key"]


@pytest.mark.parametrize("kind", ["profile", "direct"])
def test_credentials_follow_scope_at_call_time(tmp_path, monkeypatch, kind):
    home = write_home(tmp_path / "home", "test")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERALD_TEST_KEY", "launch-test-key")
    read_key = credential_reader(kind)
    with credential_scope(None, multiplex=False):
        assert read_key() == "launch-test-key"
    for name in ("a", "b", "a"):
        env_file = tmp_path / f"{name}.env"
        env_file.write_text(f"HERALD_TEST_KEY={name}-test-key\n")
        with credential_scope(secret_scope.load_env_file(env_file), multiplex=True):
            assert read_key() == f"{name}-test-key"
    with credential_scope(None, multiplex=False):
        assert read_key() == "launch-test-key"


@pytest.mark.parametrize("kind", ["profile", "direct"])
@pytest.mark.parametrize("scope", [{}, None], ids=["missing", "unscoped"])
def test_credentials_fail_closed_in_multiplex(tmp_path, monkeypatch, kind, scope):
    from hermes_herald import tools

    home = write_home(tmp_path / "home", "test")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERALD_TEST_KEY", "must-not-fall-back")
    read_key = credential_reader(kind)
    with credential_scope(scope, multiplex=True):
        if scope is None:
            with pytest.raises(secret_scope.UnscopedSecretError):
                read_key()
        elif kind == "direct":
            with pytest.raises(ValueError, match="unset or empty"):
                read_key()
            result = tools.handle_llm_direct({
                "endpoint": "test", "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            })
            assert "must-not-fall-back" not in result
            assert "unset or empty" in result
        else:
            assert read_key() == ""
            route, error = tools._resolve_profile("target-test", operation="dispatch")
            assert route == {}
            assert "api_key" in error


@pytest.mark.parametrize("kind,filename", [
    ("state", "hermes-herald-runs.json"),
    ("ledger", "hermes-herald.db"),
])
def test_storage_paths_follow_active_home(tmp_path, monkeypatch, kind, filename):
    launch = write_home(tmp_path / "launch", "launch")
    served = write_home(tmp_path / "served", "served")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    resolve = getattr(config, f"get_{kind}_file_path")
    assert resolve() == launch / filename
    with home_scope(served):
        assert resolve() == served / filename
    assert resolve() == launch / filename
    custom = tmp_path / f"custom-{filename}"
    with (served / "config.yaml").open("a") as stream:
        stream.write(f"  {kind}_file: {custom}\n")
    importlib.reload(config)
    with home_scope(served):
        assert resolve() == custom
    assert resolve() == launch / filename


def write_recovery_state(home, origin):
    pending = {
        "run_id": f"approval-{origin}", "profile": "shared-target",
        "delivery_id": f"delivery-{origin}", "command": f"display {origin}",
    }
    (home / "hermes-herald-runs.json").write_text(json.dumps({"runs": [
        {"run_id": f"chat-{origin}", "profile": "shared-target",
         "session_id": f"session-{origin}", "type": "chat", "status": "completed"},
        {"run_id": f"approval-{origin}", "profile": "shared-target",
         "status": "waiting_for_approval", "pending_approval": pending},
    ]}))


@contextmanager
def loaded_plugin(home):
    from hermes_cli.plugins import PluginManager
    from hermes_cli.plugins_manifest import parse_manifest_file

    root = Path(os.environ["HERMES_HERALD_PLUGIN_DIR"])
    manifest = parse_manifest_file(root / "plugin.yaml", root, "user", "")
    assert manifest is not None
    manager = PluginManager(scope_key=str(home.resolve()))
    try:
        manager._load_plugin(manifest)
        loaded = manager._plugins[manifest.name]
        assert loaded.enabled, loaded.error
        yield loaded.module
    finally:
        manager.unload()


@pytest.mark.parametrize("kind", ["sessions", "approvals"])
def test_boot_recovery_uses_manager_home(tmp_path, monkeypatch, kind):
    launch = write_home(tmp_path / "launch", "launch")
    served = write_home(tmp_path / "served", "served")
    for home, origin in [(launch, "launch"), (served, "served")]:
        write_recovery_state(home, origin)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    # No caller home override: register() must inherit the manager's own scope.
    with credential_scope({}, multiplex=True), loaded_plugin(served) as plugin:
        callback = importlib.import_module(f"{plugin.__name__}.callback")
        if kind == "sessions":
            assert callback.get_profile_session_id("shared-target") == "session-served"
        else:
            recovered = callback.get_pending_approval("approval-served")
            assert recovered is not None
            assert recovered["delivery_id"] == "delivery-served"
            assert callback.get_pending_approval("approval-launch") is None

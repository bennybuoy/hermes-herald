"""Profile-home and credential isolation contracts for multiplexed origins."""

from contextlib import contextmanager
import importlib

import pytest

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

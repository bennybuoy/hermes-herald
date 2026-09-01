"""Warn once when a profile resolves to non-loopback plaintext HTTP."""

import logging
import socket
from unittest.mock import patch

import pytest

from hermes_herald import tools


_PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


@pytest.fixture(autouse=True)
def _reset_plaintext_http_cache(monkeypatch):
    tools._http_plaintext_checked.clear()
    for var in _PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    tools._http_plaintext_checked.clear()


def _addrinfo_ipv4(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def _addrinfo_ipv6(ip: str):
    return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0))]


def _stub_profile(monkeypatch, url: str, api_key: str = "secret"):
    monkeypatch.setattr(
        tools.cfg,
        "get_profile_config",
        lambda profile: {"url": url, "api_key": api_key},
    )


def _warning_messages(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "plaintext HTTP" in rec.getMessage()
    ]


def test_loopback_ipv4_http_does_not_warn(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://127.0.0.1:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert resolved["url"] == "http://127.0.0.1:8652"
    assert _warning_messages(caplog) == []


def test_loopback_ipv4_octet_range_does_not_warn(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://127.255.255.254:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert resolved["url"] == "http://127.255.255.254:8652"
    assert _warning_messages(caplog) == []


def test_loopback_ipv6_http_does_not_warn(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://[::1]:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert resolved["url"] == "http://[::1]:8652"
    assert _warning_messages(caplog) == []


def test_ipv4_mapped_loopback_http_does_not_warn(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://[::ffff:127.0.0.1]:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert _warning_messages(caplog) == []


def test_localhost_http_does_not_warn_when_resolved_to_ipv4(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://localhost:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setattr(
        tools.socket, "getaddrinfo", lambda *a, **k: _addrinfo_ipv4("127.0.0.1")
    )

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert resolved["url"] == "http://localhost:8652"
    assert _warning_messages(caplog) == []


def test_localhost_http_does_not_warn_when_resolved_to_ipv6(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://localhost:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setattr(
        tools.socket, "getaddrinfo", lambda *a, **k: _addrinfo_ipv6("::1")
    )

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert _warning_messages(caplog) == []


def test_localhost_is_resolved_not_string_matched(monkeypatch, caplog):
    """A hostname of 'localhost' that resolves off-loopback must still warn."""
    _stub_profile(monkeypatch, "http://localhost:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setattr(
        tools.socket, "getaddrinfo", lambda *a, **k: _addrinfo_ipv4("10.0.0.8")
    )

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert len(_warning_messages(caplog)) == 1


def test_non_loopback_http_warns_but_does_not_reject(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://10.0.0.5:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")

    resolved, err = tools._resolve_profile("tailscale-node")

    assert err is None
    assert resolved["url"] == "http://10.0.0.5:8652"
    assert resolved["api_key"] == "secret"
    messages = _warning_messages(caplog)
    assert len(messages) == 1
    assert "tailscale-node" in messages[0]
    assert "http://10.0.0.5:8652" in messages[0]
    assert "bearer tokens" in messages[0].lower()


def test_hostname_http_warns_on_resolved_public_address(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://herald.example:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setattr(
        tools.socket, "getaddrinfo", lambda *a, **k: _addrinfo_ipv4("8.8.8.8")
    )

    resolved, err = tools._resolve_profile("remote")

    assert err is None
    assert len(_warning_messages(caplog)) == 1


def test_mixed_loopback_and_non_loopback_resolution_warns(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://split.example:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setattr(
        tools.socket,
        "getaddrinfo",
        lambda *a, **k: _addrinfo_ipv4("127.0.0.1") + _addrinfo_ipv4("10.0.0.1"),
    )

    resolved, err = tools._resolve_profile("split")

    assert err is None
    assert len(_warning_messages(caplog)) == 1


def test_https_non_loopback_does_not_warn(monkeypatch, caplog):
    _stub_profile(monkeypatch, "https://herald.example:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    with patch.object(tools.socket, "getaddrinfo") as getaddrinfo:
        resolved, err = tools._resolve_profile("remote")

    assert err is None
    assert resolved["url"] == "https://herald.example:8652"
    assert _warning_messages(caplog) == []
    getaddrinfo.assert_not_called()


def test_warning_fires_once_per_profile_url(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://10.0.0.5:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")

    tools._resolve_profile("node")
    tools._resolve_profile("node")
    tools._resolve_profile("node")

    assert len(_warning_messages(caplog)) == 1


def test_warning_fires_again_when_url_changes(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    urls = iter(("http://10.0.0.5:8652", "http://10.0.0.6:8652"))
    monkeypatch.setattr(
        tools.cfg,
        "get_profile_config",
        lambda profile: {"url": next(urls), "api_key": "secret"},
    )

    tools._resolve_profile("node")
    tools._resolve_profile("node")

    assert len(_warning_messages(caplog)) == 2


def test_warning_fires_once_per_profile_even_with_shared_url(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setattr(
        tools.cfg,
        "get_profile_config",
        lambda profile: {"url": "http://10.0.0.5:8652", "api_key": "secret"},
    )

    tools._resolve_profile("alpha")
    tools._resolve_profile("beta")

    messages = _warning_messages(caplog)
    assert len(messages) == 2
    assert any("alpha" in msg for msg in messages)
    assert any("beta" in msg for msg in messages)


def test_unresolvable_http_host_warns_and_still_resolves(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://missing.example:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")

    def boom(*_a, **_k):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(tools.socket, "getaddrinfo", boom)

    resolved, err = tools._resolve_profile("missing")

    assert err is None
    assert resolved["url"] == "http://missing.example:8652"
    assert len(_warning_messages(caplog)) == 1


def test_host_is_loopback_uses_resolved_addresses(monkeypatch):
    monkeypatch.setattr(
        tools.socket, "getaddrinfo", lambda *a, **k: _addrinfo_ipv6("::1")
    )
    assert tools._host_is_loopback("localhost") is True

    monkeypatch.setattr(
        tools.socket, "getaddrinfo", lambda *a, **k: _addrinfo_ipv4("192.168.1.4")
    )
    assert tools._host_is_loopback("localhost") is False


def test_loopback_http_warns_when_off_loopback_http_proxy_is_set(monkeypatch, caplog):
    """urlopen() would send the bearer token to a non-loopback proxy."""
    _stub_profile(monkeypatch, "http://127.0.0.1:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setenv("http_proxy", "http://10.0.0.1:8080")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    messages = _warning_messages(caplog)
    assert len(messages) == 1
    assert "tutor" in messages[0]
    assert "http://127.0.0.1:8652" in messages[0]
    assert "proxy" in messages[0].lower()
    assert "bearer tokens" in messages[0].lower()


def test_localhost_http_warns_when_http_proxy_set_and_not_in_no_proxy(
    monkeypatch, caplog
):
    _stub_profile(monkeypatch, "http://localhost:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setattr(
        tools.socket, "getaddrinfo", lambda *a, **k: _addrinfo_ipv4("127.0.0.1")
    )
    # Literal proxy IP so the getaddrinfo stub cannot reclassify the proxy
    # as loopback.
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.1:3128")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert resolved["url"] == "http://localhost:8652"
    messages = _warning_messages(caplog)
    assert len(messages) == 1
    assert "proxy" in messages[0].lower()


def test_loopback_http_does_not_warn_when_no_proxy_exempts_host(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://127.0.0.1:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setenv("http_proxy", "http://10.0.0.1:8080")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert _warning_messages(caplog) == []


def test_loopback_http_does_not_warn_when_proxy_itself_is_loopback(
    monkeypatch, caplog
):
    _stub_profile(monkeypatch, "http://127.0.0.1:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:8080")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert _warning_messages(caplog) == []


def test_https_proxy_does_not_affect_http_loopback_warning(monkeypatch, caplog):
    _stub_profile(monkeypatch, "http://127.0.0.1:8652")
    caplog.set_level(logging.WARNING, logger="hermes_herald.tools")
    monkeypatch.setenv("https_proxy", "http://10.0.0.1:8080")

    resolved, err = tools._resolve_profile("tutor")

    assert err is None
    assert _warning_messages(caplog) == []

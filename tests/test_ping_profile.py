"""Regression coverage for the cross-profile health probe."""

import io
import json
from email.message import Message
from urllib.error import HTTPError

from hermes_herald import tools


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        payload = b'{"status":"ok"}'
        return payload if size < 0 else payload[:size]


def test_ping_profile_uses_dedicated_health_endpoint(monkeypatch):
    requested_calls = []

    monkeypatch.setattr(
        tools,
        "_resolve_profile",
        lambda profile: ({"url": "http://profile.test", "api_key": "test-key"}, None),
    )

    def fake_urlopen(request, *args, **kwargs):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        method = request.get_method() if hasattr(request, "get_method") else None
        requested_calls.append((method, url))
        if method == "GET" and url == "http://profile.test/v1/health":
            return _Response()
        if method == "GET" and url == "http://profile.test/v1/runs":
            raise HTTPError(
                url,
                405,
                "Method Not Allowed",
                Message(),
                io.BytesIO(b'{"detail":"method not allowed"}'),
            )
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(tools, "urlopen", fake_urlopen)

    result = json.loads(tools.handle_ping_profile({"profile": "ada"}))

    assert result["status"] == "up"
    assert requested_calls == [("GET", "http://profile.test/v1/health")]

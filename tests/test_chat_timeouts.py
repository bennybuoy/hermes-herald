"""Chat timeout behavior at the HTTP boundary; all requests are synthetic."""
from io import BytesIO
import json
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from hermes_herald import callback, config, ledger, tools


class DelayedStream(BytesIO):
    headers = {"X-Hermes-Session-Id": "chat-session"}

    def __init__(self, timeout):
        super().__init__(b'data: {"choices":[{"delta":{"content":"done"}}]}\n\ndata: [DONE]\n\n')
        self.timeout = timeout

    def read1(self, size: int | None = -1):
        # Model a silent peer that resumes after 65 seconds, without sleeping.
        if self.timeout <= 65:
            raise TimeoutError("timed out")
        return super().read1(size)


def test_chat_accepts_silence_below_the_requested_stall_window(monkeypatch):
    requests = []

    def open_stream(request, timeout):
        requests.append(request)
        return DelayedStream(timeout)

    monkeypatch.setattr(tools, "urlopen", open_stream)
    result = tools._post_streaming_chat(
        "http://target.invalid/v1/chat/completions", "synthetic-key",
        {"messages": [{"role": "user", "content": "marker"}], "stream": True},
        session_id="chat-session", stall_timeout_seconds=90,
    )
    assert result["reply"] == "done"
    assert result["session_id"] == "chat-session"
    assert len(requests) == 1


@pytest.fixture
def chat_origin(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_ledger_file_path", lambda: tmp_path / "ledger.db")
    monkeypatch.setattr(config, "get_state_file_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(tools, "_resolve_profile", lambda *args, **kwargs: (
        {"url": "http://target.invalid", "api_key": "synthetic-key"}, None,
    ))
    monkeypatch.setattr(tools, "_resolve_parent_agent", lambda *args: None)
    callback.store_profile_session_id("synthetic-reviewer", "chat-session")
    yield
    callback.store_profile_session_id("synthetic-reviewer", "")


@pytest.mark.parametrize("failure", ["read", "open", "wrapped"])
def test_chat_timeout_preserves_recovery_session_without_resending(chat_origin, monkeypatch, failure):
    requests = []

    class BrokenStream(BytesIO):
        headers = {}

        def read1(self, size: int | None = -1):
            chunk = super().read1(size)
            if chunk:
                return chunk
            raise TimeoutError("timed out")

    def open_stream(request, timeout):
        requests.append(request)
        if failure == "open":
            raise TimeoutError("timed out")
        if failure == "wrapped":
            raise URLError(TimeoutError("timed out"))
        return BrokenStream(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')

    monkeypatch.setattr(tools, "urlopen", open_stream)
    result = json.loads(tools.handle_dispatch_chat({
        "profile": "synthetic-reviewer", "message": "do work", "stall_timeout_seconds": 90,
    }))
    assert result["status"] == "error"
    assert result["error_code"] == "chat_timeout"
    assert result["profile"] == "synthetic-reviewer"
    assert result["session_id"] == "chat-session"
    assert "may still be running" in result["error"]
    assert "Do not automatically resend" in result["error"]
    assert callback.get_profile_session_id("synthetic-reviewer") == "chat-session"
    assert ledger.list_dispatches() == []
    assert len(requests) == 1


def test_session_creation_timeout_is_reported_without_sending_chat(chat_origin, monkeypatch):
    requests = []

    def open_session(request, timeout):
        requests.append(request)
        raise TimeoutError("timed out")

    monkeypatch.setattr(tools, "urlopen", open_session)
    result = json.loads(tools.handle_dispatch_chat({
        "profile": "synthetic-reviewer", "message": "do work", "new_session": True,
    }))
    assert result["status"] == "error"
    assert "Failed to create session" in result["error"]
    assert len(requests) == 1
    assert requests[0].full_url.endswith("/api/sessions")
    assert callback.get_profile_session_id("synthetic-reviewer") == "chat-session"


@pytest.mark.parametrize("productive", [False, True])
def test_chat_stall_clock_tracks_activity_not_keepalives_or_total_duration(monkeypatch, productive):
    clock = [0.0]
    if productive:
        frames = [
            (25, b'event: hermes.tool.progress\ndata: {"status":"running"}\n\n'),
            (50, b'event: hermes.tool.progress\ndata: {"status":"completed"}\n\n'),
            (75, b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'),
            (80, b'data: [DONE]\n\n'),
        ]
    else:
        frames = [(10, b': keepalive\n\n'), (20, b': keepalive\n\n'), (31, b': keepalive\n\n')]

    class TimelineStream(BytesIO):
        headers = {}

        def read1(self, size: int | None = -1):
            clock[0], chunk = frames.pop(0)
            return chunk

    monkeypatch.setattr(tools, "urlopen", lambda *args, **kwargs: TimelineStream())
    monkeypatch.setattr(tools, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def call():
        return tools._post_streaming_chat(
            "http://target.invalid/v1/chat/completions", "synthetic-key", {},
            session_id="chat-session", stall_timeout_seconds=30,
        )

    if productive:
        assert call()["reply"] == "done"
        assert clock[0] == 80
    else:
        with pytest.raises(RuntimeError, match="stalled for 30 seconds"):
            call()

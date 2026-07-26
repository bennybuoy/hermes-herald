"""Durable ledger, routing, migration, and recovery contracts."""

import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_herald import callback, config, ledger, tools


@pytest.fixture
def ledger_file(tmp_path, monkeypatch):
    path = tmp_path / "herald.db"
    monkeypatch.setattr(config, "get_ledger_file_path", lambda: path)
    return path


def _record(**overrides):
    values = {
        "edge_id": "edge-1",
        "run_id": "run-1",
        "origin_profile": "origin",
        "target_profile": "target",
        "dispatch_type": "run",
        "delivery": "callback",
        "message": "full private message",
        "instructions": "private instructions",
        "trace_id": "trace-1",
        "status": "dispatched",
    }
    values.update(overrides)
    ledger.record_dispatch(**values)


def test_preflight_creates_private_ledger(ledger_file):
    ledger.preflight()
    assert ledger_file.exists()
    if os.name == "posix":
        assert stat.S_IMODE(ledger_file.stat().st_mode) == 0o600


def test_list_redacts_full_text_by_default(ledger_file):
    _record()
    row = ledger.list_dispatches()[0]
    assert "message" not in row
    assert "instructions" not in row
    assert row["message_preview"] == "full private message"


def test_list_can_include_full_text_explicitly(ledger_file):
    _record()
    row = ledger.list_dispatches(include_messages=True)[0]
    assert row["message"] == "full private message"
    assert row["instructions"] == "private instructions"


def test_update_dispatch_records_terminal_result(ledger_file):
    _record()
    ledger.update_dispatch(
        run_id="run-1",
        origin_profile="origin",
        status="completed",
        output_preview="done",
        duration_seconds=1.234,
    )
    row = ledger.list_dispatches()[0]
    assert row["status"] == "completed"
    assert row["completed_at"]
    assert row["output_preview"] == "done"
    assert row["duration_seconds"] == 1.23


def test_list_dispatches_applies_exact_filters(ledger_file):
    _record()
    _record(edge_id="edge-2", run_id="run-2", target_profile="other", delivery="none")
    rows = ledger.list_dispatches(target_profile="other", delivery="none")
    assert [row["run_id"] for row in rows] == ["run-2"]


def test_observed_edges_aggregate_directionally(ledger_file):
    _record()
    _record(edge_id="edge-2", run_id="run-2")
    _record(edge_id="edge-3", run_id="run-3", origin_profile="other")
    assert ledger.observed_edges() == [
        {"origin_profile": "origin", "target_profile": "target", "calls": 2},
        {"origin_profile": "other", "target_profile": "target", "calls": 1},
    ]


def test_known_run_ids_are_origin_scoped(ledger_file):
    _record()
    _record(edge_id="edge-2", run_id="run-2", origin_profile="other")
    assert ledger.known_run_ids("origin") == {"run-1"}
    assert ledger.known_run_ids("other") == {"run-2"}


def test_usage_round_trips_as_structured_data(ledger_file):
    _record(usage={"input_tokens": 4, "output_tokens": 2})
    assert ledger.list_dispatches()[0]["usage"] == {
        "input_tokens": 4,
        "output_tokens": 2,
    }


def test_previews_are_bounded(ledger_file):
    _record(message="m" * 200, output_preview="o" * 700)
    row = ledger.list_dispatches()[0]
    assert len(row["message_preview"]) == 120
    assert len(row["output_preview"]) == 500


def test_route_capabilities_require_explicit_grants_and_fail_closed(monkeypatch):
    monkeypatch.setattr(config, "_load_config", lambda: {"profiles": {"p": {}}})
    assert config.get_route_capabilities("p") == []
    monkeypatch.setattr(
        config,
        "_load_config",
        lambda: {"profiles": {"p": {"capabilities": ["chat", "unknown"]}}},
    )
    assert config.get_route_capabilities("p") == ["chat"]
    monkeypatch.setattr(
        config,
        "_load_config",
        lambda: {"profiles": {"p": {"capabilities": "chat"}}},
    )
    assert config.get_route_capabilities("p") == []


def test_topology_description_excludes_transport_credentials(monkeypatch):
    monkeypatch.setattr(
        config,
        "_load_config",
        lambda: {
            "origin_name": "origin",
            "allow_self": False,
            "profiles": {
                "target": {
                    "url": "http://target.invalid",
                    "api_key": "TOP-SECRET",
                    "capabilities": ["dispatch"],
                }
            },
        },
    )
    topology = config.describe_topology()
    rendered = json.dumps(topology)
    assert topology["origin_profile"] == "origin"
    assert topology["configured_outbound"][0]["capabilities"] == ["dispatch"]
    assert "TOP-SECRET" not in rendered
    assert "api_key" not in rendered


def test_legacy_migration_is_idempotent_and_marks_preview_provenance(
    ledger_file, monkeypatch
):
    monkeypatch.setattr(config, "get_active_profile_name", lambda: "origin")
    state = {
        "runs": [{
            "run_id": "legacy-1",
            "profile": "target",
            "message_preview": "only a preview survived",
            "status": "completed",
        }]
    }
    assert tools._migrate_legacy_run_history(state) == 1
    assert tools._migrate_legacy_run_history(state) == 0
    rows = ledger.list_dispatches(include_messages=True)
    assert len(rows) == 1
    assert rows[0]["message"] == "only a preview survived"
    assert rows[0]["model_resolution"] == "legacy_state_cache"


def test_hop_budget_rejects_before_network_contact(monkeypatch):
    post = patch.object(tools, "_post_json")
    with post as post_json:
        result = json.loads(tools.handle_dispatch_agent({
            "profile": "target",
            "message": "forward",
            "delivery": "none",
            "trace_id": "trace",
            "parent_edge_id": "parent",
            "parent_hop": 2,
            "max_hops": 2,
        }))
    assert result["status"] == "error"
    assert "maximum hop budget" in result["error"]
    post_json.assert_not_called()


def test_run_recovery_state_failure_returns_handle_and_records_ledger(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        tools, "_resolve_profile",
        lambda profile, operation=None: ({"url": "http://target", "api_key": "secret"}, None),
    )
    monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
    monkeypatch.setattr(tools, "_post_json", lambda *a, **k: {"run_id": "run-test"})
    monkeypatch.setattr(tools, "_persist_run", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: recorded.append(kw))
    result = json.loads(tools.handle_dispatch_agent({
        "profile": "target", "message": "hello", "delivery": "none",
    }))
    assert result["run_id"] == "run-test"
    assert result["status"] == "dispatched"
    assert "recovery state" in result["warning"]
    assert recorded[0]["run_id"] == "run-test"


def test_chat_recovery_state_failure_returns_reply_and_records_ledger(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        tools, "_resolve_profile",
        lambda profile, operation=None: ({"url": "http://target", "api_key": "secret"}, None),
    )
    monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
    monkeypatch.setattr(tools, "_post_streaming_chat", lambda *a, **k: {
        "session_id": "session-1",
        "reply": "the reply",
        "model": "hermes-agent",
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    })
    monkeypatch.setattr(tools, "_load_state", lambda: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: recorded.append(kw))
    monkeypatch.setattr(callback, "get_profile_session_id", lambda profile: "session-1")
    monkeypatch.setattr(callback, "capture_session_routing", lambda parent=None: {})
    result = json.loads(tools.handle_dispatch_chat({
        "profile": "target", "message": "hello",
    }))
    assert result["status"] == "completed"
    assert result["session_id"] == "session-1"
    assert result["reply"] == "the reply"
    assert "recovery state" in result["warning"]
    assert recorded[0]["dispatch_type"] == "chat"
    assert recorded[0]["output_preview"] == "the reply"


def test_chat_progress_uses_exact_resolved_parent_without_injected_kwarg(monkeypatch):
    progress_events = []
    parent = SimpleNamespace(
        tool_progress_callback=lambda *args, **kwargs: progress_events.append(
            (args, kwargs)
        )
    )

    def fake_stream(*_args, **kwargs):
        kwargs["progress_callback"](
            "hermes.tool.progress",
            {"status": "running", "tool": "terminal", "label": "Running terminal"},
        )
        return {
            "session_id": "session-1",
            "reply": "done",
            "model": "hermes-agent",
            "usage": {},
        }

    monkeypatch.setattr(
        tools, "_resolve_profile",
        lambda profile, operation=None: ({"url": "http://target", "api_key": "secret"}, None),
    )
    monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
    monkeypatch.setattr(tools, "_resolve_parent_agent", lambda supplied=None: parent)
    monkeypatch.setattr(tools, "_post_streaming_chat", fake_stream)
    monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
    monkeypatch.setattr(tools, "_save_state", lambda state: None)
    monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kwargs: None)
    monkeypatch.setattr(callback, "get_profile_session_id", lambda profile: "session-1")
    monkeypatch.setattr(callback, "capture_session_routing", lambda parent=None: {})

    result = json.loads(tools.handle_dispatch_chat({
        "profile": "target",
        "message": "run a tool",
    }))

    assert result["status"] == "completed"
    assert progress_events == [(("tool.started",), {
        "tool_name": "terminal",
        "preview": "[target] Running terminal",
    })]


def test_new_chat_sessions_use_unique_human_readable_titles(monkeypatch):
    """Repeated new_session calls must not collide with Hermes title uniqueness."""
    titles = []
    created = iter(("session-1", "session-2"))

    def fake_post_json(_url, _key, body, **_kwargs):
        titles.append(body["title"])
        return {"session": {"id": next(created)}}

    monkeypatch.setattr(
        config,
        "get_profile_config",
        lambda profile: {
            "url": "http://target.invalid",
            "api_key": "transport-key",
            "capabilities": ["chat"],
        },
    )
    monkeypatch.setattr(config, "get_route_capabilities", lambda profile: ["chat"])
    monkeypatch.setattr(tools, "_post_json", fake_post_json)
    monkeypatch.setattr(tools, "_post_streaming_chat", lambda *a, **k: {
        "session_id": k["session_id"],
        "reply": "ok",
        "model": "hermes-agent",
        "usage": {},
    })
    monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
    monkeypatch.setattr(tools, "_save_state", lambda state: None)
    monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kwargs: None)
    monkeypatch.setattr(callback, "get_profile_session_id", lambda profile: "")
    monkeypatch.setattr(callback, "store_profile_session_id", lambda *args: None)
    monkeypatch.setattr(callback, "capture_session_routing", lambda parent=None: {})

    for _ in range(2):
        result = json.loads(tools.handle_dispatch_chat({
            "profile": "target",
            "message": "hello",
            "new_session": True,
        }))
        assert result["status"] == "completed"

    assert len(titles) == 2
    assert titles[0].startswith("Dispatch to target · ")
    assert titles[1].startswith("Dispatch to target · ")
    assert titles[0] != titles[1]

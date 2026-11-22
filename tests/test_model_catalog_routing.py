"""Tests for multi-route model catalog discovery, unregistered models, and direct provider routing."""

from __future__ import annotations

import builtins
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_herald import callback, config, host_lifecycle, model_catalog, tools
from hermes_herald.dispatch_identity import DispatchIdentity, IdentityError


def _make_remote_catalog_get_json(
    *,
    v1_models: dict | None = None,
    options_payload: dict | Exception | None = None,
    calls: list[str] | None = None,
):
    default_v1 = {
        "data": [
            {"id": "hermes-agent", "root": "hermes-agent", "parent": None},
            {
                "id": "fast-alias",
                "root": "glm-5.2",
                "parent": "hermes-agent",
                "provider": "ollama-cloud",
            },
        ]
    }
    default_options = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "providers": [
            {
                "slug": "anthropic",
                "name": "Anthropic",
                "authenticated": True,
                "is_current": True,
                "models": ["claude-sonnet-4-6", "claude-opus-4-6"],
            },
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "authenticated": True,
                "models": [
                    "anthropic/claude-sonnet-4.6",
                    "anthropic/claude-sonnet-4.6:nitro",
                    "openai/gpt-5.4",
                ],
            },
            {
                "slug": "custom:vllm",
                "name": "vLLM",
                "authenticated": True,
                "is_user_defined": True,
                "models": [],
            },
            {
                "slug": "google",
                "name": "Google",
                "authenticated": False,
                "models": ["gemini-2.5-pro"],
            },
            {
                "slug": "xai",
                "source": "canonical",
                "models": [],
            },
            {
                "slug": "moa",
                "authenticated": True,
                "models": ["default"],
            },
        ],
    }

    def fake_get(url, api_key, timeout=10.0):
        if calls is not None:
            calls.append(url)
        if url.endswith("/v1/models"):
            return v1_models if v1_models is not None else default_v1
        if "/api/model/options" in url:
            if isinstance(options_payload, Exception):
                raise options_payload
            return options_payload if options_payload is not None else default_options
        if url.endswith("/v1/capabilities"):
            return {
                "features": {
                    "runs_idempotency": {
                        "supported": True,
                        "durable": True,
                        "retention_seconds": 86400,
                    }
                }
            }
        raise AssertionError(f"Unexpected GET URL: {url}")

    return fake_get


# ---------------------------------------------------------------------------
# Remote list_profile_models: combined /v1/models + /api/model/options
# ---------------------------------------------------------------------------


class TestRemoteCatalogDiscovery:
    def test_remote_list_combines_aliases_and_authenticated_provider_catalog(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile: ({"url": "https://target.example/p/reviewer", "api_key": "sec-tok"}, None),
        )
        monkeypatch.setattr(tools, "_get_json", _make_remote_catalog_get_json(calls=calls))

        raw = tools.handle_list_profile_models({"profile": "reviewer"})
        result = json.loads(raw)

        assert calls == [
            "https://target.example/p/reviewer/v1/models",
            "https://target.example/p/reviewer/api/model/options?include_unconfigured=false",
        ]
        assert "sec-tok" not in raw
        assert result["options_available"] is True
        assert result["target_default"] == {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }
        assert result["pass_as_model"] == ["fast-alias"]

        provider_slugs = [p["provider"] for p in result["configured_providers"]]
        # Unauthenticated google, empty canonical xai, and virtual moa are excluded;
        # authenticated user-defined custom:vllm with empty /models is preserved.
        assert provider_slugs == ["anthropic", "custom:vllm", "openrouter"]

        openrouter_row = next(p for p in result["configured_providers"] if p["provider"] == "openrouter")
        assert openrouter_row["catalog_partial"] is True
        assert "org/model" in openrouter_row["note"]

        vllm_row = next(p for p in result["configured_providers"] if p["provider"] == "custom:vllm")
        assert vllm_row["catalog_partial"] is True
        assert vllm_row["models"] == []

        # Direct anthropic + OpenRouter claude-sonnet-4.6 / :nitro group together.
        assert result["multi_route_models"] == [
            {
                "model_key": "claude-sonnet-4-6",
                "routes": [
                    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                    {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
                    {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6:nitro"},
                ],
            }
        ]
        assert result["route_count"] == 5

    def test_remote_query_matches_cross_separator_and_vendor_prefixes(self, monkeypatch):
        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile: ({"url": "https://target.example", "api_key": "k"}, None),
        )
        monkeypatch.setattr(tools, "_get_json", _make_remote_catalog_get_json())

        queried = json.loads(tools.handle_list_profile_models({
            "profile": "reviewer",
            "query": "claude-sonnet-4-6",
        }))
        assert queried["route_match_count"] == 3
        matched_pairs = {(r["provider"], r["model"]) for r in queried["route_matches"]}
        assert matched_pairs == {
            ("anthropic", "claude-sonnet-4-6"),
            ("openrouter", "anthropic/claude-sonnet-4.6"),
            ("openrouter", "anthropic/claude-sonnet-4.6:nitro"),
        }

        # Dotted query also matches hyphenated direct Anthropic ID.
        dotted = json.loads(tools.handle_list_profile_models({
            "profile": "reviewer",
            "query": "sonnet-4.6",
        }))
        assert dotted["route_match_count"] == 3

    def test_remote_list_degrades_cleanly_when_options_endpoint_404(self, monkeypatch):
        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile: ({"url": "https://legacy.example", "api_key": "k"}, None),
        )
        monkeypatch.setattr(
            tools,
            "_get_json",
            _make_remote_catalog_get_json(options_payload=RuntimeError("HTTP 404 Not Found")),
        )

        result = json.loads(tools.handle_list_profile_models({"profile": "legacy"}))
        assert result["options_available"] is False
        assert result["target_default"] is None
        assert result["configured_providers"] == []
        assert result["available_routes"] == []
        assert result["pass_as_model"] == ["fast-alias"]


# ---------------------------------------------------------------------------
# Remote route verification & dispatch_agent / dispatch_chat
# ---------------------------------------------------------------------------


class TestRemoteRouteVerification:
    def _patch_dispatch_agent(self, monkeypatch, *, profile_cfg=None, options_payload=None, v1_models=None):
        captured: dict = {}
        pcfg = {"url": "http://127.0.0.1:8642/p/worker", "api_key": "tok"}
        if profile_cfg:
            pcfg.update(profile_cfg)

        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile, operation=None: (dict(pcfg), None),
        )
        monkeypatch.setattr(
            tools,
            "_get_json",
            _make_remote_catalog_get_json(v1_models=v1_models, options_payload=options_payload),
        )
        monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
        monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)
        monkeypatch.setattr(
            "gateway.session_context.get_session_env",
            lambda key, default="": default,
        )

        def fake_post(url, api_key, body, timeout=30.0, *, idempotency_key=""):
            captured["url"] = url
            captured["body"] = dict(body)
            return {"run_id": "run_123"}

        monkeypatch.setattr(tools, "_post_json", fake_post)
        monkeypatch.setattr(tools, "_persist_run", lambda *a, **kw: captured.update(persist=kw))
        monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: captured.update(ledger=kw))
        monkeypatch.setattr(callback, "start_listener", lambda **kw: captured.update(listener=kw))
        return captured

    def test_dispatch_agent_direct_catalog_provider_and_model(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)
        result = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Audit auth.",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }))

        assert result["status"] == "dispatched", result
        assert result["provider"] == "anthropic"
        assert result["requested_model"] == "claude-sonnet-4-6"
        assert result["resolved_model"] == "claude-sonnet-4-6"
        assert result["model_resolution"] == "target_provider_catalog"
        assert "unregistered_model" not in result
        assert captured["body"]["provider"] == "anthropic"
        assert captured["body"]["model"] == "claude-sonnet-4-6"
        assert captured["listener"]["model_resolution"] == "target_provider_catalog"

    def test_dispatch_agent_direct_unregistered_openrouter_slug(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)
        result = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Benchmark new release.",
            "provider": "openrouter",
            "model": "moonshotai/kimi-k3-preview:nitro",
        }))

        assert result["status"] == "dispatched", result
        assert result["provider"] == "openrouter"
        assert result["requested_model"] == "moonshotai/kimi-k3-preview:nitro"
        assert result["model_resolution"] == "target_provider_unregistered"
        assert result["unregistered_model"] is True
        assert captured["body"]["provider"] == "openrouter"
        assert captured["body"]["model"] == "moonshotai/kimi-k3-preview:nitro"
        assert captured["ledger"]["model_resolution"] == "target_provider_unregistered"

    def test_dispatch_agent_custom_provider_allows_whitespace_model_on_empty_catalog(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)
        result = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Run local.",
            "provider": "vllm",
            "model": "My Custom Model v2",
        }))

        assert result["status"] == "dispatched", result
        assert result["provider"] == "custom:vllm"
        assert result["unregistered_model"] is True
        assert captured["body"]["provider"] == "custom:vllm"
        assert captured["body"]["model"] == "My Custom Model v2"

    def test_dispatch_agent_rejects_whitespace_on_standard_provider(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)
        result = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Run.",
            "provider": "anthropic",
            "model": "claude sonnet",
        }))
        assert result["status"] == "error"
        assert "whitespace" in result["error"]
        assert "body" not in captured

    def test_dispatch_agent_alias_works_even_when_options_404(self, monkeypatch):
        captured = self._patch_dispatch_agent(
            monkeypatch,
            options_payload=RuntimeError("HTTP 404"),
        )
        result = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Quick review.",
            "model": "fast-alias",
        }))
        assert result["status"] == "dispatched", result
        assert result["model_resolution"] == "target_model_routes"
        assert result["resolved_model"] == "glm-5.2"
        assert captured["body"]["model"] == "fast-alias"
        assert captured["body"]["provider"] == "ollama-cloud"

    def test_dispatch_agent_rejects_unauthenticated_and_virtual_providers(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)

        unauth = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "provider": "google",
            "model": "gemini-2.5-pro",
        }))
        assert unauth["status"] == "error"
        assert "not authenticated on worker" in unauth["error"]
        assert "anthropic" in unauth["error"] and "openrouter" in unauth["error"]

        ambig_openai = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "provider": "openai",
            "model": "gpt-5.4",
        }))
        assert ambig_openai["status"] == "error"
        assert "bare 'openai' alias is ambiguous" in ambig_openai["error"]

        virtual = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "provider": "auto",
            "model": "claude-sonnet-4-6",
        }))
        assert virtual["status"] == "error"
        assert "virtual/ambiguous" in virtual["error"]

        virtual_alias = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "provider": "auto",
            "model": "fast-alias",
        }))
        assert virtual_alias["status"] == "error"
        assert "virtual/ambiguous" in virtual_alias["error"]
        assert "body" not in captured

    def test_dispatch_agent_rejects_virtual_primary_even_with_provider(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)
        result = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "provider": "openrouter",
            "model": "hermes-agent",
        }))
        assert result["status"] == "error"
        assert "advertised_primary.model" in result["error"]
        assert "body" not in captured

    def test_dispatch_agent_bare_model_disambiguation_hints(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)

        # Multi-route match: claude-sonnet-4-6 exists on anthropic and openrouter.
        multi = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "model": "claude-sonnet-4-6",
        }))
        assert multi["status"] == "error"
        assert "ambiguous on worker" in multi["error"]
        assert "provider='anthropic'" in multi["error"]
        assert "provider='openrouter'" in multi["error"]

        # Single-route match: claude-opus-4-6 only on anthropic.
        single = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "model": "claude-opus-4-6",
        }))
        assert single["status"] == "error"
        assert "Retry with provider='anthropic', model='claude-opus-4-6'" in single["error"]

        # Zero matches (unregistered model without provider).
        unregistered = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "model": "brandnew/model-x",
        }))
        assert unregistered["status"] == "error"
        assert "pass provider= explicitly alongside model=" in unregistered["error"]
        assert "openrouter" in unregistered["error"]
        assert "body" not in captured

    def test_dispatch_agent_provider_only_validates_authenticated_provider(self, monkeypatch):
        captured = self._patch_dispatch_agent(monkeypatch)

        bad = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "provider": "google",
        }))
        assert bad["status"] == "error"
        assert "not authenticated" in bad["error"]
        assert "body" not in captured

        ok = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Test.",
            "provider": "openrouter",
        }))
        assert ok["status"] == "dispatched", ok
        assert ok["provider"] == "openrouter"
        assert ok["model_resolution"] == "target_provider_override"
        assert captured["body"]["provider"] == "openrouter"
        assert "model" not in captured["body"]

    def test_dispatch_agent_profile_config_provider_inheritance_rules(self, monkeypatch):
        # Inherited when neither model nor provider is passed on the tool call.
        captured = self._patch_dispatch_agent(
            monkeypatch,
            profile_cfg={"provider": "openrouter", "model": "moonshot/kimi-k3"},
        )
        res = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Use profile default.",
        }))
        assert res["status"] == "dispatched", res
        assert captured["body"]["provider"] == "openrouter"
        assert captured["body"]["model"] == "moonshot/kimi-k3"
        assert res["unregistered_model"] is True

        # NOT inherited when caller passes an explicit model_routes alias.
        captured2 = self._patch_dispatch_agent(
            monkeypatch,
            profile_cfg={"provider": "openrouter", "model": "moonshot/kimi-k3"},
        )
        res2 = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Override with alias.",
            "model": "fast-alias",
        }))
        assert res2["status"] == "dispatched", res2
        assert captured2["body"]["model"] == "fast-alias"
        assert captured2["body"]["provider"] == "ollama-cloud"

        # Symmetric guard: explicit provider= differing from pcfg["provider"] MUST NOT
        # inherit pcfg["model"] across providers.
        captured3 = self._patch_dispatch_agent(
            monkeypatch,
            profile_cfg={"provider": "openrouter", "model": "moonshotai/kimi-k3:nitro"},
        )
        res3 = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Switch provider only.",
            "provider": "anthropic",
        }))
        assert res3["status"] == "dispatched", res3
        assert res3["provider"] == "anthropic"
        assert res3["model_resolution"] == "target_provider_override"
        assert captured3["body"]["provider"] == "anthropic"
        assert "model" not in captured3["body"]

        # Same provider (case-insensitive) or unpinned alias still inherits pcfg["model"].
        captured4 = self._patch_dispatch_agent(
            monkeypatch,
            profile_cfg={"provider": "openrouter", "model": "moonshot/kimi-k3"},
        )
        res4 = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Same provider explicit.",
            "provider": "OpenRouter",
        }))
        assert res4["status"] == "dispatched", res4
        assert captured4["body"]["provider"] == "openrouter"
        assert captured4["body"]["model"] == "moonshot/kimi-k3"

        captured5 = self._patch_dispatch_agent(
            monkeypatch,
            profile_cfg={"model": "fast-alias"},
        )
        res5 = json.loads(tools.handle_dispatch_agent({
            "profile": "worker",
            "message": "Unpinned alias with explicit matching provider.",
            "provider": "ollama-cloud",
        }))
        assert res5["status"] == "dispatched", res5
        assert captured5["body"]["model"] == "fast-alias"
        assert captured5["body"]["provider"] == "ollama-cloud"

    def test_dispatch_chat_provider_override_does_not_leak_configured_model(self, monkeypatch):
        captured: dict = {}

        def _patch_chat(profile_cfg):
            captured.clear()
            pcfg = {"url": "http://127.0.0.1:8642/p/tutor", "api_key": "k", **profile_cfg}
            monkeypatch.setattr(
                tools, "_resolve_profile",
                lambda profile, operation=None: (dict(pcfg), None),
            )
            monkeypatch.setattr(tools, "_get_json", _make_remote_catalog_get_json())
            monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)

            def fake_stream(url, api_key, body, **kwargs):
                captured["body"] = dict(body)
                return {"session_id": "sess-1", "reply": "Done", "model": body["model"], "usage": {}}

            monkeypatch.setattr(tools, "_post_streaming_chat", fake_stream)
            monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
            monkeypatch.setattr(tools, "_save_state", lambda state: None)
            monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: captured.update(ledger=kw))
            monkeypatch.setattr(callback, "get_profile_session_id", lambda profile: "sess-1")
            monkeypatch.setattr(callback, "capture_session_routing", lambda parent=None: {})

        # Explicit provider="anthropic" must NOT inherit pcfg["model"] bound to openrouter.
        _patch_chat({"provider": "openrouter", "model": "moonshotai/kimi-k3:nitro"})
        res = json.loads(tools.handle_dispatch_chat({
            "profile": "tutor",
            "message": "Use Anthropic default.",
            "provider": "anthropic",
        }))
        assert res["status"] == "completed", res
        assert res["provider"] == "anthropic"
        assert res["model_resolution"] == "target_provider_override"
        assert captured["body"]["provider"] == "anthropic"
        assert captured["body"]["model"] == "hermes-agent"

        # Matching provider (case-insensitive) still inherits pcfg["model"].
        _patch_chat({"provider": "openrouter", "model": "moonshotai/kimi-k3:nitro"})
        same = json.loads(tools.handle_dispatch_chat({
            "profile": "tutor",
            "message": "Stay on openrouter.",
            "provider": "OpenRouter",
        }))
        assert same["status"] == "completed", same
        assert captured["body"]["provider"] == "openrouter"
        assert captured["body"]["model"] == "moonshotai/kimi-k3:nitro"

        # Unpinned alias (no pcfg["provider"]) still inherits pcfg["model"].
        _patch_chat({"model": "fast-alias"})
        unpinned = json.loads(tools.handle_dispatch_chat({
            "profile": "tutor",
            "message": "Pin alias provider.",
            "provider": "ollama-cloud",
        }))
        assert unpinned["status"] == "completed", unpinned
        assert captured["body"]["model"] == "fast-alias"
        assert captured["body"]["provider"] == "ollama-cloud"

    def test_dispatch_chat_direct_unregistered_openrouter_route(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile, operation=None: ({"url": "http://127.0.0.1:8642/p/tutor", "api_key": "k"}, None),
        )
        monkeypatch.setattr(tools, "_get_json", _make_remote_catalog_get_json())
        monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)

        def fake_stream(url, api_key, body, **kwargs):
            captured["body"] = dict(body)
            return {"session_id": "sess-1", "reply": "Done", "model": body["model"], "usage": {}}

        monkeypatch.setattr(tools, "_post_streaming_chat", fake_stream)
        monkeypatch.setattr(tools, "_load_state", lambda: {"runs": []})
        monkeypatch.setattr(tools, "_save_state", lambda state: None)
        monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: captured.update(ledger=kw))
        monkeypatch.setattr(callback, "get_profile_session_id", lambda profile: "sess-1")
        monkeypatch.setattr(callback, "capture_session_routing", lambda parent=None: {})

        result = json.loads(tools.handle_dispatch_chat({
            "profile": "tutor",
            "message": "Explain closures.",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.6:nitro",
        }))

        assert result["status"] == "completed", result
        assert result["provider"] == "openrouter"
        assert result["requested_model"] == "anthropic/claude-sonnet-4.6:nitro"
        assert result["model_resolution"] == "target_provider_catalog"
        assert captured["body"]["provider"] == "openrouter"
        assert captured["body"]["model"] == "anthropic/claude-sonnet-4.6:nitro"

        unreg = json.loads(tools.handle_dispatch_chat({
            "profile": "tutor",
            "message": "Try brand new model.",
            "provider": "openrouter",
            "model": "acme/unregistered-2026",
        }))
        assert unreg["status"] == "completed", unreg
        assert unreg["unregistered_model"] is True
        assert unreg["model_resolution"] == "target_provider_unregistered"
        assert captured["body"]["provider"] == "openrouter"
        assert captured["body"]["model"] == "acme/unregistered-2026"

    def test_dispatch_identity_fingerprints_configured_profile_provider(self, monkeypatch, tmp_path):
        # Exercise DispatchIdentity.prepare() against an isolated SQLite ledger and
        # confirm changing pcfg["provider"] triggers idempotency_conflict.
        monkeypatch.setattr(config, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(config, "get_ledger_file_path", lambda: tmp_path / "ledger.db")
        monkeypatch.setattr(config, "get_active_profile_name", lambda: "origin")
        routing = {"session_id": "sess-1", "session_key": "key-1"}

        pcfg_a = {"url": "http://127.0.0.1:8642", "api_key": "k", "model": "m", "provider": "anthropic"}
        pcfg_b = {"url": "http://127.0.0.1:8642", "api_key": "k", "model": "m", "provider": "openrouter"}
        pcfg_none = {"url": "http://127.0.0.1:8642", "api_key": "k", "model": "m"}

        args = {"profile": "worker", "message": "hi", "idempotency_key": "provider-fp-01"}
        first = DispatchIdentity(args)
        assert first.prepare(pcfg_a, routing) is None

        with pytest.raises(IdentityError) as exc_info:
            DispatchIdentity(args).prepare(pcfg_b, routing)
        assert exc_info.value.code == "idempotency_conflict"

        with pytest.raises(IdentityError) as exc_info_none:
            DispatchIdentity(args).prepare(pcfg_none, routing)
        assert exc_info_none.value.code == "idempotency_conflict"


# ---------------------------------------------------------------------------
# Runtime provenance surfacing (check_dispatch + callback)
# ---------------------------------------------------------------------------


class TestRuntimeProvenance:
    def test_annotate_run_status_surfaces_served_runtime(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_state", lambda: {
            "runs": [{
                "run_id": "run-rt-1",
                "requested_model": "acme/new",
                "resolved_model": "acme/new",
                "model_resolution": "target_provider_unregistered",
            }]
        })
        status = {
            "run_id": "run-rt-1",
            "status": "completed",
            "model": "acme/new",
            "runtime": {
                "provider": "openrouter",
                "model": "acme/new",
                "route_source": "raw_request",
            },
        }
        tools._annotate_run_status_model(status, "run-rt-1")

        assert status["api_reported_model"] == "acme/new"
        assert status["api_reported_model_is_runtime_evidence"] is False
        assert status["served_provider"] == "openrouter"
        assert status["served_model"] == "acme/new"
        assert status["served_runtime"]["route_source"] == "raw_request"
        assert status["model_resolution"] == "target_provider_unregistered"

    def test_callback_delegation_event_preserves_resolution_and_served_runtime(self):
        evt = callback._build_delegation_event(
            run_id="run-cb-1",
            profile="worker",
            output="done",
            usage={},
            message_preview="hello",
            session_id="s1",
            session_key="k1",
            status="completed",
            model="acme/new",
            requested_model="acme/new",
            resolved_model="acme/new",
            model_resolution="target_provider_unregistered",
            served_runtime={"provider": "openrouter", "model": "acme/new", "route_source": "raw_request"},
        )
        assert evt["model_resolution"] == "target_provider_unregistered"
        assert evt["served_provider"] == "openrouter"
        assert evt["served_model"] == "acme/new"
        assert evt["served_runtime"]["route_source"] == "raw_request"


# ---------------------------------------------------------------------------
# Local discovery, unregistered llm_call pass-through & public-API fallback
# ---------------------------------------------------------------------------


class TestLocalCatalogAndUnregisteredLlmCall:
    def _local_inventory(self):
        return {
            "configured_default": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            },
            "configured_default_in_catalog": True,
            "providers": [
                {
                    "provider": "anthropic",
                    "models": ["claude-sonnet-4-6"],
                    "model_count": 1,
                    "is_user_defined": False,
                    "catalog_partial": False,
                },
                {
                    "provider": "openrouter",
                    "models": ["anthropic/claude-sonnet-4.6"],
                    "model_count": 1,
                    "is_user_defined": False,
                    "catalog_partial": True,
                    "note": "Curated subset only.",
                },
                {
                    "provider": "custom:local-proxy",
                    "models": [],
                    "model_count": 0,
                    "is_user_defined": True,
                    "catalog_partial": True,
                    "aliases": ["local-proxy"],
                },
            ],
        }

    def test_llm_call_allows_unregistered_model_on_configured_provider(self, monkeypatch):
        inv_calls: list[dict] = []
        monkeypatch.setattr(
            tools,
            "_configured_local_model_inventory",
            lambda **kw: (inv_calls.append(kw), self._local_inventory())[1],
        )
        host_llm = MagicMock()
        host_llm.complete.return_value = SimpleNamespace(
            text="ok",
            provider="openrouter",
            model="moonshotai/kimi-k3:nitro",
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=1, total_tokens=2,
                cache_read_tokens=0, cache_write_tokens=0, cost_usd=None,
            ),
        )

        result = json.loads(tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "provider": "openrouter",
                "model": "moonshotai/kimi-k3:nitro",
            },
            _llm=host_llm,
        ))

        assert result["text"] == "ok"
        assert result["requested_provider"] == "openrouter"
        assert result["requested_model"] == "moonshotai/kimi-k3:nitro"
        assert result["unregistered_model"] is True
        assert inv_calls and inv_calls[0]["probe_custom_providers"] is False
        host_llm.complete.assert_called_once()
        assert host_llm.complete.call_args.kwargs["provider"] == "openrouter"
        assert host_llm.complete.call_args.kwargs["model"] == "moonshotai/kimi-k3:nitro"

    def test_llm_call_allows_unregistered_model_on_empty_custom_provider(self, monkeypatch):
        monkeypatch.setattr(
            tools,
            "_configured_local_model_inventory",
            lambda **kw: self._local_inventory(),
        )
        host_llm = MagicMock()
        host_llm.complete.return_value = SimpleNamespace(
            text="custom ok",
            provider="custom:local-proxy",
            model="Local GGUF v1",
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=1, total_tokens=2,
                cache_read_tokens=0, cache_write_tokens=0, cost_usd=None,
            ),
        )

        result = json.loads(tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "provider": "local-proxy",
                "model": "Local GGUF v1",
            },
            _llm=host_llm,
        ))

        assert result["text"] == "custom ok"
        assert result["requested_provider"] == "custom:local-proxy"
        assert result["unregistered_model"] is True

    @pytest.mark.parametrize(
        "provider,model,expected_substr",
        [
            ("auto", "claude-sonnet-4-6", "virtual/ambiguous"),
            ("moa", "default", "virtual/ambiguous"),
            ("openai", "gpt-5.4", "bare 'openai' alias is ambiguous"),
            ("google", "gemini-2.5-pro", "not configured for this profile"),
            ("anthropic", "model with spaces", "whitespace"),
            ("openrouter", "bad\nmodel", "control characters"),
        ],
    )
    def test_llm_call_rejects_invalid_or_unconfigured_routes(
        self, monkeypatch, provider, model, expected_substr
    ):
        monkeypatch.setattr(
            tools,
            "_configured_local_model_inventory",
            lambda **kw: self._local_inventory(),
        )
        host_llm = MagicMock()

        result = json.loads(tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "provider": provider,
                "model": model,
            },
            _llm=host_llm,
        ))

        assert result["status"] == "error"
        assert expected_substr in result["error"]
        host_llm.complete.assert_not_called()

    def test_local_list_surfaces_multi_route_and_credential_free_llm_direct(self, monkeypatch):
        monkeypatch.setattr(
            tools,
            "_configured_local_model_inventory",
            lambda **kw: self._local_inventory(),
        )
        monkeypatch.setattr(config, "_load_config", lambda: {
            "llm_direct": {
                "enabled": True,
                "default_endpoint": "lab",
                "endpoints": {
                    "lab": {
                        "base_url": "https://secret-lab.internal/v1",
                        "api_key": "${LAB_SECRET_KEY}",
                        "default_model": "unregistered-lab-1",
                    },
                },
            },
        })

        raw = tools.handle_list_profile_models({"query": "claude-sonnet-4-6"})
        result = json.loads(raw)

        assert "LAB_SECRET_KEY" not in raw
        assert "secret-lab.internal" not in raw
        assert result["multi_route_models"] == [
            {
                "model_key": "claude-sonnet-4-6",
                "routes": [
                    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                    {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
                ],
            }
        ]
        assert result["llm_direct_endpoints"] == [{
            "endpoint": "lab",
            "is_default": True,
            "default_model": "unregistered-lab-1",
            "allowed_models": None,
            "allows_unregistered_models": True,
        }]
        assert result["match_count"] == 2
        assert len(result["provider_matches"]) == 2

    def test_public_api_fallback_when_core_inventory_unavailable(self, monkeypatch):
        real_import = builtins.__import__

        def block_inventory(name, *args, **kwargs):
            if name == "hermes_cli.inventory":
                raise ImportError("blocked for fallback test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_inventory)

        import hermes_cli.config as hermes_cfg
        import agent.secret_scope as secret_scope
        import providers as hermes_providers

        monkeypatch.setattr(hermes_cfg, "load_config_readonly", lambda: {
            "model": {"provider": "openrouter", "default": "custom/unregistered-active"},
            "providers": {
                "lab": {"base_url": "http://127.0.0.1:8000/v1", "models": ["lab-v1"]},
            },
            "custom_providers": [
                {"name": "Local Box", "base_url": "http://127.0.0.1:11434/v1", "model": "qwen-local"},
            ],
        })
        monkeypatch.setattr(
            secret_scope,
            "get_secret",
            lambda name, default="": "sk-test" if name == "OPENROUTER_API_KEY" else "",
        )
        monkeypatch.setattr(
            hermes_providers,
            "list_providers",
            lambda: [
                SimpleNamespace(
                    name="openrouter",
                    display_name="OpenRouter",
                    aliases=(),
                    env_vars=("OPENROUTER_API_KEY",),
                    static_models=("anthropic/claude-sonnet-4.6",),
                ),
                SimpleNamespace(
                    name="anthropic",
                    display_name="Anthropic",
                    aliases=(),
                    env_vars=("ANTHROPIC_API_KEY",),
                    static_models=("claude-sonnet-4-6",),
                ),
            ],
        )

        inv = model_catalog.load_local_model_inventory()
        assert inv["discovery_source"] == "public_config_fallback"
        assert inv["configured_default"] == {
            "provider": "openrouter",
            "model": "custom/unregistered-active",
        }
        assert inv["configured_default_in_catalog"] is False
        slugs = [p["provider"] for p in inv["providers"]]
        assert slugs == ["custom:local-box", "lab", "openrouter"]
        assert "anthropic" not in slugs


# ---------------------------------------------------------------------------
# Host lifecycle safe error classification (never leaks capability tokens)
# ---------------------------------------------------------------------------


class TestHostLifecycleSafeErrors:
    @pytest.mark.parametrize(
        "raw_exc,expected_substr",
        [
            (
                ValueError("Unknown request fields: provider (private-test-capability)"),
                "per-launch provider support",
            ),
            (
                ValueError("No API key / missing credentials for provider private-test-capability"),
                "no usable credentials",
            ),
            (
                ValueError("Provider disabled in config private-test-capability"),
                "disabled in this profile",
            ),
            (
                ValueError("Command not found on PATH: private-test-capability"),
                "not found on PATH",
            ),
            (
                ValueError("Arbitrary opaque failure private-test-capability"),
                "owning active session",
            ),
        ],
    )
    def test_safe_lifecycle_error_classifies_without_leaking_raw_text(
        self, raw_exc, expected_substr
    ):
        msg = host_lifecycle.HostLifecycleAdapter._safe_lifecycle_error("launch", raw_exc)
        assert expected_substr in msg
        assert "private-test-capability" not in msg


# ---------------------------------------------------------------------------
# Cross-feature integration: multimodal + unregistered catalog + host seams
# ---------------------------------------------------------------------------


_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"
    b"\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestCrossFeatureIntegration:
    def test_llm_call_unregistered_vision_model_with_images_and_base64_redaction(
        self, monkeypatch, tmp_path
    ):
        img_path = tmp_path / "scan.png"
        img_path.write_bytes(_TINY_PNG)

        inv_calls: list[dict] = []
        inventory = {
            "configured_default": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            "configured_default_in_catalog": True,
            "providers": [
                {
                    "provider": "openrouter",
                    "models": ["anthropic/claude-sonnet-4.6"],
                    "model_count": 1,
                    "is_user_defined": False,
                    "catalog_partial": True,
                }
            ],
        }
        monkeypatch.setattr(
            tools,
            "_configured_local_model_inventory",
            lambda **kw: (inv_calls.append(kw), inventory)[1],
        )

        host_llm = MagicMock()
        host_llm.complete.return_value = SimpleNamespace(
            text="transcribed text",
            provider="openrouter",
            model="acme/unregistered-vision-2026:nitro",
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=4, total_tokens=14,
                cache_read_tokens=0, cache_write_tokens=0, cost_usd=None,
            ),
        )

        result = json.loads(tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "OCR this scan."}],
                "images": [str(img_path)],
                "provider": "openrouter",
                "model": "acme/unregistered-vision-2026:nitro",
            },
            _llm=host_llm,
        ))

        assert result["text"] == "transcribed text"
        assert result["unregistered_model"] is True
        assert result["requested_provider"] == "openrouter"
        assert result["requested_model"] == "acme/unregistered-vision-2026:nitro"
        assert inv_calls and inv_calls[0]["probe_custom_providers"] is False

        sent_messages = host_llm.complete.call_args.kwargs["messages"]
        assert len(sent_messages) == 1
        parts = sent_messages[0]["content"]
        assert isinstance(parts, list) and len(parts) == 2
        assert parts[0] == {"type": "text", "text": "OCR this scan."}
        sent_data_url = parts[1]["image_url"]["url"]
        assert sent_data_url.startswith("data:image/png;base64,")
        raw_b64 = sent_data_url.split(",", 1)[1]

        # Provider failure echoing the inline base64 URL must redact payload while
        # preserving surrounding diagnostic text.
        host_llm.complete.side_effect = RuntimeError(
            f"provider rejected {sent_data_url} due to quota"
        )
        err_result = json.loads(tools.handle_llm_call(
            {
                "messages": [{"role": "user", "content": "OCR this scan."}],
                "images": [str(img_path)],
                "provider": "openrouter",
                "model": "acme/unregistered-vision-2026:nitro",
            },
            _llm=host_llm,
        ))
        assert err_result["status"] == "error"
        assert "data:image/png;base64,[redacted]" in err_result["error"]
        assert raw_b64 not in err_result["error"]
        assert "due to quota" in err_result["error"]

    def test_dispatch_agent_listener_receives_both_session_id_and_model_resolution(
        self, monkeypatch
    ):
        captured: dict = {}
        monkeypatch.setattr(
            tools, "_resolve_profile",
            lambda profile, operation=None: ({"url": "http://127.0.0.1:8642/p/worker", "api_key": "tok"}, None),
        )
        monkeypatch.setattr(tools, "_get_json", _make_remote_catalog_get_json())
        monkeypatch.setattr(tools, "_preflight_dispatch_ledger", lambda: None)
        monkeypatch.setattr(tools, "_async_delivery_supported", lambda: True)
        monkeypatch.setattr(
            "gateway.session_context.get_session_env",
            lambda key, default="": default,
        )
        monkeypatch.setattr(tools, "_post_json", lambda *a, **kw: {"run_id": "run-cross-1"})
        monkeypatch.setattr(tools, "_persist_run", lambda *a, **kw: None)
        monkeypatch.setattr(tools, "_record_dispatch_ledger", lambda **kw: None)
        real_start_listener = callback.start_listener
        monkeypatch.setattr(callback, "start_listener", lambda **kw: captured.update(listener=kw))

        res = json.loads(tools.handle_dispatch_agent(
            {
                "profile": "worker",
                "message": "Check cross seam.",
                "provider": "openrouter",
                "model": "acme/unregistered-2026",
            },
            session_id="sess-host-42",
        ))
        assert res["status"] == "dispatched", res
        assert captured["listener"]["session_id"] == "sess-host-42"
        assert captured["listener"]["model_resolution"] == "target_provider_unregistered"

        # Confirm real callback.start_listener accepts both session_id and
        # model_resolution without raising TypeError.
        started_args: list = []

        class FakeThread:
            def __init__(self, target, args, name, daemon):
                started_args.append(args)

            def start(self):
                pass

        monkeypatch.setattr(callback.threading, "Thread", FakeThread)
        monkeypatch.setattr(callback, "capture_session_routing", lambda parent, session_id="": {
            "session_id": session_id or "fallback",
            "session_key": "k-42",
            "parent_session_id": session_id or "fallback",
            "origin_ui_session_id": "",
        })
        real_start_listener(
            run_id="run-cross-2",
            profile="worker",
            url="http://127.0.0.1:8642/p/worker",
            api_key="tok",
            message_preview="Check cross seam.",
            requested_model="acme/unregistered-2026",
            resolved_model="acme/unregistered-2026",
            session_id="sess-host-42",
            model_resolution="target_provider_unregistered",
        )
        callback._cleanup_listener("run-cross-2")
        assert started_args and started_args[0][6] == "sess-host-42"
        assert started_args[0][10] == "target_provider_unregistered"

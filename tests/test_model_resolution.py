"""Regression tests for local and delegated model resolution."""
import json
import os
import sys
import types
import tempfile
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from hermes_herald import tools


class TestModelResolution:
    """Test _resolve_model_creds with mocked switch_model."""

    def test_empty_model_returns_none(self):
        """Empty model name returns all-None creds (inherits parent)."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"

        creds = tools._resolve_model_creds("", parent)
        assert creds["model"] is None
        assert creds["provider"] is None

    def test_same_provider_inherits(self):
        """Same provider as parent inherits credentials."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"
        parent.base_url = "http://localhost:8399/v1"
        parent.api_key = "test-key"
        parent.api_mode = "chat_completions"

        mock_result = SimpleNamespace(
            success=True, new_model="glm-5.2", target_provider="ollama-cloud",
            base_url=parent.base_url, api_key=parent.api_key, api_mode=parent.api_mode,
        )

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            creds = tools._resolve_model_creds("glm-5.2", parent)

        assert creds["model"] == "glm-5.2"
        assert creds["provider"] is None

    def test_different_provider_overrides(self):
        """Different provider gets fresh credentials."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"
        parent.base_url = "http://localhost:8399/v1"
        parent.api_key = "test-key"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.new_model = "claude-sonnet-4"
        mock_result.target_provider = "anthropic"
        mock_result.base_url = "https://api.anthropic.com"
        mock_result.api_key = "sk-ant-xxx"
        mock_result.api_mode = "anthropic"

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            creds = tools._resolve_model_creds("claude-sonnet-4", parent)

        assert creds["model"] == "claude-sonnet-4"
        assert creds["provider"] == "anthropic"
        assert creds["base_url"] == "https://api.anthropic.com"
        assert creds["api_key"] == "sk-ant-xxx"

    def test_resolution_failure_raises(self):
        """Failed model resolution raises ValueError."""
        parent = MagicMock()
        parent.provider = "ollama-cloud"
        parent.model = "glm-5.2"

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Unknown model 'nonexistent'"

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            with pytest.raises(ValueError, match="Unknown model"):
                tools._resolve_model_creds("nonexistent", parent)


@pytest.fixture
def isolated_model_profile(tmp_path, monkeypatch):
    """Only synthetic profile state; no credential or network fallback."""
    import os
    import socket
    from pathlib import Path
    import hermes_cli.model_switch as model_switch
    import tools.delegate_tool_config as delegate_config
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    assert Path(tools.__file__).resolve() == Path(__file__).resolve().parents[1] / "tools.py"
    if source_dir := os.environ.get("HERMES_SOURCE_DIR"):
        source = Path(source_dir).resolve()
        assert Path.cwd().resolve() == source
        assert Path(model_switch.__file__).resolve() == source / "hermes_cli/model_switch.py"
        assert Path(delegate_config.__file__).resolve() == source / "tools/delegate_tool_config.py"

    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(CLI_CONFIG={}))

    def no_network(*args, **kwargs):
        raise AssertionError("model-resolution tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    home_token = set_hermes_home_override(tmp_path)
    secret_token = set_secret_scope({})
    try:
        yield tmp_path
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.parametrize("section", ["providers", "custom_providers"])
def test_resolution_uses_interleaved_active_profiles(
    isolated_model_profile, monkeypatch, section,
):
    """The CLI import cache belongs to launch A, not to every served profile."""
    import hermes_cli.model_switch as model_switch
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    configs = {}
    homes = {}
    for profile in ("a", "b"):
        entry = {
            "name": f"route-{profile}",
            "base_url": f"https://{profile}.invalid/v1",
            "api_key": f"synthetic-{profile}",
            "models": [f"model-{profile}"],
        }
        configs[profile] = {
            section: {f"route-{profile}": entry} if section == "providers" else [entry],
        }
        homes[profile] = isolated_model_profile / profile
        homes[profile].mkdir()
        (homes[profile] / "config.yaml").write_text(
            json.dumps(configs[profile]), encoding="utf-8",
        )
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(CLI_CONFIG=configs["a"]))
    observed = []

    def resolve(**kwargs):
        observed.append((kwargs["user_providers"], kwargs["custom_providers"]))
        return model_switch.ModelSwitchResult(
            success=True, new_model="child-model", target_provider="opencode",
            base_url="https://parent.invalid/v1", api_key="synthetic-parent",
            api_mode="chat_completions",
        )

    monkeypatch.setattr(model_switch, "switch_model", resolve)
    parent = SimpleNamespace(
        provider="opencode", model="parent-model", base_url="https://parent.invalid/v1",
        api_key="synthetic-parent", api_mode="chat_completions",
    )
    for profile in ("a", "b", "a"):
        token = set_hermes_home_override(homes[profile])
        try:
            tools._resolve_model_creds("child-model", parent)
        finally:
            reset_hermes_home_override(token)

    expected = [
        (configs[p].get("providers", {}), configs[p].get("custom_providers"))
        for p in ("a", "b", "a")
    ]
    assert observed == expected


def _child_runtime(parent, creds, delegation_cfg=None):
    """Exercise the same real core runtime assembler used by _build_child_agent."""
    from tools.delegate_tool_config import _resolve_child_runtime

    return _resolve_child_runtime(
        parent, delegation_cfg or {}, parent.api_key, model=creds["model"],
        override_provider=creds["provider"], override_base_url=creds["base_url"],
        override_api_key=creds["api_key"], override_api_mode=creds["api_mode"],
        override_acp_command=None, override_acp_args=None,
    )


def _transport_parent():
    return SimpleNamespace(
        provider="opencode", requested_provider="opencode", model="parent-model",
        base_url="https://parent.invalid/v1", api_key="synthetic-parent",
        api_mode="anthropic_messages", acp_command="parent-only-command",
        acp_args=["parent-only-arg"], capabilities={"native_compaction": True},
        providers_allowed=["parent-only-filter"],
        _fallback_chain=[{"provider": "openrouter", "model": "parent-fallback"}],
    )


@pytest.mark.parametrize("changed", ["base_url", "api_key", "api_mode", "all", "provider"])
def test_changed_route_preserves_accepted_transport_in_core_runtime(
    isolated_model_profile, monkeypatch, changed,
):
    import hermes_cli.model_switch as model_switch

    parent = _transport_parent()
    accepted = {key: getattr(parent, key) for key in ("base_url", "api_key", "api_mode")}
    changes = {
        "base_url": "https://child.invalid/v1",
        "api_key": "synthetic-child", "api_mode": "chat_completions",
    }
    if changed == "all":
        accepted.update(changes)
    elif changed == "provider":
        accepted.update(changes)
    else:
        accepted[changed] = changes[changed]
    target_provider = "deepseek" if changed == "provider" else parent.provider
    result = model_switch.ModelSwitchResult(
        success=True, new_model="child-model", target_provider=target_provider, **accepted,
    )
    monkeypatch.setattr(model_switch, "switch_model", lambda **kwargs: result)

    creds = tools._resolve_model_creds("child-model", parent)
    runtime = _child_runtime(parent, creds)

    assert {key: runtime[key] for key in accepted} == accepted
    assert runtime["provider"] == target_provider
    assert runtime["model"] == "child-model"
    # Pin the WHOLE route: an inherited ACP transport would bypass the new bundle.
    assert runtime["acp_command"] is None
    assert runtime["acp_args"] == []
    assert runtime["capabilities"] is None
    assert runtime["providers_allowed"] is None
    assert runtime["fallback_model"] is None
    declared_fallback = [{"provider": "openrouter", "model": "child-fallback"}]
    configured = _child_runtime(parent, creds, {"fallback_providers": declared_fallback})
    import tools.delegate_tool_config as delegate_config
    # Legacy core only clears parent fallback for a provider pin; it does not
    # support delegation-owned fallback_providers. Both hosts still pin the route.
    scoped_fallback = hasattr(delegate_config, "_resolve_child_fallback_chain")
    assert configured["fallback_model"] == (declared_fallback if scoped_fallback else None)


@pytest.mark.parametrize("model_name", ["", "child-model"])
def test_unchanged_route_inherits_live_parent_transport(
    isolated_model_profile, monkeypatch, model_name,
):
    import hermes_cli.model_switch as model_switch

    parent = _transport_parent()
    parent._client_kwargs = {
        "base_url": "https://live-parent.invalid/v1", "api_key": "synthetic-live-parent",
    }
    result = model_switch.ModelSwitchResult(
        success=True, new_model=model_name, target_provider=parent.provider,
        base_url=parent.base_url, api_key=parent.api_key, api_mode=parent.api_mode,
    )
    monkeypatch.setattr(model_switch, "switch_model", lambda **kwargs: result)
    creds = tools._resolve_model_creds(model_name, parent)
    runtime = _child_runtime(parent, creds)

    assert runtime["model"] == (model_name or parent.model)
    assert runtime["provider"] == parent.provider
    assert runtime["base_url"] == parent._client_kwargs["base_url"]
    import tools.delegate_tool_config as delegate_config
    # Legacy recovers only the live URL, not its key; the paired-endpoint API
    # is a newer host guarantee, not a Herald credential override.
    paired_endpoint = hasattr(delegate_config, "_inherit_parent_endpoint")
    assert runtime["api_key"] == (parent._client_kwargs["api_key"] if paired_endpoint else parent.api_key)
    assert runtime["api_mode"] == parent.api_mode
    assert runtime["acp_command"] == parent.acp_command
    assert runtime["acp_args"] == parent.acp_args
    assert runtime["capabilities"] == parent.capabilities
    assert runtime["providers_allowed"] == parent.providers_allowed
    # New core also pins fallback for an explicit model; legacy only pins providers.
    scoped_fallback = hasattr(delegate_config, "_resolve_child_fallback_chain")
    assert runtime["fallback_model"] == (None if model_name and scoped_fallback else parent._fallback_chain)


def test_real_switch_same_provider_dual_wire_route(
    isolated_model_profile, monkeypatch,
):
    """Real /model normalization and child-runtime assembly, offline resolver/catalog seams."""
    import hermes_cli.model_switch as model_switch
    import hermes_cli.runtime_provider as runtime_provider

    parent = _transport_parent()
    parent.base_url = "https://opencode.ai/zen"
    # Avoid any credential, live catalog or metadata fetch; retain real mode/URL shaping.
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **kwargs: {
        "provider": "opencode", "base_url": parent.base_url,
        "api_key": parent.api_key, "api_mode": parent.api_mode,
    })
    monkeypatch.setattr(model_switch, "resolve_alias", lambda *args: None)
    monkeypatch.setattr(model_switch, "list_provider_models", lambda *args: [])
    import hermes_cli.models as models
    monkeypatch.setattr(models, "detect_provider_for_model", lambda *args: None)
    import hermes_cli.models_validate as models_validate
    monkeypatch.setattr(models_validate, "validate_requested_model", lambda *args, **kwargs: {
        "accepted": True, "persist": False, "recognized": True,
    })
    monkeypatch.setattr(model_switch, "get_model_capabilities", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_switch, "get_model_info", lambda *args, **kwargs: None)

    creds = tools._resolve_model_creds("deepseek-v4-flash", parent)
    runtime = _child_runtime(parent, creds)

    assert runtime["provider"] == "opencode"
    assert runtime["model"] == "deepseek-v4-flash"
    assert runtime["base_url"] == "https://opencode.ai/zen/v1"
    assert runtime["api_key"] == parent.api_key
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["acp_command"] is None


@pytest.mark.parametrize("route", ["inherit", "same-provider", "cross-provider"])
def test_real_core_child_construction_keeps_route_bundle(
    isolated_model_profile, monkeypatch, route,
):
    """Build a real AIAgent without running inference or starting a detached task."""
    import hermes_cli.model_switch as model_switch
    from tools.delegate_tool import _build_child_agent

    parent = _transport_parent()
    parent.acp_command = None
    parent.acp_args = []
    parent.enabled_toolsets = []
    parent.request_overrides = {"extra_body": {"synthetic_parent_option": True}}
    accepted = {key: getattr(parent, key) for key in ("base_url", "api_key", "api_mode")}
    target_provider = parent.provider
    if route != "inherit":
        accepted.update(
            base_url="https://child.invalid/v1", api_key="synthetic-child",
            api_mode="chat_completions",
        )
        if route == "cross-provider":
            target_provider = "deepseek"
    # Legacy's native provider initialization rejects invented catalog models.
    child_model = "deepseek-v4-flash" if route == "cross-provider" else "child-model"
    result = model_switch.ModelSwitchResult(
        success=True, new_model=child_model, target_provider=target_provider, **accepted,
    )
    monkeypatch.setattr(model_switch, "switch_model", lambda **kwargs: result)
    creds = tools._resolve_model_creds(child_model, parent)
    child = _build_child_agent(
        task_index=0, goal="offline model-routing test", context=None, toolsets=None,
        model=creds["model"], max_iterations=2, task_count=1, parent_agent=parent,
        override_provider=creds["provider"], override_base_url=creds["base_url"],
        override_api_key=creds["api_key"], override_api_mode=creds["api_mode"],
    )
    try:
        assert child.provider == target_provider
        assert child.model == child_model
        assert child.base_url == accepted["base_url"]
        assert child.api_key == accepted["api_key"]
        assert child.api_mode == accepted["api_mode"]
        assert child.acp_command is None
        import tools.delegate_tool_config as delegate_config
        scoped_fallback = hasattr(delegate_config, "_resolve_child_fallback_chain")
        assert child._fallback_chain == (parent._fallback_chain if route == "inherit" and not scoped_fallback else [])
        assert child.request_overrides == (
            parent.request_overrides if route == "inherit" else {}
        )
    finally:
        child.close()


class TestLocalModelRouteDiscovery:
    """The caller can discover only routes declared by its active profile."""

    def test_schema_allows_local_discovery_without_a_profile_name(self):
        assert "profile" not in tools.LIST_PROFILE_MODELS_SCHEMA["parameters"].get(
            "required", []
        )

    def test_local_inventory_uses_core_explicit_only_contract(self, monkeypatch):
        import agent.auxiliary_client as auxiliary_client
        import hermes_cli.inventory as inventory

        context = SimpleNamespace(
            current_provider="ollama-cloud",
            current_model="glm-5.2",
            current_base_url="http://ollama-cloud.example/v1",
            user_providers={
                "backup": {
                    "base_url": "http://backup.example/v1",
                    "models": ["backup-model"],
                }
            },
            custom_providers=[{
                "name": "llamaherd",
                "base_url": "http://llamaherd.example/v1",
                "models": {"glm-5.2": {}},
            }],
            excluded_providers=[],
        )
        monkeypatch.setattr(inventory, "load_picker_context", lambda: context)
        monkeypatch.setattr(
            auxiliary_client, "_read_main_provider", lambda: "ollama-cloud"
        )
        monkeypatch.setattr(auxiliary_client, "_read_main_model", lambda: "glm-5.2")
        monkeypatch.setattr(
            auxiliary_client,
            "_read_main_base_url",
            lambda: "http://ollama-cloud.example/v1",
        )
        monkeypatch.setattr(auxiliary_client, "_read_main_api_key", lambda: "test-key")
        import hermes_cli.model_switch as model_switch

        monkeypatch.setattr(
            model_switch,
            "switch_model",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("discovery must report configured slugs without resolving them")
            ),
        )
        captured = {}

        def explicit_payload(*args, **kwargs):
            captured.update(kwargs)
            return {
                "providers": [
                    {"slug": "ollama-cloud", "models": ["glm-5.2"]},
                    {"slug": "backup", "models": ["backup-model"]},
                    {"slug": "custom:llamaherd", "models": ["glm-5.2"]},
                    {"slug": "custom:openai-2", "models": ["model-b"]},
                    {"slug": "moa", "models": ["default"]},
                ]
            }

        monkeypatch.setattr(inventory, "build_models_payload", explicit_payload)

        result = json.loads(tools.handle_list_profile_models({}))

        assert result["scope"] == "local"
        assert result["configured_default"] == {
            "provider": "ollama-cloud",
            "model": "glm-5.2",
        }
        assert result["available_routes"] == [
            {"provider": "backup", "model": "backup-model"},
            {"provider": "custom:llamaherd", "model": "glm-5.2"},
            {"provider": "custom:openai-2", "model": "model-b"},
            {"provider": "ollama-cloud", "model": "glm-5.2"},
        ]
        assert result["route_count"] == 4
        assert "do not pass a bare model name" in result["contract"]
        assert captured["explicit_only"] is True
        assert captured["include_unconfigured"] is False

    def test_named_profile_preserves_remote_authenticated_alias_contract(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            tools,
            "_resolve_profile",
            lambda profile: ({
                "url": "https://reviewer.example",
                "api_key": "remote-secret",
            }, None),
        )
        request = {}

        def remote_models(url, api_key, timeout):
            request.update(url=url, api_key=api_key, timeout=timeout)
            return {
                "data": [
                    {"id": "glm-5.2", "root": "glm-5.2"},
                    {
                        "id": "review-fast",
                        "root": "glm-5.2",
                        "parent": "glm-5.2",
                        "provider": "nous",
                    },
                ]
            }

        monkeypatch.setattr(tools, "_get_json", remote_models)

        raw = tools.handle_list_profile_models({"profile": "reviewer"})
        result = json.loads(raw)

        assert request == {
            "url": "https://reviewer.example/v1/models",
            "api_key": "remote-secret",
            "timeout": 10.0,
        }
        assert result["profile"] == "reviewer"
        assert result["advertised_primary"] == {
            "model": "glm-5.2",
            "dispatchable_as_override": False,
            "is_runtime_evidence": False,
        }
        assert result["pass_as_model"] == ["review-fast"]
        assert result["dispatchable_models"] == [{
            "alias": "review-fast",
            "pass_as": "review-fast",
            "resolved_model": "glm-5.2",
            "provider": "nous",
            "provider_pinned": True,
        }]
        assert result["dispatchable_model_count"] == 1
        assert "pass provider=" in result["contract"]
        assert "remote-secret" not in raw

        queried = json.loads(tools.handle_list_profile_models({
            "profile": "reviewer", "query": "review",
        }))
        assert queried["query"] == "review"
        assert queried["match_pass_as"] == ["review-fast"]
        assert queried["pass_as_model"] == ["review-fast"]

        missed = json.loads(tools.handle_list_profile_models({
            "profile": "reviewer", "query": "astra",
        }))
        assert missed["match_count"] == 0
        assert missed["match_pass_as"] == []
        assert missed["pass_as_model"] == ["review-fast"]
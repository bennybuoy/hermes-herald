"""Real lifecycle -> AIAgent -> credential leasing -> synthetic HTTP boundary.

Configuration, auth persistence and transport are disposable. No child methods
or credentials are patched by Herald. Core owns credential refresh/rotation
inside the selected provider; provider identity must never become the parent's
merely because two named routes share an endpoint.
"""
import json
import queue
from types import SimpleNamespace

import httpx
import pytest

lifecycle = pytest.importorskip("agent.subagent_lifecycle")
if "provider" not in lifecycle.SubagentLaunchRequest.__dataclass_fields__:
    pytest.skip("Host lacks per-launch provider lifecycle support", allow_module_level=True)

from hermes_herald.native_lifecycle import NativeLifecycleAdapter


@pytest.mark.parametrize("route", ["inherit", "same-provider", "cross-provider", "shared-endpoint"])
def test_core_owns_pinned_provider_credentials_at_inference(route, monkeypatch, tmp_path):
    from agent import credential_pool
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from gateway.session_context import set_session_vars, clear_session_vars
    from run_agent import AIAgent
    from tools import process_registry as registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parent_url = "https://parent.invalid/v1"
    child_url = parent_url if route in {"inherit", "same-provider", "shared-endpoint"} else "https://child.invalid/v1"
    child_name = "origin" if route in {"inherit", "same-provider"} else "worker"
    config = {
        "model": {"context_length": 128000},
        "delegation": {"max_spawn_depth": 1, "worktree_isolation": False},
        "compression": {"enabled": False},
        "custom_providers": [
            {"name": "origin", "base_url": parent_url, "api_key": "synthetic-origin", "models": ["parent-model", "child-model"]},
            {"name": "worker", "base_url": child_url, "api_key": "synthetic-worker", "models": ["child-model"]},
        ],
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    home_token = set_hermes_home_override(tmp_path)
    secret_token = set_secret_scope({})
    pools = {}
    for name, url in [("origin", parent_url), ("worker", child_url)]:
        slug = f"custom:{name}"
        entry = credential_pool.PooledCredential(provider=slug, id=name, label=name,
            auth_type="api_key", priority=0, source="manual", access_token=f"synthetic-{name}-pooled", base_url=url)
        pools[slug] = credential_pool.CredentialPool(slug, [entry])
    monkeypatch.setattr(credential_pool.CredentialPool, "_persist", lambda self: None)
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider, **kw: pools.get(provider))
    notifications = queue.Queue()
    monkeypatch.setattr(registry.process_registry, "completion_queue", notifications)
    observed = []
    parent = None

    def respond(request):
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "child-model"}]})
        assert request.method == "POST" and request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        child, = parent._active_children
        observed.append((request.headers["authorization"], str(request.url), body["model"],
                         child.requested_provider, child._credential_pool, list(child._fallback_chain)))
        message = {"content": "controlled completion"}
        if body.get("stream"):
            data = {"id": "synthetic", "object": "chat.completion.chunk", "created": 1, "model": "child-model",
                    "choices": [{"index": 0, "delta": message, "finish_reason": "stop"}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                content=("data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n").encode())
        return httpx.Response(200, json={"id": "synthetic", "object": "chat.completion", "created": 1,
            "model": "child-model", "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}})

    original_init = httpx.Client.__init__
    def synthetic_http(self, *args, **kwargs):
        kwargs.update(transport=httpx.MockTransport(respond), mounts=None, proxy=None, trust_env=False)
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "__init__", synthetic_http)
    manager = PluginManager(scope_key=str(tmp_path))
    ctx = PluginContext(PluginManifest(name="hermes-herald"), manager)
    tokens = None
    try:
        parent = AIAgent(provider="custom:origin", base_url=parent_url, api_key="synthetic-origin",
                        model="parent-model", api_mode="chat_completions", enabled_toolsets=[],
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        credential_pool=pools["custom:origin"])
        initial_entries = [e.to_dict() for e in pools["custom:origin"].entries()]
        original_key, original_url = parent.api_key, parent.base_url
        tokens = set_session_vars(platform="cli", session_id=parent.session_id, async_delivery=True)
        args = {"goal": "Return a controlled completion", "model": "child-model"}
        if route != "inherit":
            args["provider"] = f"custom:{child_name}"
        with lifecycle.bind_subagent_parent(parent):
            response = json.loads(NativeLifecycleAdapter(ctx)(args, session_id=parent.session_id))
        assert response["status"] == "dispatched", response
        completion = notifications.get(timeout=30)
        assert completion["status"] == "completed", completion
        assert completion["summary"] == "controlled completion"
        assert observed, "No child request reached the HTTP boundary"
        for authorization, url, model, requested_provider, pool, fallback in observed:
            assert authorization == f"Bearer synthetic-{child_name}-pooled"
            assert url == child_url + "/chat/completions"
            assert model == "child-model"
            assert requested_provider == f"custom:{child_name}"
            assert pool is pools[f"custom:{child_name}"]
            assert fallback == []
        assert parent._credential_pool is pools["custom:origin"]
        assert (parent.api_key, parent.base_url) == (original_key, original_url)
        assert [e.to_dict() for e in pools["custom:origin"].entries()] == initial_entries
        assert all(not p._active_leases for p in pools.values())
        assert notifications.empty()
    finally:
        if tokens is not None:
            clear_session_vars(tokens)
        if parent is not None:
            parent.close()
        manager.unload()
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

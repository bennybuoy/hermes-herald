"""Accepted Herald credentials survive real core construction and startup leasing.

Only catalog/auth lookup and the HTTP transport are synthetic. The handler,
AIAgent constructor, pool resolver, startup lease, credential swap and child
conversation all run as installed, including on the legacy monolithic layout.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import httpx
import pytest

from hermes_herald import tools


@pytest.mark.parametrize(
    "change",
    ["unchanged", "key", "endpoint", "provider", "provider-env"],
)
def test_accepted_bundle_at_child_inference_boundary(change, monkeypatch, tmp_path):
    from agent import credential_pool
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import model_switch, models_validate, runtime_provider
    from run_agent import AIAgent
    from tools import delegate_tool
    from tools.process_registry import process_registry

    source = Path(os.environ["HERMES_SOURCE_DIR"]).resolve()
    plugin = Path(os.environ["HERMES_HERALD_PLUGIN_DIR"]).resolve()
    assert Path(tools.__file__).resolve() == plugin / "tools.py"
    # Definitions may live in delegate_tool or its newer split siblings.
    for symbol in (AIAgent, delegate_tool._build_child_agent,
                   delegate_tool._resolve_child_credential_pool,
                   delegate_tool._run_single_child, AIAgent._swap_credential):
        assert Path(inspect.getfile(symbol)).resolve().is_relative_to(source)

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  context_length: 128000\n"
        "delegation:\n  max_iterations: 2\n  max_spawn_depth: 1\n"
        "  worktree_isolation: false\n"
        "compression:\n  enabled: false\n",
        encoding="utf-8",
    )

    parent_provider = "openrouter"
    parent_url = "https://parent.invalid/v1"
    parent_key = "synthetic-parent-key"
    different_provider = change in {"provider", "provider-env"}
    child_provider = "deepseek" if different_provider else parent_provider
    child_url = "https://child.invalid/v1" if change == "endpoint" or different_provider else parent_url
    if change == "provider-env":
        child_url = "https://api.deepseek.com/v1"
        monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-env-key")
    child_key = parent_key if change == "unchanged" else "synthetic-accepted-key"
    # Legacy native-provider setup normalizes unknown models to its default.
    child_model = "deepseek-v4-flash" if different_provider else "herald-pool-model"

    def entry(provider, url, ident, priority=0):
        return credential_pool.PooledCredential(
            provider=provider, id=ident, label=ident, auth_type="api_key",
            priority=priority, source="manual", access_token="synthetic-pooled-key",
            base_url=url,
        )

    parent_entries = [entry(parent_provider, parent_url, "parent-account")]
    if change == "endpoint":
        parent_entries.append(entry(parent_provider, child_url, "endpoint-account", 1))
    parent_pool = credential_pool.CredentialPool(parent_provider, parent_entries)
    target_pool = (
        credential_pool.CredentialPool(child_provider, [entry(child_provider, child_url, "other-account")])
        if different_provider else parent_pool
    )
    # Auth persistence/lookup only; all selection and lease methods remain real.
    monkeypatch.setattr(credential_pool.CredentialPool, "_persist", lambda self: None)
    monkeypatch.setattr(
        credential_pool, "load_pool",
        lambda provider, **kwargs: target_pool if provider == child_provider else None,
    )
    monkeypatch.setattr(
        runtime_provider, "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": child_provider, "base_url": child_url, "api_key": child_key,
            "api_mode": "chat_completions",
        },
    )
    # The real switch_model pipeline consumes a deterministic catalog entry.
    monkeypatch.setattr(model_switch, "resolve_alias", lambda *a, **kw: (child_provider, child_model, "pool-test"))
    monkeypatch.setattr(model_switch, "get_model_capabilities", lambda *a, **kw: None)
    monkeypatch.setattr(model_switch, "get_model_info", lambda *a, **kw: None)
    monkeypatch.setattr(
        models_validate, "validate_requested_model",
        lambda *a, **kw: {"accepted": True, "recognized": True, "persist": False},
    )

    observations = []
    parent = None

    def respond(request):
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": child_model}]})
        assert request.method == "POST" and request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == child_model
        # Observe the live child only after actual core startup leasing and swap.
        child, = parent._active_children
        observations.append({
            "authorization": request.headers["authorization"],
            "url": str(request.url), "provider": child.provider,
            "api_mode": child.api_mode, "key": child.api_key,
            "pool": child._credential_pool,
            "leases": dict(target_pool._active_leases),
            "child": child,
        })
        if body.get("stream"):
            chunk = {
                "id": "synthetic", "object": "chat.completion.chunk", "created": 1,
                "model": child_model,
                "choices": [{"index": 0, "delta": {"content": "synthetic completion"}, "finish_reason": "stop"}],
            }
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode(),
            )
        return httpx.Response(200, json={
            "id": "synthetic", "object": "chat.completion", "created": 1,
            "model": child_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "synthetic completion"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        })

    original_http_init = httpx.Client.__init__

    def synthetic_http_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        # Newer core mounts its shared keepalive transports by scheme; those
        # would otherwise take precedence over the synthetic default transport.
        kwargs["mounts"] = None
        kwargs["proxy"] = None
        kwargs["trust_env"] = False
        return original_http_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", synthetic_http_init)
    parent = AIAgent(
        provider=parent_provider, base_url=parent_url, api_key=parent_key,
        api_mode="chat_completions", model="herald-parent-model",
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
        credential_pool=parent_pool,
    )
    # The ordinary core pool resolver really offers a pool on every test route.
    assert delegate_tool._resolve_child_credential_pool(child_provider, parent, child_url) is target_pool
    initial_entries = [e.to_dict() for e in parent_pool.entries()]
    tokens = set_session_vars(platform="cli", session_id=parent.session_id, async_delivery=True)
    try:
        dispatched = json.loads(tools.handle_delegate_subagent(
            {"goal": "Return a synthetic completion", "model": "pool-test", "toolsets": []},
            parent_agent=parent,
        ))
        assert dispatched["status"] == "dispatched", dispatched
        completion = process_registry.completion_queue.get(timeout=30)
        assert completion["delegation_id"] == dispatched["task_id"]
        assert completion["status"] == "completed", completion
        assert completion["summary"] == "synthetic completion"
        assert observations, "The actual child never reached the synthetic HTTP boundary"
        expected_key = "synthetic-pooled-key" if change == "unchanged" else child_key
        for observed in observations:
            assert observed["authorization"] == f"Bearer {expected_key}"
            assert observed["key"] == expected_key
            assert observed["url"] == f"{child_url}/chat/completions"
            assert observed["provider"] == child_provider
            assert observed["api_mode"] == "chat_completions"
            if change == "unchanged":
                assert observed["pool"] is parent_pool
                assert observed["leases"] == {"parent-account": 1}
                assert callable(observed["child"]._recover_with_credential_pool)
            else:
                assert observed["pool"] is None
                assert observed["leases"] == {}
        assert parent._credential_pool is parent_pool
        assert parent.api_key == parent_key
        assert parent.base_url == parent_url
        assert [e.to_dict() for e in parent_pool.entries()] == initial_entries
        assert parent_pool._active_leases == {}
        assert target_pool._active_leases == {}
    finally:
        clear_session_vars(tokens)
        parent.close()

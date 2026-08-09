"""Hermes Herald — dispatch tasks to other Hermes profiles.

Three dispatch modes:
  - dispatch_agent: async, SSE callback, verified target model_routes aliases
  - dispatch_chat: sync, streamed, session-persistent, verified model aliases
  - delegate_subagent: in-process subagent with per-call model and timeout policy

Inference:
  - llm_call: bare LLM call through Hermes provider routing

Plus management tools:
  - check_dispatch: GET /v1/runs/{run_id}, polls a single run
  - collect_dispatches: batch-poll multiple runs at once
  - dispatch_status: list persisted runs from state file
  - cancel_dispatch: cooperative cancellation via /v1/runs/{id}/stop
  - ping_profile: health check for target profile reachability
  - approve_dispatch: deny a pending remote approval request
  - list_profile_models: discover executable local or remote model routes

Config (config.yaml):
  hermes_herald:
    profiles:
      reviewer:
        url: http://localhost:8651
        api_key: ${REVIEWER_API_KEY}
        model: reviewer-fast  # optional exact model_routes alias for both dispatch tools
    state_file: /custom/private/path/hermes-herald-runs.json  # optional
"""

from __future__ import annotations

import logging
from pathlib import Path

from .tools import (
    ALL_SCHEMAS,
    handle_dispatch_agent,
    handle_check_dispatch,
    handle_collect_dispatches,
    handle_dispatch_status,
    handle_dispatch_chat,
    handle_cancel_dispatch,
    handle_delegate_subagent,
    handle_llm_call,
    handle_ping_profile,
    handle_approve_dispatch,
    handle_list_profile_models,
)

logger = logging.getLogger(__name__)

_TOOLS = (
    ("dispatch_agent", DISPATCH_AGENT_SCHEMA := ALL_SCHEMAS[0], handle_dispatch_agent, "🚀"),
    ("check_dispatch", ALL_SCHEMAS[1], handle_check_dispatch, "🔍"),
    ("collect_dispatches", ALL_SCHEMAS[2], handle_collect_dispatches, "📥"),
    ("dispatch_status", ALL_SCHEMAS[3], handle_dispatch_status, "📋"),
    ("dispatch_chat", ALL_SCHEMAS[4], handle_dispatch_chat, "💬"),
    ("cancel_dispatch", ALL_SCHEMAS[5], handle_cancel_dispatch, "🛑"),
    ("delegate_subagent", ALL_SCHEMAS[6], handle_delegate_subagent, "🔀"),
    ("llm_call", ALL_SCHEMAS[7], handle_llm_call, "🧠"),
    ("ping_profile", ALL_SCHEMAS[8], handle_ping_profile, "📶"),
    ("approve_dispatch", ALL_SCHEMAS[9], handle_approve_dispatch, "✅"),
    ("list_profile_models", ALL_SCHEMAS[10], handle_list_profile_models, "🧭"),
)


def register(ctx) -> None:
    """Register all Hermes Herald tools. Called once by the plugin loader."""
    def _host_llm_handler(args: dict, **kwargs) -> str:
        return handle_llm_call(args, _llm=ctx.llm, **kwargs)

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="hermes_herald",
            schema=schema,
            handler=_host_llm_handler if name == "llm_call" else handler,
            emoji=emoji,
        )
    logger.info(
        "hermes-herald: registered %d tools (dispatch_agent, check_dispatch, "
        "collect_dispatches, dispatch_status, dispatch_chat, cancel_dispatch, "
        "delegate_subagent, llm_call, ping_profile, approve_dispatch, "
        "list_profile_models)",
        len(_TOOLS),
    )
    skill_md = Path(__file__).parent / "skills" / "agent-dispatch" / "SKILL.md"
    if skill_md.exists():
        ctx.register_skill("agent-dispatch", skill_md)
    # Recover session_ids from state file so dispatch_chat can resume
    # conversations after a gateway restart.
    try:
        from .callback import recover_session_ids
        n = recover_session_ids()
        if n:
            logger.info("hermes-herald: recovered %d session_id(s) from state file", n)
    except Exception:
        pass  # best-effort
    # Recover pending-approval metadata so check_dispatch / approve_dispatch
    # work across a plugin/session restart.
    try:
        from .callback import recover_pending_approvals
        n = recover_pending_approvals()
        if n:
            logger.info(
                "hermes-herald: recovered %d pending approval(s) from state file", n
            )
    except Exception:
        pass  # best-effort

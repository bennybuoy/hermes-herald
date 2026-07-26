"""Tool schemas and handlers for the Hermes Herald plugin.

Eleven tools for cross-profile dispatch, local delegation, and bare inference:
  - dispatch_agent: POST /v1/runs (async, SSE callback or detached graph edge)
  - dispatch_chat: streaming POST /v1/chat/completions, sync session-persistent
  - delegate_subagent: in-process subagent with per-call model and timeout policy
  - llm_call: bare inference through Hermes provider routing
  - check_dispatch: GET /v1/runs/{run_id}, returns status
  - collect_dispatches: batch GET /v1/runs/{run_id} for multiple runs
  - dispatch_status: query durable SQLite calls and directed topology
  - cancel_dispatch: POST /v1/runs/{run_id}/stop, cooperative cancellation
  - ping_profile: GET /v1/health, health check for target profile
  - approve_dispatch: POST /v1/runs/{run_id}/approval, resolve approvals
  - list_profile_models: GET /v1/models, discover safe aliases for both dispatch modes

All HTTP is done with urllib.request (stdlib). Handlers are synchronous
except delegate_subagent which runs in a background thread.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import HTTPError, URLError

from . import config as cfg

logger = logging.getLogger(__name__)


class _NoRedirectHandler(HTTPRedirectHandler):
    """Reject redirects so bearer credentials never cross origins."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def urlopen(request, *args, **kwargs):
    """Open one HTTP request without following redirects.

    Kept as a module-level seam so tests can replace ``urlopen`` without
    touching urllib's global opener.
    """
    return build_opener(_NoRedirectHandler()).open(request, *args, **kwargs)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

DISPATCH_AGENT_SCHEMA: Dict[str, Any] = {
    "name": "dispatch_agent",
    "description": (
        "Dispatch an async task to another Hermes profile's API server. "
        "Returns a run_id handle immediately — the task runs in a separate "
        "session on the target profile with its own model, skills, and memory. "
        "By default, results are auto-delivered to this session via SSE. Set "
        "delivery='none' for a truly detached run with no listener or callback. "
        "Every call is recorded in the durable SQLite ledger. After an origin "
        "restart, use "
        "dispatch_status and check_dispatch to recover and poll known runs. "
        "Multiple dispatches can be issued in one turn for parallel execution. "
        "For synchronous multi-turn conversations with session persistence, "
        "use dispatch_chat instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": (
                    "Profile name from hermes_herald.profiles config "
                    "(e.g. 'vincent', 'y10', 'qc-engagement')."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The task message to send to the target profile. "
                    "Be specific and self-contained — the target profile "
                    "has no context from this conversation."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional system prompt override for the target session. "
                    "Use to set role/behavior (e.g. 'You are a pedagogical "
                    "reviewer...')."
                ),
            },
            "delivery": {
                "type": "string",
                "enum": ["callback", "none"],
                "default": "callback",
                "description": (
                    "Result-delivery mode. 'callback' (default) monitors the "
                    "run and delivers its terminal result to this session. "
                    "'none' starts a truly detached fire-and-forget run: no "
                    "listener and no callback are created; use check_dispatch "
                    "or dispatch_status later. Detached tasks receive a small "
                    "routing context for optional graph forwarding; callback "
                    "chains receive it when lineage or max_hops is supplied."
                ),
            },
            "trace_id": {
                "type": "string",
                "description": (
                    "Optional graph trace identifier. When forwarding a task "
                    "that contains a HERALD ROUTING CONTEXT, preserve its "
                    "trace_id. Omit for a new graph."
                ),
            },
            "parent_edge_id": {
                "type": "string",
                "description": (
                    "Optional parent dispatch edge. When forwarding a detached "
                    "or guarded callback task, set this to the incoming routing "
                    "context's edge_id."
                ),
            },
            "parent_hop": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Incoming routing context's hop_count when forwarding a "
                    "task. Herald increments it; do not increment it "
                    "yourself. Omit for a new graph."
                ),
            },
            "max_hops": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional per-trace hop budget for dispatch graph work. "
                    "Preserve the incoming value when forwarding. Omit for no "
                    "Herald hop limit. A refused edge makes no network call."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional target model_routes alias for this run. The "
                    "plugin verifies the exact alias through the target's "
                    "authenticated /v1/models endpoint before starting the "
                    "task. Arbitrary model names and unverifiable aliases "
                    "fail closed; omit this field to use the target profile's "
                    "default runtime. Overrides a model alias set in the "
                    "hermes_herald profile config."
                ),
            },
        },
        "required": ["profile", "message"],
    },
}

CHECK_DISPATCH_SCHEMA: Dict[str, Any] = {
    "name": "check_dispatch",
    "description": (
        "Check the status of a dispatched task. Returns result if complete, "
        "or current status if still running."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by dispatch_agent.",
            },
            "profile": {
                "type": "string",
                "description": (
                    "Profile name the task was dispatched to (needed to "
                    "know which API server to poll)."
                ),
            },
        },
        "required": ["run_id", "profile"],
    },
}

COLLECT_DISPATCHES_SCHEMA: Dict[str, Any] = {
    "name": "collect_dispatches",
    "description": (
        "Check multiple dispatched tasks at once. Returns completed results "
        "and a list of still-running run_ids. Call this after dispatching "
        "multiple parallel tasks to gather whatever's done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_ids": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "profile": {"type": "string"},
                    },
                },
                "description": (
                    "List of {run_id, profile} objects to check."
                ),
            },
        },
        "required": ["run_ids"],
    },
}

DISPATCH_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "dispatch_status",
    "description": (
        "Query the durable SQLite dispatch ledger, including calls from "
        "previous sessions, and inspect the current profile's credential-free "
        "directed topology. Bounded v1 JSON history is migrated once into the "
        "ledger. Full task text is opt-in."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer", "minimum": 1, "maximum": 500,
                "default": 100,
                "description": "Maximum newest ledger rows to return.",
            },
            "target_profile": {"type": "string"},
            "origin_profile": {"type": "string"},
            "dispatch_type": {
                "type": "string", "enum": ["run", "chat"],
            },
            "status": {"type": "string"},
            "delivery": {
                "type": "string", "enum": ["callback", "none", "sync"],
            },
            "trace_id": {"type": "string"},
            "include_messages": {
                "type": "boolean", "default": False,
                "description": (
                    "Include full locally stored task text and instructions. "
                    "Defaults false; previews are always returned."
                ),
            },
            "include_topology": {
                "type": "boolean", "default": True,
                "description": (
                    "Include configured outbound routes and observed directed "
                    "edge counts. No API keys are returned."
                ),
            },
        },
    },
}

DISPATCH_CHAT_SCHEMA: Dict[str, Any] = {
    "name": "dispatch_chat",
    "description": (
        "Synchronously dispatch one conversational turn to another Hermes "
        "profile. The call blocks the current agent turn until the reply "
        "arrives, while a streaming /v1/chat/completions connection observes "
        "assistant and tool activity and preserves history through the target "
        "session ID. Subsequent calls to the same profile continue that "
        "conversation. Use model to select an exact target model_routes alias; "
        "omit it to use the target default. Optional instructions apply only "
        "to this turn. Use this when the next decision depends on the target's "
        "reply. For independent or parallel work, use dispatch_agent instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": (
                    "Profile name from hermes_herald.profiles config "
                    "(e.g. 'marie', 'ada', 'richard', 'rosalind')."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The message to send to the target profile. "
                    "Be specific and self-contained."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional system prompt override for this turn. "
                    "Resend it on subsequent calls when it is still required."
                ),
            },
            "new_session": {
                "type": "boolean",
                "description": (
                    "If true, start a fresh conversation instead of "
                    "continuing the existing session for this profile. "
                    "Default: false (continue existing session)."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional exact model_routes alias advertised by the "
                    "target's authenticated /v1/models endpoint. Herald "
                    "verifies the alias before sending. Omit to use the "
                    "target profile's default runtime."
                ),
            },
            "stall_timeout_seconds": {
                "type": "number",
                "minimum": 30,
                "description": (
                    "Seconds without assistant or tool activity before Herald "
                    "treats the chat as stalled. Activity resets the timer; "
                    "SSE transport keepalives do not. Overrides "
                    "hermes_herald.chat_timeout (default 600s). Minimum 30."
                ),
            },
        },
        "required": ["profile", "message"],
    },
}

CANCEL_DISPATCH_SCHEMA: Dict[str, Any] = {
    "name": "cancel_dispatch",
    "description": (
        "Cancel a running dispatched task by sending a stop request to the "
        "target profile's API server. The agent receives an interrupt signal "
        "and the asyncio task is cancelled. Also stops the local SSE listener "
        "so no stale result is delivered. The run may take a few seconds to "
        "actually stop (the agent finishes its current step first)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by dispatch_agent.",
            },
            "profile": {
                "type": "string",
                "description": (
                    "Profile name the task was dispatched to (needed to "
                    "know which API server to contact)."
                ),
            },
        },
        "required": ["run_id", "profile"],
    },
}


APPROVE_DISPATCH_SCHEMA: Dict[str, Any] = {
    "name": "approve_dispatch",
    "description": (
        "Resolve a pending approval request for a dispatched run. When a "
        "dispatched task invokes a protected terminal command, the target "
        "enters 'waiting_for_approval' and an approval-required notice is "
        "delivered to this session. Use this tool to resolve that exact "
        "request with an explicit choice — the run then resumes (or, for "
        "'deny', halts the protected command). Approval is scoped to the "
        "exact originating session and {profile, run_id, approval_id} request; "
        "the redacted command shown in the approval notice is the action being "
        "authorized. Never auto-approves "
        "— an explicit choice and confirmation through Hermes's human-owned "
        "approval surface are always required. This tool is only for API "
        "run IDs returned by dispatch_agent. Never use it for local terminal, "
        "cron, pending_approval, or /approve requests."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by dispatch_agent.",
            },
            "profile": {
                "type": "string",
                "description": (
                    "Profile name the task was dispatched to. Must match the "
                    "profile the run was dispatched to — approval cannot cross "
                    "profile boundaries."
                ),
            },
            "approval_id": {
                "type": "string",
                "description": (
                    "Fresh approval request ID shown in the approval-required "
                    "notice. It binds this decision to that exact delivered "
                    "request and prevents stale approval replay."
                ),
            },
            "choice": {
                "type": "string",
                "enum": ["once", "session", "always", "deny"],
                "description": (
                    "Approval choice: 'once' approves this command only, "
                    "'session' approves for the rest of the run, 'always' "
                    "approves for all future runs, 'deny' refuses the command."
                ),
            },
            "resolve_all": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, deny all currently pending approvals for this "
                    "run atomically. Bulk positive approval is refused because "
                    "only the current command was shown. Defaults to false."
                ),
            },
        },
        "required": ["run_id", "profile", "approval_id", "choice"],
    },
}


# delegate_subagent schema — defined here (before ALL_SCHEMAS) so the list
# can reference it. The handler and helper functions are at the bottom of
# the file.
DELEGATE_SUBAGENT_SCHEMA: Dict[str, Any] = {
    "name": "delegate_subagent",
    "description": (
        "Spawn an in-process subagent with a per-call model override. "
        "Unlike delegate_task (which inherits the parent model), this tool "
        "lets you pick a different model for the subagent — useful for "
        "fan-out across models (e.g. review code with a fast model while "
        "the parent uses a reasoning model). The subagent runs in the same "
        "process with its own isolated context, terminal session, and "
        "toolset. The immediate return contains task metadata; the final "
        "result (summary or error) is auto-delivered when the child completes.\n\n"
        "The model name is resolved leniently via the same /model switch "
        "pipeline — bare names like 'opus', 'gpt-5', 'glm' work, as do "
        "full 'vendor/model' slugs. If the resolved provider differs from "
        "the parent's, the subagent gets fresh credentials for that provider. "
        "If it's the same provider/aggregator, credentials are inherited. "
        "Set inherit_soul=true to load the active parent profile's full "
        "SOUL.md as the child's identity; it is off by default.\n\n"
        "Runs asynchronously in a daemon background thread and returns a "
        "task_id immediately. Activity resets a stall timer (10 minutes by "
        "default), so productive children can run indefinitely. An optional "
        "wall-clock threshold requests cooperative interruption for that call. "
        "The final result (summary or error) is auto-delivered as "
        "a new message when the child finishes. This is in-process rather "
        "than durable: use dispatch_agent for profile isolation or work that "
        "must survive the current process."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What the subagent should accomplish. Be specific and "
                    "self-contained — the subagent knows nothing about "
                    "your conversation history."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Model for this subagent (e.g. 'gemma4:31b', 'glm-5.2', "
                    "'opus', 'gpt-5'). Resolved via the model switch pipeline. "
                    "If omitted, inherits the parent model (same as "
                    "delegate_task)."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Background information the subagent needs: file paths, "
                    "error messages, project structure, constraints. This is "
                    "always included whether or not inherit_context is enabled."
                ),
            },
            "inherit_context": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, prepend a bounded copy of recent parent "
                    "user/assistant text to context. System messages, tool "
                    "calls, tool results, memory, and hidden state are excluded. "
                    "Defaults to false."
                ),
            },
            "inherit_soul": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, load the active parent profile's full SOUL.md "
                    "as the subagent's primary identity. Defaults to false. "
                    "This does not inherit conversation history, USER.md, "
                    "memory, or project context files."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional exact toolset subset for the subagent. A non-empty "
                    "list is intersected with the parent's capabilities; an "
                    "empty list creates a model-only child. If omitted, "
                    "inherit_toolsets controls the behaviour."
                ),
            },
            "inherit_toolsets": {
                "type": "boolean",
                "default": True,
                "description": (
                    "When toolsets is omitted, inherit the parent's safe "
                    "toolset subset if true, or create a model-only child if "
                    "false. Defaults to true. Explicit toolsets always wins."
                ),
            },
            "stall_timeout_seconds": {
                "type": "number",
                "minimum": 30,
                "default": 600,
                "description": (
                    "Seconds without child activity before interruption is "
                    "requested because the child appears stalled. Tool events "
                    "and streamed response text reset the timer. Defaults to "
                    "600 seconds."
                ),
            },
            "interrupt_after_seconds": {
                "type": "number",
                "minimum": 30,
                "description": (
                    "Optional wall-clock threshold after which cooperative "
                    "interruption is requested. This stops waiting and reports "
                    "a timeout, but cannot guarantee immediate termination of "
                    "a blocking provider or tool call. Omit it to allow an "
                    "active child to run indefinitely."
                ),
            },
        },
        "required": ["goal"],
    },
}


LLM_CALL_SCHEMA: Dict[str, Any] = {
    "name": "llm_call",
    "description": (
        "Make a bare LLM inference call without the full agent loop, tool "
        "schemas, or subagent overhead. Send messages and get text back. "
        "Runs synchronously through the host's provider routing, fallback "
        "chains, and credential pool — no API keys to manage.\n\n"
        "Use this for: classification, translation, summarization, scoring, "
        "extraction, or any task where you just need the model's response "
        "without tool-calling or agentic context.\n\n"
        "The returned provider is populated only when the provider response "
        "explicitly identifies itself; requested_provider and "
        "configured_provider preserve routing intent without mislabeling a "
        "fallback provider as the server that actually handled the call.\n\n"
        "For a full agent with tools and terminal access, use "
        "delegate_subagent or dispatch_agent instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["system", "user", "assistant"],
                            "description": "Message role.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Message content.",
                        },
                    },
                    "required": ["role", "content"],
                },
                "description": (
                    "Chat messages to send. Each is {role, content}. "
                    "Use role='user' for your prompt, role='assistant' "
                    "for prior turns, role='system' for instructions. "
                    "If you pass a system message here, do NOT also "
                    "pass system_prompt — use one or the other."
                ),
            },
            "system_prompt": {
                "type": "string",
                "description": (
                    "Optional system prompt. Prepended as a system-role "
                    "message before your messages list. Use this OR include "
                    "a system message in 'messages', not both."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model override (e.g. 'gemma4:31b', "
                    "'gpt-5', 'claude-sonnet-4'). Uses the active "
                    "model by default."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional provider override (e.g. 'openrouter', "
                    "'anthropic', 'ollama-cloud'). Uses the active "
                    "provider by default."
                ),
            },
            "temperature": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 2.0,
                "description": "Optional sampling temperature (0.0 to 2.0).",
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Best-effort provider-dependent output-token hint. "
                    "Hermes omits token caps on routes that reject them, so "
                    "this is not guaranteed on every provider."
                ),
            },
            "json_mode": {
                "type": "boolean",
                "description": (
                    "If true, add a cross-provider JSON-only instruction, "
                    "request response_format where supported, and reject "
                    "responses that are not valid JSON objects. Default: false."
                ),
            },
        },
        "required": ["messages"],
    },
}


PING_PROFILE_SCHEMA: Dict[str, Any] = {
    "name": "ping_profile",
    "description": (
        "Check if a target Hermes profile's API server is reachable. "
        "Returns status (up/down), response time in ms, and model if "
        "available. Use before dispatch_agent/dispatch_chat to verify "
        "the target is running instead of discovering it via a timeout."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": (
                    "Profile name from hermes_herald.profiles config."
                ),
            },
        },
        "required": ["profile"],
    },
}


LIST_PROFILE_MODELS_SCHEMA: Dict[str, Any] = {
    "name": "list_profile_models",
    "description": (
        "List the models a target profile can safely accept for dispatch_agent. "
        "Returns the target's advertised primary identity for information and "
        "separately lists exact "
        "model_routes aliases that may be passed as dispatch_agent(model=...)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": "Profile name from hermes_herald.profiles config.",
            },
        },
        "required": ["profile"],
    },
}


ALL_SCHEMAS = [
    DISPATCH_AGENT_SCHEMA,
    CHECK_DISPATCH_SCHEMA,
    COLLECT_DISPATCHES_SCHEMA,
    DISPATCH_STATUS_SCHEMA,
    DISPATCH_CHAT_SCHEMA,
    CANCEL_DISPATCH_SCHEMA,
    DELEGATE_SUBAGENT_SCHEMA,
    LLM_CALL_SCHEMA,
    PING_PROFILE_SCHEMA,
    APPROVE_DISPATCH_SCHEMA,
    LIST_PROFILE_MODELS_SCHEMA,
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, api_key: str, body: dict, timeout: float = 30.0) -> dict:
    """POST JSON to the API server. Returns the parsed JSON response."""
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


def _post_streaming_chat(
    url: str,
    api_key: str,
    body: dict,
    *,
    session_id: str,
    stall_timeout_seconds: float,
    progress_callback=None,
) -> dict:
    """Consume one OpenAI-compatible streaming chat turn with stall detection."""
    data = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Hermes-Session-Id": session_id,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=max(1.0, min(stall_timeout_seconds, 60.0))) as resp:
            result_session_id = resp.headers.get("X-Hermes-Session-Id", session_id)
            read_available = getattr(resp, "read1", None) or resp.read
            buffer = ""
            reply_parts: List[str] = []
            usage: Dict[str, int] = {}
            response_model = ""
            last_activity = time.monotonic()
            saw_done = False

            while True:
                chunk = read_available(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")

                while "\n\n" in buffer:
                    event_block, buffer = buffer.split("\n\n", 1)
                    stripped = event_block.strip()
                    if not stripped:
                        continue
                    if stripped.startswith(":"):
                        if time.monotonic() - last_activity > stall_timeout_seconds:
                            raise RuntimeError(
                                f"dispatch_chat stalled for {stall_timeout_seconds:g} seconds "
                                "without assistant or tool activity."
                            )
                        continue

                    event_name = "message"
                    data_lines: List[str] = []
                    for line in event_block.split("\n"):
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    if not data_lines:
                        continue
                    payload_text = "\n".join(data_lines)
                    if payload_text == "[DONE]":
                        saw_done = True
                        break
                    try:
                        payload = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue

                    if event_name == "hermes.tool.progress":
                        last_activity = time.monotonic()
                        if progress_callback:
                            progress_callback(event_name, payload)
                        continue

                    choices = payload.get("choices") if isinstance(payload, dict) else None
                    if isinstance(choices, list) and choices:
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        raw_delta = choice.get("delta")
                        delta: Dict[str, Any] = raw_delta if isinstance(raw_delta, dict) else {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            reply_parts.append(content)
                            last_activity = time.monotonic()
                        elif delta.get("role"):
                            last_activity = time.monotonic()
                        finish_reason = choice.get("finish_reason")
                        if finish_reason and finish_reason != "stop":
                            raw_error = payload.get("error")
                            error: Dict[str, Any] = (
                                raw_error if isinstance(raw_error, dict) else {}
                            )
                            detail = error.get("message") or f"finish_reason={finish_reason}"
                            raise RuntimeError(f"Target chat failed: {detail}")
                    if isinstance(payload, dict):
                        if isinstance(payload.get("model"), str):
                            response_model = payload["model"]
                        raw_usage = payload.get("usage")
                        if isinstance(raw_usage, dict):
                            usage = {
                                "input_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                                "output_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
                                "total_tokens": int(raw_usage.get("total_tokens", 0) or 0),
                            }
                if saw_done:
                    break

            if not saw_done:
                raise RuntimeError("Target chat stream closed before data: [DONE].")
            return {
                "session_id": result_session_id,
                "reply": "".join(reply_parts),
                "model": response_model,
                "usage": usage,
            }
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


def _post_empty(url: str, api_key: str, timeout: float = 10.0) -> dict:
    """POST with no body to the API server (e.g. /stop endpoint). Returns parsed JSON."""
    req = Request(
        url,
        data=b"",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


def _get_json(url: str, api_key: str, timeout: float = 10.0) -> dict:
    """GET from the API server. Returns the parsed JSON response."""
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            return {"run_id": url.rsplit("/", 1)[-1], "status": "not_found"}
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {err_body or e.reason}"
        ) from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

_MAX_STATE_ENTRIES = 200

# Serializes all state file reads/writes — SSE callback threads and
# tool handler threads can race on read-modify-write cycles without this.
_state_lock = threading.Lock()


def _load_state() -> dict:
    """Load the run-state JSON file. Returns empty structure if missing."""
    path = cfg.get_state_file_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"runs": []}
        return data
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return {"runs": []}


def _save_state(data: dict) -> None:
    """Atomically write the run-state JSON file."""
    path = cfg.get_state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Trim old entries
    runs = data.get("runs", [])
    if len(runs) > _MAX_STATE_ENTRIES:
        data["runs"] = runs[-_MAX_STATE_ENTRIES:]
    # Atomic write: temp file + replace (cross-platform, overwrites existing)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _preflight_dispatch_ledger() -> None:
    from .ledger import preflight

    preflight()


def _record_dispatch_ledger(**record) -> None:
    from .ledger import record_dispatch

    record_dispatch(**record)


def _list_dispatch_ledger(**filters) -> list[dict]:
    from .ledger import list_dispatches

    return list_dispatches(**filters)


def _known_dispatch_run_ids(origin_profile: str) -> set[str]:
    from .ledger import known_run_ids

    return known_run_ids(origin_profile)


def _migrate_legacy_run_history(state: dict) -> int:
    """Import bounded v1 JSON history into SQLite once, without duplicates."""
    origin = cfg.get_active_profile_name()
    known = _known_dispatch_run_ids(origin)
    migrated = 0
    for run in state.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "").strip()
        profile = str(run.get("profile") or "").strip()
        if not run_id or not profile or run_id in known:
            continue
        dispatch_type = "chat" if run.get("type") == "chat" else "run"
        edge_id = str(run.get("edge_id") or "").strip() or uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"hermes-herald:{origin}:{run_id}",
        ).hex
        record = {
            "edge_id": edge_id,
            "run_id": run_id,
            "origin_profile": origin,
            "target_profile": profile,
            "dispatch_type": dispatch_type,
            "delivery": str(
                run.get("delivery")
                or ("sync" if dispatch_type == "chat" else "callback")
            ),
            # The v1 cache retained only these bounded previews. Mark their
            # provenance explicitly rather than pretending full text survived.
            "message": str(run.get("message_preview") or ""),
            "trace_id": str(run.get("trace_id") or ""),
            "parent_edge_id": str(run.get("parent_edge_id") or ""),
            "hop_count": int(run.get("hop_count") or 1),
            "max_hops": run.get("max_hops"),
            "origin_session_id": str(run.get("session_id") or ""),
            "requested_model": str(run.get("requested_model") or ""),
            "resolved_model": str(run.get("resolved_model") or run.get("model") or ""),
            "model_resolution": "legacy_state_cache",
            "status": str(run.get("status") or "unknown"),
            "output_preview": str(run.get("output_preview") or ""),
            "duration_seconds": run.get("duration_seconds"),
            "usage": run.get("usage") if isinstance(run.get("usage"), dict) else {},
            "dispatched_at": str(run.get("dispatched_at") or ""),
            "completed_at": str(run.get("completed_at") or ""),
        }
        try:
            _record_dispatch_ledger(**record)
        except Exception:
            # Another local caller may have won the deterministic migration
            # race. Suppress only that proven duplicate; real failures surface.
            if run_id not in _known_dispatch_run_ids(origin):
                raise
        known.add(run_id)
        migrated += 1
    return migrated


def _observed_dispatch_edges() -> list[dict]:
    from .ledger import observed_edges

    return observed_edges()


def _update_dispatch_ledger(run_id: str, status: str, **fields) -> None:
    from .ledger import update_dispatch

    update_dispatch(
        run_id=run_id,
        origin_profile=cfg.get_active_profile_name(),
        status=status,
        **fields,
    )


def _routing_envelope(
    message: str,
    *,
    trace_id: str,
    edge_id: str,
    parent_edge_id: str,
    hop_count: int,
    max_hops: Optional[int],
    delivery: str,
) -> str:
    """Attach explicit, model-visible lineage to async graph work."""
    lines = [
        "[HERALD ROUTING CONTEXT]",
        f"trace_id: {trace_id}",
        f"edge_id: {edge_id}",
        f"hop_count: {hop_count}",
    ]
    if parent_edge_id:
        lines.append(f"parent_edge_id: {parent_edge_id}")
    if max_hops is not None:
        lines.append(f"max_hops: {max_hops}")
    lines.extend([
        f"When forwarding this work with dispatch_agent, keep delivery='{delivery}', "
        "preserve trace_id and max_hops, set parent_edge_id to this edge_id, "
        "and pass this hop_count as parent_hop.",
        "[/HERALD ROUTING CONTEXT]",
        "",
        message,
    ])
    return "\n".join(lines)


def _persist_run(
    run_id: str,
    profile: str,
    message_preview: str,
    model: str = "",
    requested_model: str = "",
    resolved_model: str = "",
    model_resolution: str = "",
    edge_id: str = "",
    trace_id: str = "",
    parent_edge_id: str = "",
    delivery: str = "callback",
    hop_count: int = 1,
    max_hops: Optional[int] = None,
) -> None:
    """Add a run entry to the state file with explicit model provenance.

    ``model`` is retained for compatibility with existing state consumers.
    New dispatches separately record the requested route alias and the model
    resolved by authenticated target route discovery. Neither field is
    inferred from run status/completion echoes.
    """
    with _state_lock:
        state = _load_state()
        state.setdefault("runs", []).append({
            "run_id": run_id,
            "profile": profile,
            "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message_preview": message_preview[:120],
            "session_id": os.environ.get("HERMES_SESSION_ID", ""),
            "model": model or resolved_model or "",
            "requested_model": requested_model or "",
            "resolved_model": resolved_model or "",
            "model_resolution": model_resolution or "",
            "edge_id": edge_id,
            "trace_id": trace_id,
            "parent_edge_id": parent_edge_id,
            "delivery": delivery,
            "hop_count": hop_count,
            "max_hops": max_hops,
            "status": "dispatched",
            "completed_at": "",
            "duration_seconds": None,
            "output_preview": "",
            "usage": {},
        })
        _save_state(state)


def _update_run_status(
    run_id: str,
    status: str,
    output_preview: str = "",
    duration_seconds: Optional[float] = None,
    usage: Optional[dict] = None,
    model: str = "",
    requested_model: str = "",
    resolved_model: str = "",
) -> None:
    """Update a run entry with terminal status and provenance.

    Called when a dispatch completes (via SSE callback or dispatch_chat reply),
    fails, or is cancelled. Merges the new fields into the existing run record
    without clobbering the dispatch-time metadata.
    """
    try:
        with _state_lock:
            state = _load_state()
            runs = state.get("runs", [])
            for run in runs:
                if run.get("run_id") == run_id:
                    run["status"] = status
                    if status in {"completed", "failed", "cancelled"}:
                        run["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    else:
                        run["completed_at"] = ""
                    if output_preview:
                        run["output_preview"] = output_preview[:500]
                    if duration_seconds is not None:
                        run["duration_seconds"] = round(duration_seconds, 2)
                    if usage:
                        run["usage"] = usage
                    if model:
                        run["model"] = model
                    if requested_model:
                        run["requested_model"] = requested_model
                    if resolved_model:
                        run["resolved_model"] = resolved_model
                    break
            _save_state(state)
    except Exception as exc:
        logger.warning("Could not update recovery state for %s: %s", run_id, exc)

    try:
        _update_dispatch_ledger(
            run_id,
            status,
            output_preview=output_preview,
            duration_seconds=duration_seconds,
            usage=usage,
            requested_model=requested_model,
            resolved_model=resolved_model or model,
        )
    except Exception as exc:
        logger.warning("Could not update dispatch ledger for %s: %s", run_id, exc)


def _update_pending_approval(
    run_id: str,
    approval_data: dict,
    approval_queue: Optional[list[dict]] = None,
) -> None:
    """Record redacted pending-approval metadata on a run in the state file.

    Called by the SSE listener when an ``approval.request`` event arrives.
    The ``approval_data`` is already redacted by the target (command preview,
    description, choices) and contains no credentials. Marks the run status as
    ``waiting_for_approval`` so it surfaces distinctly in dispatch_status and
    survives a plugin/session restart for recovery.
    """
    with _state_lock:
        state = _load_state()
        for run in state.get("runs", []):
            if run.get("run_id") == run_id:
                run["pending_approval"] = approval_data
                if approval_queue is not None:
                    run["pending_approval_queue"] = approval_queue
                run["status"] = "waiting_for_approval"
                break
        _save_state(state)
    try:
        _update_dispatch_ledger(run_id, "waiting_for_approval")
    except Exception as exc:
        logger.warning(
            "Could not update approval status in ledger for %s: %s",
            run_id,
            exc,
        )


def _clear_pending_approval(run_id: str) -> None:
    """Remove pending-approval metadata from a run in the state file.

    Called after the approval is resolved (approve_dispatch / approval.responded),
    the run terminates, is cancelled, or a 409 indicates no active approval.
    Leaves the run record's status untouched so terminal handlers can still set
    the final status.
    """
    with _state_lock:
        state = _load_state()
        for run in state.get("runs", []):
            if run.get("run_id") == run_id:
                run.pop("pending_approval", None)
                run.pop("pending_approval_queue", None)
                break
        _save_state(state)


def _annotate_run_status_model(status: dict, run_id: str) -> None:
    """Label target run-status model metadata without treating it as proof.

    Hermes currently stores the submitted ``body.model`` in run status. That
    value is a request echo even when no route matched, so expose it under an
    explicit name and attach locally verified route provenance when available.
    """
    reported_model = status.pop("model", None)
    if reported_model:
        status["api_reported_model"] = reported_model
        status["api_reported_model_is_runtime_evidence"] = False

    with _state_lock:
        state = _load_state()
        persisted = next(
            (run for run in state.get("runs", []) if run.get("run_id") == run_id),
            None,
        )
    if not persisted:
        return
    for field in ("requested_model", "resolved_model", "model_resolution"):
        if persisted.get(field):
            status[field] = persisted[field]


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _tool_error(msg: str) -> str:
    """Return a JSON error string for the tool result."""
    return json.dumps({"status": "error", "error": msg})


def _resolve_profile(
    profile: str,
    operation: Optional[str] = None,
) -> tuple[dict, Optional[str]]:
    """Resolve a profile name to its config dict.

    Returns (config_dict, error_message). If the profile is found,
    error_message is None. If not found, config_dict is an empty dict
    and error_message explains why.
    """
    pcfg = cfg.get_profile_config(profile) or {}
    if not pcfg:
        available = cfg.list_profiles()
        avail_str = ", ".join(available) if available else "(none configured)"
        return {}, (
            f"Profile '{profile}' not found in hermes_herald.profiles "
            f"config. Available profiles: {avail_str}"
        )
    if operation in {"dispatch", "chat"}:
        origin = cfg.get_active_profile_name()
        if profile == origin and not cfg.allow_self_routing():
            return {}, (
                f"Self-routing is disabled for active profile '{origin}'. "
                "Set hermes_herald.allow_self: true and keep an explicit "
                "self profile entry to enable it."
            )
        capabilities = cfg.get_route_capabilities(profile)
        if operation not in capabilities:
            return {}, (
                f"Outbound route from '{origin}' to '{profile}' does not "
                f"allow {operation}. Configured capabilities: "
                f"{', '.join(capabilities) if capabilities else '(none)'}"
            )
    url = pcfg.get("url", "").strip()
    if not url:
        return {}, f"Profile '{profile}' has no 'url' configured."
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return {}, (
            f"Profile '{profile}' has an invalid URL. Only absolute http:// "
            f"or https:// API-server URLs are supported."
        )
    if parsed.username is not None or parsed.password is not None:
        return {}, f"Profile '{profile}' URL must not contain userinfo credentials."
    api_key = pcfg.get("api_key", "")
    if not api_key:
        return {}, (
            f"Profile '{profile}' has no 'api_key' configured (or the env "
            f"var it references is not set)."
        )
    resolved = dict(pcfg)
    resolved["url"] = url.rstrip("/")
    return resolved, None


def _verify_run_model_route(
    profile: str,
    pcfg: dict,
    requested_model: str,
) -> tuple[dict, Optional[str]]:
    """Verify an explicit /v1/runs model as a target model_routes alias.

    Hermes treats the request ``model`` field as an exact lookup key in the
    target API server's ``platforms.api_server.extra.model_routes`` mapping.
    Unknown values are accepted by the API but run on the target default, so
    this plugin must discover and verify the route before starting a task.

    ``GET /v1/models`` identifies route aliases with a non-empty ``parent``
    and exposes their configured target model in ``root``.  The endpoint's
    primary entry has ``parent: null`` and is only an API identity/default; it
    is not evidence that an explicit request caused model routing.
    """
    models_url = f"{pcfg['url']}/v1/models"
    try:
        response = _get_json(models_url, pcfg["api_key"], timeout=10.0)
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return {}, (
                f"Cannot verify model route '{requested_model}' for {profile}: "
                f"authentication to {models_url} failed. Check the profile "
                f"api_key. No task was started. ({msg})"
            )
        if "Cannot reach" in msg:
            return {}, (
                f"Cannot verify model route '{requested_model}' for {profile}: "
                f"the target API server is unreachable. No task was started. "
                f"({msg})"
            )
        return {}, (
            f"Cannot verify model route '{requested_model}' for {profile}. "
            f"No task was started. ({msg})"
        )
    except Exception as e:
        return {}, (
            f"Cannot verify model route '{requested_model}' for {profile}: "
            f"invalid response from {models_url}. No task was started. "
            f"({type(e).__name__})"
        )

    entries = response.get("data") if isinstance(response, dict) else None
    if not isinstance(entries, list):
        return {}, (
            f"Cannot verify model route '{requested_model}' for {profile}: "
            f"the target does not expose a compatible authenticated "
            f"/v1/models route listing. Upgrade/configure the target API "
            f"server or omit model to use its default. No task was started."
        )

    route_entries = [
        entry for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and isinstance(entry.get("parent"), str)
        and entry["parent"].strip()
    ]
    matched = next(
        (entry for entry in route_entries if entry["id"] == requested_model),
        None,
    )
    if matched is not None:
        resolved_model = matched.get("root")
        if isinstance(resolved_model, str) and resolved_model.strip():
            return {
                "requested_model": requested_model,
                "resolved_model": resolved_model.strip(),
                "resolution_source": "target_model_routes",
            }, None
        return {}, (
            f"Cannot verify model route '{requested_model}' for {profile}: "
            f"the target advertised the alias without a resolved root model. "
            f"No task was started."
        )

    advertised_primary = any(
        isinstance(entry, dict)
        and entry.get("id") == requested_model
        and not entry.get("parent")
        for entry in entries
    )
    available = sorted(entry["id"] for entry in route_entries)
    available_text = ", ".join(available) if available else "(none)"
    if advertised_primary:
        reason = (
            "it is only the target's advertised default/API identity, not a "
            "configured model_routes alias"
        )
    else:
        reason = "it is not an exact configured model_routes alias"
    return {}, (
        f"Model override '{requested_model}' is not supported for {profile}: "
        f"{reason}. Available route aliases: {available_text}. Configure an "
        f"exact platforms.api_server.extra.model_routes alias on the target, "
        f"pass one of the available aliases, or omit model to use the target "
        f"default. No task was started."
    )


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _async_delivery_supported() -> bool:
    """Whether this commissioning session can receive a detached result."""
    try:
        from gateway.session_context import async_delivery_supported

        return bool(async_delivery_supported())
    except Exception:
        # Detached work without a positive delivery capability can become an
        # invisible orphan. Missing/broken seams therefore fail closed.
        return False

def handle_dispatch_agent(args: dict, **kwargs) -> str:
    """Dispatch an async task to another Hermes profile."""
    profile = args.get("profile", "").strip()
    message = args.get("message", "")
    instructions = args.get("instructions")
    model_override = args.get("model")
    delivery = args.get("delivery", "callback")
    trace_id = args.get("trace_id", "")
    parent_edge_id = args.get("parent_edge_id", "")
    parent_hop = args.get("parent_hop", 0)
    max_hops = args.get("max_hops")

    if not profile:
        return _tool_error("'profile' is required.")
    if not message:
        return _tool_error("'message' is required.")
    if delivery not in {"callback", "none"}:
        return _tool_error("'delivery' must be 'callback' or 'none'.")
    if not isinstance(trace_id, str) or not isinstance(parent_edge_id, str):
        return _tool_error("'trace_id' and 'parent_edge_id' must be strings.")
    if isinstance(parent_hop, bool) or not isinstance(parent_hop, int) or parent_hop < 0:
        return _tool_error("'parent_hop' must be a non-negative integer.")
    if max_hops is not None and (
        isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1
    ):
        return _tool_error("'max_hops' must be a positive integer when provided.")
    if parent_edge_id and "parent_hop" not in args:
        return _tool_error(
            "Forwarded work with parent_edge_id must also include the "
            "incoming routing context's parent_hop."
        )
    if parent_hop and not parent_edge_id:
        return _tool_error("'parent_hop' requires 'parent_edge_id'.")
    hop_count = parent_hop + 1
    if max_hops is not None and hop_count > max_hops:
        return _tool_error(
            f"Detached trace '{trace_id or '(new)'}' reached its maximum hop "
            f"budget ({max_hops}); no remote run was started."
        )
    if delivery == "callback" and not _async_delivery_supported():
        return _tool_error(
            "dispatch_agent is asynchronous, but this session cannot receive "
            "detached results (for example, a stateless HTTP or one-shot "
            "session). No remote run was started. Use dispatch_chat for a "
            "synchronous result, or delivery='none' for an explicitly "
            "unmonitored fire-and-forget run."
        )

    pcfg, err = _resolve_profile(profile, operation="dispatch")
    if err:
        return _tool_error(err)

    try:
        _preflight_dispatch_ledger()
    except Exception as exc:
        return _tool_error(
            f"Dispatch ledger is unavailable; no remote run was started. ({exc})"
        )

    edge_id = uuid.uuid4().hex
    trace_id = trace_id.strip() or uuid.uuid4().hex
    parent_edge_id = parent_edge_id.strip()

    # Build request body — omit null/empty fields. An explicit tool argument
    # takes precedence over a configured profile alias.
    transmitted_message = message
    if delivery == "none" or parent_edge_id or max_hops is not None:
        transmitted_message = _routing_envelope(
            message,
            trace_id=trace_id,
            edge_id=edge_id,
            parent_edge_id=parent_edge_id,
            hop_count=hop_count,
            max_hops=max_hops,
            delivery=delivery,
        )
    body: Dict[str, Any] = {"input": transmitted_message}
    if instructions:
        body["instructions"] = instructions
    configured_model = pcfg.get("model")
    requested_model = model_override or configured_model or ""
    if requested_model and not isinstance(requested_model, str):
        return _tool_error(
            "'model' must be a string when provided in the tool call or "
            "hermes_herald profile config. No task was started."
        )
    requested_model = requested_model.strip()
    route_resolution: Dict[str, str] = {}
    if requested_model:
        route_resolution, route_error = _verify_run_model_route(
            profile, pcfg, requested_model,
        )
        if route_error:
            return _tool_error(route_error)
        # Send the exact verified alias. Sending the resolved root would miss
        # Hermes' exact alias lookup and silently select the target default.
        body["model"] = requested_model

    try:
        result = _post_json(
            f"{pcfg['url']}/v1/runs",
            pcfg["api_key"],
            body,
            timeout=30.0,
        )
    except RuntimeError as e:
        # Distinguish auth errors from connection errors
        msg = str(e)
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"hermes_herald.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"Dispatch to {profile} failed: {msg}")

    run_id = result.get("run_id", "")
    if not run_id:
        return _tool_error(
            f"Unexpected response from {profile}: no run_id in {json.dumps(result)}"
        )

    # Persist bounded live-recovery state and the durable audit edge. Capture
    # the origin before starting a background listener because ContextVars do
    # not necessarily survive arbitrary callback threads.
    resolved_model = route_resolution.get("resolved_model", "")
    from .callback import capture_session_routing

    routing = capture_session_routing(kwargs.get("parent_agent"))
    origin_session_id = (
        routing.get("session_key") or routing.get("session_id") or ""
    )
    recovery_warning = ""
    try:
        _persist_run(
            run_id,
            profile,
            message,
            model=resolved_model,
            requested_model=requested_model,
            resolved_model=resolved_model,
            model_resolution=route_resolution.get("resolution_source", ""),
            edge_id=edge_id,
            trace_id=trace_id,
            parent_edge_id=parent_edge_id,
            delivery=delivery,
            hop_count=hop_count,
            max_hops=max_hops,
        )
    except Exception as exc:
        recovery_warning = f"Profile-local recovery state could not be updated: {exc}"
        logger.warning("Remote run %s started without recovery-state write: %s", run_id, exc)
    try:
        _record_dispatch_ledger(
            edge_id=edge_id,
            run_id=run_id,
            origin_profile=cfg.get_active_profile_name(),
            target_profile=profile,
            dispatch_type="run",
            delivery=delivery,
            message=message,
            instructions=instructions or "",
            trace_id=trace_id,
            parent_edge_id=parent_edge_id,
            hop_count=hop_count,
            max_hops=max_hops,
            origin_session_id=origin_session_id,
            requested_model=requested_model,
            resolved_model=resolved_model,
            model_resolution=route_resolution.get("resolution_source", ""),
        )
    except Exception as exc:
        logger.error("Remote run %s started but ledger insert failed: %s", run_id, exc)
        return json.dumps({
            "run_id": run_id,
            "profile": profile,
            "status": "dispatched_unrecorded",
            "delivery": delivery,
            "edge_id": edge_id,
            "trace_id": trace_id,
            "error": f"Remote run started, but the dispatch ledger write failed: {exc}",
        })

    if delivery == "callback":
        # One thread consumes the single non-replayable SSE stream; if it
        # disconnects, the same thread switches to authenticated status polling.
        try:
            from .callback import start_listener
            start_listener(
                run_id=run_id,
                profile=profile,
                url=pcfg["url"],
                api_key=pcfg["api_key"],
                message_preview=message[:120],
                requested_model=requested_model,
                resolved_model=resolved_model,
                parent_agent=kwargs.get("parent_agent"),
            )
        except Exception as exc:
            logger.error("Could not start SSE listener for %s: %s", run_id, exc)
            return json.dumps({
                "run_id": run_id,
                "profile": profile,
                "status": "dispatched_unmonitored",
                "delivery": delivery,
                "edge_id": edge_id,
                "trace_id": trace_id,
                "error": (
                    "Remote run started, but callback monitoring could not be "
                    f"initialized. Use check_dispatch. ({exc})"
                ),
            })

    response = {
        "run_id": run_id,
        "profile": profile,
        "status": "dispatched",
        "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message_preview": message[:120],
        "delivery": delivery,
        "edge_id": edge_id,
        "trace_id": trace_id,
        "parent_edge_id": parent_edge_id,
        "hop_count": hop_count,
        "max_hops": max_hops,
        "callback": (
            "SSE — result will be delivered automatically when complete"
            if delivery == "callback" else "none"
        ),
    }
    if requested_model:
        response.update({
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            "model_resolution": "target_model_routes",
        })
    if recovery_warning:
        response["warning"] = recovery_warning
    return json.dumps(response)


def handle_check_dispatch(args: dict, **kwargs) -> str:
    """Check the status of a single dispatched task."""
    run_id = args.get("run_id", "").strip()
    profile = args.get("profile", "").strip()

    if not run_id:
        return _tool_error("'run_id' is required.")
    if not profile:
        return _tool_error("'profile' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    try:
        status = _get_json(
            f"{pcfg['url']}/v1/runs/{run_id}",
            pcfg["api_key"],
            timeout=10.0,
        )
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"hermes_herald.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"Check dispatch {run_id} on {profile} failed: {msg}")

    # If not_found, check state file for context
    if status.get("status") == "not_found":
        return _tool_error(
            f"Run {run_id} not found on {profile}. It may have been from "
            f"a previous gateway restart (runs are in-memory and don't "
            f"survive a restart). Check dispatch_status for persisted records."
        )

    # Enrich with profile name for the agent
    status["profile"] = profile
    _annotate_run_status_model(status, run_id)

    # Surface pending-approval metadata distinctly when the run is waiting.
    # Prefer the live in-memory metadata (fresh from the SSE stream); fall back
    # to the persisted state-file copy (e.g. after a restart). Never include
    # credentials — the approval metadata is already redacted by the target.
    if status.get("status") == "waiting_for_approval":
        pending = None
        try:
            from .callback import get_pending_approval
            pending = get_pending_approval(run_id)
        except Exception:
            pending = None
        if pending is None:
            with _state_lock:
                state = _load_state()
                for run in state.get("runs", []):
                    if run.get("run_id") == run_id:
                        pending = run.get("pending_approval")
                        break
        if pending:
            status["pending_approval"] = pending

    live_status = str(status.get("status") or "unknown")
    summary = status.get("summary") or status.get("output") or ""
    if not isinstance(summary, str):
        summary = json.dumps(summary, ensure_ascii=False)
    _update_run_status(
        run_id,
        live_status,
        output_preview=summary,
        duration_seconds=status.get("duration_seconds"),
        usage=status.get("usage") if isinstance(status.get("usage"), dict) else None,
    )

    return json.dumps(status)


def handle_collect_dispatches(args: dict, **kwargs) -> str:
    """Check multiple dispatched tasks at once."""
    run_ids = args.get("run_ids", [])
    if not run_ids:
        return _tool_error("'run_ids' is required (list of {run_id, profile} objects).")

    completed: List[dict] = []
    running: List[dict] = []
    failed: List[dict] = []

    for entry in run_ids:
        run_id = entry.get("run_id", "").strip()
        profile = entry.get("profile", "").strip()
        if not run_id or not profile:
            failed.append({
                "run_id": run_id or "?",
                "profile": profile or "?",
                "status": "invalid",
                "error": "Missing run_id or profile",
            })
            continue

        pcfg, err = _resolve_profile(profile)
        if err:
            failed.append({
                "run_id": run_id,
                "profile": profile,
                "status": "config_error",
                "error": err,
            })
            continue

        try:
            status = _get_json(
                f"{pcfg['url']}/v1/runs/{run_id}",
                pcfg["api_key"],
                timeout=10.0,
            )
        except RuntimeError as e:
            failed.append({
                "run_id": run_id,
                "profile": profile,
                "status": "error",
                "error": str(e),
            })
            continue

        status["profile"] = profile
        _annotate_run_status_model(status, run_id)
        live_status = str(status.get("status") or "unknown")
        summary = status.get("summary") or status.get("output") or ""
        if not isinstance(summary, str):
            summary = json.dumps(summary, ensure_ascii=False)
        _update_run_status(
            run_id,
            live_status,
            output_preview=summary,
            duration_seconds=status.get("duration_seconds"),
            usage=status.get("usage") if isinstance(status.get("usage"), dict) else None,
        )

        if status.get("status") == "completed":
            completed.append(status)
        elif status.get("status") in ("failed", "cancelled", "not_found", "invalid"):
            failed.append(status)
        else:
            # queued, running, or any other state
            running.append(status)

    return json.dumps({
        "completed": completed,
        "running": running,
        "failed": failed,
        "summary": {
            "total": len(run_ids),
            "completed": len(completed),
            "running": len(running),
            "failed": len(failed),
        },
    })


def handle_dispatch_status(args: dict, **kwargs) -> str:
    """Query durable call history plus live approval and topology state."""
    try:
        limit = int(args.get("limit", 100))
    except (TypeError, ValueError):
        return _tool_error("'limit' must be an integer.")
    if not 1 <= limit <= 500:
        return _tool_error("'limit' must be between 1 and 500.")
    include_messages = args.get("include_messages", False)
    include_topology = args.get("include_topology", True)
    if not isinstance(include_messages, bool) or not isinstance(include_topology, bool):
        return _tool_error("'include_messages' and 'include_topology' must be booleans.")
    state = _load_state()
    try:
        migrated = _migrate_legacy_run_history(state)
        dispatches = _list_dispatch_ledger(
            limit=limit,
            include_messages=include_messages,
            origin_profile=str(args.get("origin_profile", "") or ""),
            target_profile=str(args.get("target_profile", "") or ""),
            dispatch_type=str(args.get("dispatch_type", "") or ""),
            status=str(args.get("status", "") or ""),
            delivery=str(args.get("delivery", "") or ""),
            trace_id=str(args.get("trace_id", "") or ""),
        )
    except Exception as exc:
        return _tool_error(f"Could not query the dispatch ledger: {exc}")

    # Approval metadata remains in the bounded recovery cache because it is
    # transient control state, not immutable call provenance.
    awaiting = [
        run for run in state.get("runs", [])
        if run.get("status") == "waiting_for_approval"
        or run.get("pending_approval")
    ]
    response = {
        "total": len(dispatches),
        "legacy_runs_migrated": migrated,
        "awaiting_approval": awaiting,
        "dispatches": dispatches,
        # Compatibility alias for callers written against Herald v1 state.
        "runs": dispatches,
    }
    if include_topology:
        topology = cfg.describe_topology()
        topology["observed_edges"] = _observed_dispatch_edges()
        response["topology"] = topology
    return json.dumps(response, indent=2)


def handle_dispatch_chat(args: dict, **kwargs) -> str:
    """Synchronously dispatch a task with session persistence.

    Uses streaming POST /v1/chat/completions with X-Hermes-Session-Id so the
    target loads/saves history while Herald observes model and tool activity.

    Session IDs are tracked per-profile in callback.py so subsequent
    calls continue the same conversation thread.
    """
    profile = args.get("profile", "").strip()
    message = args.get("message", "")
    instructions = args.get("instructions")
    new_session = args.get("new_session", False)
    model_override = args.get("model")
    call_timeout = args.get("stall_timeout_seconds")

    if not profile:
        return _tool_error("'profile' is required.")
    if not message:
        return _tool_error("'message' is required.")

    pcfg, err = _resolve_profile(profile, operation="chat")
    if err:
        return _tool_error(err)

    try:
        _preflight_dispatch_ledger()
    except Exception as exc:
        return _tool_error(
            f"Dispatch ledger is unavailable; no message was sent. ({exc})"
        )

    configured_model = pcfg.get("model")
    requested_model = model_override or configured_model or ""
    model_provenance = {
        "requested_model": "",
        "resolved_model": "",
        "resolution_source": "target_default",
    }
    if requested_model:
        model_provenance, model_error = _verify_run_model_route(
            profile, pcfg, requested_model,
        )
        if model_error:
            return _tool_error(model_error)

    from .callback import get_profile_session_id, store_profile_session_id

    # Get or create a session for this profile
    session_id = get_profile_session_id(profile)

    if not session_id or new_session:
        # Create a new session on the target API server
        # Hermes session titles are unique. Keep the target recognisable while
        # allowing repeated ``new_session=True`` calls and recovery after a
        # lost local session handle.
        create_body: Dict[str, Any] = {
            "title": f"Dispatch to {profile} · {uuid.uuid4().hex[:8]}"
        }
        try:
            create_result = _post_json(
                f"{pcfg['url']}/api/sessions",
                pcfg["api_key"],
                create_body,
                timeout=10.0,
            )
        except RuntimeError as e:
            msg = str(e)
            if "Cannot reach" in msg:
                return _tool_error(
                    f"Cannot reach {profile} API server at {pcfg['url']}. "
                    f"Is the gateway running for that profile? ({msg})"
                )
            return _tool_error(f"Failed to create session on {profile}: {msg}")

        session_obj = create_result.get("session", {})
        session_id = session_obj.get("id", "")
        if not session_id:
            return _tool_error(
                f"Unexpected response creating session on {profile}: {json.dumps(create_result)}"
            )
        store_profile_session_id(profile, session_id)

    # Resolve timeout: per-call override > config default (600s)
    from .config import get_chat_timeout
    effective_timeout = get_chat_timeout()
    if call_timeout is not None:
        try:
            effective_timeout = float(call_timeout)
        except (TypeError, ValueError):
            return _tool_error(
                "'stall_timeout_seconds' must be a number when provided."
            )
        if effective_timeout < 30:
            return _tool_error("'stall_timeout_seconds' must be at least 30 seconds.")

    # Stream one OpenAI-compatible turn while preserving target session history.
    chat_messages: List[Dict[str, Any]] = []
    if instructions:
        chat_messages.append({"role": "system", "content": instructions})
    chat_messages.append({"role": "user", "content": message})
    chat_body: Dict[str, Any] = {
        "model": requested_model or "hermes-agent",
        "messages": chat_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    parent_agent = _resolve_parent_agent(kwargs.get("parent_agent"))
    parent_progress = getattr(parent_agent, "tool_progress_callback", None)

    def _relay_progress(_event_name: str, payload: dict) -> None:
        if not parent_progress:
            return
        status = str(payload.get("status") or "running")
        event_type = "tool.completed" if status == "completed" else "tool.started"
        tool_name = str(payload.get("tool") or "remote_tool")
        preview = f"[{profile}] {payload.get('label') or tool_name}"
        try:
            parent_progress(event_type, tool_name=tool_name, preview=preview)
        except Exception:
            logger.debug("Could not relay dispatch_chat progress", exc_info=True)

    chat_started = time.monotonic()
    try:
        result = _post_streaming_chat(
            f"{pcfg['url']}/v1/chat/completions",
            pcfg["api_key"],
            chat_body,
            session_id=session_id,
            stall_timeout_seconds=effective_timeout,
            progress_callback=_relay_progress,
        )
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"hermes_herald.profiles config. ({msg})"
            )
        if "404" in msg:
            # Session was deleted/expired — clear cache and retry with new session
            store_profile_session_id(profile, "")
            return _tool_error(
                f"Session {session_id} not found on {profile}. It may have "
                f"expired. Retry with new_session=true to start a fresh conversation."
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"dispatch_chat to {profile} failed: {msg}")

    # Extract the reply
    reply = result.get("reply", "")
    result_session_id = result.get("session_id", session_id)
    if result_session_id != session_id:
        store_profile_session_id(profile, result_session_id)

    usage = result.get("usage", {})

    # Record bounded recovery state plus the durable full-text call ledger.
    chat_record_id = f"chat-{uuid.uuid4().hex}"
    edge_id = uuid.uuid4().hex
    recovery_warning = ""
    try:
        with _state_lock:
            state = _load_state()
            state.setdefault("runs", []).append({
                "run_id": chat_record_id,
                "profile": profile,
                "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "message_preview": message[:120],
                "session_id": result_session_id,
                "model": result.get("model") or model_provenance["resolved_model"],
                "requested_model": model_provenance["requested_model"],
                "resolved_model": model_provenance["resolved_model"],
                "model_resolution": model_provenance["resolution_source"],
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "duration_seconds": None,
                "output_preview": reply[:500] if reply else "",
                "usage": usage,
                "type": "chat",
            })
            _save_state(state)
    except Exception as exc:
        recovery_warning = f"Profile-local recovery state could not be updated: {exc}"
        logger.warning("Chat completed without recovery-state write: %s", exc)

    from .callback import capture_session_routing

    routing = capture_session_routing(parent_agent)
    try:
        _record_dispatch_ledger(
            edge_id=edge_id,
            run_id=chat_record_id,
            origin_profile=cfg.get_active_profile_name(),
            target_profile=profile,
            dispatch_type="chat",
            delivery="sync",
            message=message,
            instructions=instructions or "",
            origin_session_id=(
                routing.get("session_key") or routing.get("session_id") or ""
            ),
            status="completed",
            output_preview=reply,
            duration_seconds=time.monotonic() - chat_started,
            usage=usage,
            requested_model=model_provenance["requested_model"],
            resolved_model=model_provenance["resolved_model"],
            model_resolution=model_provenance["resolution_source"],
        )
    except Exception as exc:
        logger.error("Chat completed but ledger insert failed: %s", exc)
        return json.dumps({
            "profile": profile,
            "session_id": result_session_id,
            "status": "completed_unrecorded",
            "reply": reply,
            "usage": usage,
            "error": f"Reply received, but the dispatch ledger write failed: {exc}",
        })

    response = {
        "profile": profile,
        "session_id": result_session_id,
        "status": "completed",
        "edge_id": edge_id,
        "reply": reply,
        "usage": usage,
        "requested_model": model_provenance["requested_model"],
        "resolved_model": model_provenance["resolved_model"],
        "model_resolution": model_provenance["resolution_source"],
    }
    if recovery_warning:
        response["warning"] = recovery_warning
    return json.dumps(response)


def handle_cancel_dispatch(args: dict, **kwargs) -> str:
    """Cancel a running dispatched task via POST /v1/runs/{run_id}/stop."""
    run_id = args.get("run_id", "").strip()
    profile = args.get("profile", "").strip()

    if not run_id:
        return _tool_error("'run_id' is required.")
    if not profile:
        return _tool_error("'profile' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    # Send the stop request to the target API server FIRST, then clear
    # local approval metadata only after the stop succeeds (or the run is
    # confirmed gone). This prevents losing recovery state if the network
    # call fails — the target would remain blocked while the origin has
    # discarded its approval details.
    listener_removed = False
    try:
        from .callback import cancel_delivery_transaction

        result, listener_removed = cancel_delivery_transaction(
            run_id,
            lambda: _post_empty(
                f"{pcfg['url']}/v1/runs/{run_id}/stop",
                pcfg["api_key"],
                timeout=15.0,
            ),
        )
    except RuntimeError as e:
        msg = str(e)
        if "404" in msg:
            # Run not found — might have already completed or been from a
            # previous gateway restart. Safe to clear local state now.
            try:
                from .callback import stop_listener
                listener_removed = stop_listener(run_id)
            except Exception:
                listener_removed = False
            try:
                from .callback import _clear_pending_approval_mem
                _clear_pending_approval_mem(run_id)
            except Exception:
                pass
            _clear_pending_approval(run_id)
            return json.dumps({
                "run_id": run_id,
                "profile": profile,
                "status": "not_found",
                "message": (
                    "Run not found on target — it may have already completed "
                    "or the gateway was restarted. Local SSE listener removed."
                ),
                "listener_removed": listener_removed,
            })
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"hermes_herald.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"Cancel dispatch {run_id} on {profile} failed: {msg}")

    # Stop succeeded — now suppress local late delivery and clear approval
    # metadata. A failed target request leaves the listener untouched.
    try:
        from .callback import _clear_pending_approval_mem
        _clear_pending_approval_mem(run_id)
    except Exception:
        pass
    _clear_pending_approval(run_id)

    return json.dumps({
        "run_id": run_id,
        "profile": profile,
        "status": result.get("status", "stopping"),
        "message": (
            "Stop signal sent. The agent will finish its current step "
            "then halt. The run.cancelled event will fire on the SSE stream."
        ),
        "listener_removed": listener_removed,
        "api_response": result,
    })


# ---------------------------------------------------------------------------
# approve_dispatch — resolve a pending approval for a dispatched run
# ---------------------------------------------------------------------------


_VALID_APPROVAL_CHOICES = ("once", "session", "always", "deny")


def _request_dispatch_approval_consent(
    *,
    profile: str,
    run_id: str,
    choice: str,
    resolve_all: bool,
    approval_data: Optional[dict],
) -> bool:
    """Require a decision from the human-owned Hermes approval surface.

    ``approve_dispatch`` is model-callable, so checking its arguments is not
    proof that a human authorized the remote command. Core elicitation routes
    to the active TUI/gateway user and fails closed without a human surface.
    """
    command = str((approval_data or {}).get("command") or "(preview unavailable)")
    reason = str((approval_data or {}).get("description") or "not provided")
    pattern_key = str((approval_data or {}).get("pattern_key") or "")
    pattern_keys = (approval_data or {}).get("pattern_keys")
    if not pattern_key and isinstance(pattern_keys, list):
        pattern_key = ", ".join(str(item) for item in pattern_keys if item)
    scope = "all pending requests" if resolve_all else "the current request"
    message = (
        f"Remote approval on profile={profile}, run_id={run_id}: "
        f"send choice={choice!r} for {scope}?\n\n"
        "Target-provided redacted command preview (untrusted data):\n"
        f"{command}\n\n"
        f"Target approval pattern/scope: {pattern_key or 'not provided'}"
    )
    description = (
        "Hermes Herald remote-command approval. Target-provided reason "
        f"(untrusted data): {reason}"
    )
    try:
        from tools.approval import request_elicitation_consent

        return request_elicitation_consent(
            message,
            description,
            surface="hermes-herald-dispatch-approval",
        ) == "accept"
    except Exception as exc:
        logger.warning(
            "hermes-herald: human approval gate failed closed for %s/%s: %s",
            profile,
            run_id,
            exc,
        )
        return False


def handle_approve_dispatch(args: dict, **kwargs) -> str:
    """Resolve a pending approval request for a dispatched run.

    Posts to the target's ``/v1/runs/{run_id}/approval`` endpoint with an
    explicit choice. Approval is scoped to the exact ``{profile, run_id}``
    pair: if we hold live pending-approval metadata for the run, the profile
    must match exactly (fail closed). The run must currently be in
    ``waiting_for_approval``. The SSE listener is left running so the run's
    normal completion is still delivered.
    """
    run_id = args.get("run_id", "").strip()
    profile = args.get("profile", "").strip()
    approval_id = args.get("approval_id", "")
    choice = args.get("choice", "")
    resolve_all = args.get("resolve_all", False)

    if not run_id:
        return _tool_error("'run_id' is required.")
    if run_id.startswith("local-"):
        return _tool_error(
            "approve_dispatch only resolves run IDs returned by dispatch_agent; "
            "it cannot resolve local terminal, cron, or /approve requests."
        )
    if not profile:
        return _tool_error("'profile' is required.")
    if not isinstance(choice, str) or not choice.strip():
        return _tool_error(
            "'choice' is required (one of: once, session, always, deny). "
            "approve_dispatch never auto-approves."
        )
    choice = choice.strip()
    if choice not in _VALID_APPROVAL_CHOICES:
        return _tool_error(
            f"'choice' must be one of {list(_VALID_APPROVAL_CHOICES)}, "
            f"got '{choice}'."
        )
    if not isinstance(approval_id, str) or not approval_id.strip():
        return _tool_error(
            "'approval_id' is required. Copy the fresh ID from the approval "
            "notice delivered to the originating session."
        )
    approval_id = approval_id.strip()
    if not isinstance(resolve_all, bool):
        return _tool_error("'resolve_all' must be a boolean when provided.")
    if resolve_all and choice != "deny":
        return _tool_error(
            "resolve_all is fail-closed for positive approvals because the "
            "human has only inspected the current command. Use resolve_all "
            "only with choice='deny', or resolve positive requests one at a time."
        )

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    # Security boundary: approval is scoped to the exact {profile, run_id}
    # pair. Check live in-memory metadata first; if not available (after
    # restart, timeout, or missed delivery), fall back to the persisted
    # state file to validate the profile. Never allow cross-profile
    # resolution even when in-memory metadata is missing.
    try:
        from .callback import get_pending_approval
        pending = get_pending_approval(run_id)
    except Exception:
        pending = None
    if pending is not None and pending.get("profile") != profile:
        return _tool_error(
            f"Approval is scoped to the exact profile/run_id pair. Run "
            f"{run_id} is pending approval on profile "
            f"'{pending.get('profile')}', not '{profile}'."
        )
    # Fall back to state file when in-memory metadata is unavailable.
    persisted_pending = None
    if pending is None:
        with _state_lock:
            state = _load_state()
            state_profile = None
            for run in state.get("runs", []):
                if run.get("run_id") == run_id:
                    state_profile = run.get("profile")
                    candidate = run.get("pending_approval")
                    if isinstance(candidate, dict):
                        persisted_pending = dict(candidate)
                    break
        if state_profile is not None and state_profile != profile:
            return _tool_error(
                f"Approval is scoped to the exact profile/run_id pair. Run "
                f"{run_id} was dispatched to profile "
                f"'{state_profile}', not '{profile}'."
            )

    approval_context = pending or persisted_pending
    if isinstance(approval_context, dict):
        expected_approval_id = str(approval_context.get("delivery_id") or "")
        if not expected_approval_id or approval_id != expected_approval_id:
            return _tool_error(
                f"'approval_id' does not match the current approval request for "
                f"{profile}/{run_id}. Use the fresh ID from its delivered notice."
            )

        try:
            from .callback import capture_session_routing

            current_route = capture_session_routing(kwargs.get("parent_agent"))
        except Exception:
            current_route = {}
        expected_ui = str(approval_context.get("origin_ui_session_id") or "")
        expected_key = str(approval_context.get("origin_session_key") or "")
        expected_session = str(approval_context.get("origin_session_id") or "")
        if expected_ui:
            owner_matches = (
                str(current_route.get("origin_ui_session_id") or "") == expected_ui
            )
        elif expected_key:
            owner_matches = str(current_route.get("session_key") or "") == expected_key
        elif expected_session:
            owner_matches = (
                str(current_route.get("session_id") or "") == expected_session
            )
        else:
            owner_matches = False
        if not owner_matches:
            return _tool_error(
                f"Approval {approval_id} belongs to the originating session "
                f"for {profile}/{run_id}; this session cannot resolve it."
            )

    # Fetch current status and require waiting_for_approval.
    try:
        status = _get_json(
            f"{pcfg['url']}/v1/runs/{run_id}",
            pcfg["api_key"],
            timeout=10.0,
        )
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"hermes_herald.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"approve_dispatch {run_id} on {profile} failed: {msg}")

    if status.get("status") == "not_found":
        # Wrong profile/run_id pair, or the run is gone. Clear any stale
        # pending metadata and fail closed.
        _clear_pending_approval(run_id)
        try:
            from .callback import _clear_pending_approval_mem
            _clear_pending_approval_mem(run_id)
        except Exception:
            pass
        return _tool_error(
            f"Run {run_id} not found on {profile}. The {profile}/{run_id} pair "
            f"is invalid — the run may belong to a different profile or has "
            f"already completed."
        )
    target_status = status.get("status")
    promoted_fifo_head = bool(
        isinstance(approval_context, dict)
        and approval_context.get("fifo_promoted") is True
    )
    if target_status != "waiting_for_approval" and not (
        target_status == "running" and promoted_fifo_head
    ):
        return _tool_error(
            f"Run {run_id} on {profile} is not awaiting approval (current "
            f"status: {target_status}). approve_dispatch only resolves a fresh "
            f"target request or a locally verified promoted FIFO head."
        )

    if not isinstance(approval_context, dict):
        return _tool_error(
            f"No owned approval notice is available for {profile}/{run_id}. "
            "Herald cannot verify an approval_id or originating session, so it "
            "will not contact the target. Cancel the run or dispatch it again."
        )

    allowed_choices = (
        approval_context.get("choices")
        if isinstance(approval_context, dict)
        else None
    )
    if isinstance(allowed_choices, list) and choice not in allowed_choices:
        return _tool_error(
            f"Choice '{choice}' is not offered by the target for "
            f"{profile}/{run_id}. Offered choices: {allowed_choices}."
        )
    if choice != "deny" and not (
        approval_context
        and isinstance(approval_context.get("command"), str)
        and approval_context.get("command", "").strip()
    ):
        return _tool_error(
            f"No redacted command preview is available for {profile}/{run_id}. "
            "Herald will not approve an unseen protected command. Use "
            "choice='deny' or cancel the run."
        )
    if choice == "always" and not (
        approval_context
        and (
            approval_context.get("pattern_key")
            or approval_context.get("pattern_keys")
        )
    ):
        return _tool_error(
            f"The target did not provide a permanent approval scope for "
            f"{profile}/{run_id}. Herald will not send choice='always'. Use "
            "choice='once' or 'session', or deny the command."
        )
    if not _request_dispatch_approval_consent(
        profile=profile,
        run_id=run_id,
        choice=choice,
        resolve_all=resolve_all,
        approval_data=approval_context,
    ):
        return _tool_error(
            f"Human confirmation was not granted for {profile}/{run_id}. "
            "No approval choice was sent to the target."
        )

    # POST the approval resolution. Bulk scope is deny-only and transactional:
    # fresh callbacks block behind the local pending lock and are not erased as
    # part of the snapshot that the target denial resolves.
    try:
        request = lambda: _post_json(
            f"{pcfg['url']}/v1/runs/{run_id}/approval",
            pcfg["api_key"],
            {"choice": choice, "all": bool(resolve_all)},
            timeout=15.0,
        )
        if resolve_all:
            from .callback import deny_all_approvals_transaction

            result = deny_all_approvals_transaction(run_id, request)
        else:
            result = request()
    except RuntimeError as e:
        msg = str(e)
        if "HTTP 404" in msg:
            _clear_pending_approval(run_id)
            try:
                from .callback import _clear_pending_approval_mem
                _clear_pending_approval_mem(run_id)
            except Exception:
                pass
            return _tool_error(f"Run {run_id} not found on {profile}. ({msg})")
        if "HTTP 409" in msg:
            # No active approval session — clear stale pending metadata.
            _clear_pending_approval(run_id)
            try:
                from .callback import _clear_pending_approval_mem
                _clear_pending_approval_mem(run_id)
            except Exception:
                pass
            return _tool_error(
                f"No active approval session for run {run_id} on {profile}. "
                f"It may have already been resolved or the run moved on. "
                f"Pending metadata cleared. ({msg})"
            )
        if "HTTP 400" in msg:
            return _tool_error(f"Invalid approval request for {run_id}: {msg}")
        if "401" in msg or "403" in msg:
            return _tool_error(
                f"Auth failed for {profile}. Check the api_key in "
                f"hermes_herald.profiles config. ({msg})"
            )
        if "Cannot reach" in msg:
            return _tool_error(
                f"Cannot reach {profile} API server at {pcfg['url']}. "
                f"Is the gateway running for that profile? ({msg})"
            )
        return _tool_error(f"approve_dispatch {run_id} on {profile} failed: {msg}")

    # The target resolves approvals FIFO. Advance the same local FIFO head that
    # the human saw, then publish the next queued request (if any). The SSE
    # approval.responded event is marked as already applied so it cannot pop a
    # second request when it arrives after this HTTP response.
    from .callback import (
        _advance_pending_approval,
        _clear_pending_approval_mem,
        _deliver_approval_required,
        get_pending_approval,
        get_pending_approval_queue,
    )
    if resolve_all:
        _clear_pending_approval_mem(run_id)
        promoted = None
        apply_local_transition = True
    else:
        current_after_post = get_pending_approval(run_id)
        apply_local_transition = bool(
            current_after_post
            and current_after_post.get("delivery_id") == approval_id
        )
        promoted = (
            _advance_pending_approval(
                run_id,
                approval_id,
                local_response=True,
            )
            if apply_local_transition
            else None
        )
    if apply_local_transition:
        if promoted:
            _update_pending_approval(
                run_id,
                promoted,
                get_pending_approval_queue(run_id),
            )
            _deliver_approval_required(
                promoted,
                str(promoted.get("origin_session_id") or ""),
                str(promoted.get("origin_session_key") or ""),
            )
        else:
            _clear_pending_approval(run_id)
    # Current core reports "running" after resolving one FIFO entry even when
    # more entries remain. Preserve the truthful local waiting state.
    _update_run_status(
        run_id,
        status=(
            "waiting_for_approval"
            if get_pending_approval(run_id)
            else "running"
        ),
    )

    return json.dumps({
        "run_id": run_id,
        "profile": profile,
        "choice": choice,
        "resolve_all": bool(resolve_all),
        "resolved": result.get("resolved"),
        "status": result.get("status", "running"),
        "message": (
            f"Approval resolved with choice='{choice}'. The run continues; "
            f"its completion will be delivered as usual."
        ),
        "api_response": result,
    })


# ---------------------------------------------------------------------------
# delegate_subagent — in-process subagent with per-call model and timeout policy
# (schema is defined at the top of the file, before ALL_SCHEMAS)
# ---------------------------------------------------------------------------


class _SubagentPolicyTimeout(TimeoutError):
    """Raised when a delegate_subagent timeout policy stops waiting on a child."""

    def __init__(self, kind: str, seconds: float):
        self.kind = kind
        self.seconds = seconds
        if kind == "stall":
            detail = f"no activity for {seconds:g}s"
        else:
            detail = f"cooperative interrupt threshold of {seconds:g}s reached"
        super().__init__(detail)


def _describe_subagent_error(exc: Exception) -> tuple[str, Optional[str]]:
    """Return a user-facing error and stable timeout kind, when applicable."""
    if isinstance(exc, _SubagentPolicyTimeout):
        return f"{exc.kind} timeout: {exc}", exc.kind
    return f"Subagent execution failed: {type(exc).__name__}: {exc}", None


def _parse_subagent_timeout_policy(
    args: dict,
    *,
    core_timeout_seconds: Optional[float],
) -> tuple[float, Optional[float]]:
    """Validate and resolve per-call delegate_subagent timeout controls."""
    if core_timeout_seconds is not None:
        raise ValueError(
            "delegate_subagent's activity-based timeout policy requires "
            "delegation.child_timeout_seconds to be 0 or unset; otherwise the "
            "core hard timeout can terminate the child before this per-call policy."
        )

    def _seconds(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{name}' must be a number of seconds.")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 30:
            raise ValueError(f"'{name}' must be at least 30 seconds.")
        return parsed

    stall = _seconds("stall_timeout_seconds", args.get("stall_timeout_seconds", 600))
    hard_value = args.get("interrupt_after_seconds")
    hard = None if hard_value is None else _seconds("interrupt_after_seconds", hard_value)
    return stall, hard


def _run_child_with_timeout_policy(
    *,
    child,
    run_child,
    stall_timeout_seconds: float,
    interrupt_after_seconds: Optional[float],
    poll_interval_seconds: float = 1.0,
):
    """Run a child under stall and cooperative interrupt thresholds."""
    done = threading.Event()
    result_box: Dict[str, Any] = {}
    outcome_lock = threading.Lock()
    activity_lock = threading.Lock()
    last_activity = [time.monotonic()]
    started_at = time.monotonic()
    original_callback = getattr(child, "tool_progress_callback", None)

    def _progress_callback(*args, **kwargs):
        with activity_lock:
            last_activity[0] = time.monotonic()
        if original_callback:
            return original_callback(*args, **kwargs)
        return None

    if original_callback and hasattr(original_callback, "_flush"):
        setattr(_progress_callback, "_flush", getattr(original_callback, "_flush"))

    def _worker():
        try:
            result = run_child()
        except Exception as exc:
            with outcome_lock:
                if not done.is_set():
                    result_box["error"] = exc
                    done.set()
        else:
            with outcome_lock:
                if not done.is_set():
                    result_box["result"] = result
                    done.set()

    child.tool_progress_callback = _progress_callback
    worker = threading.Thread(
        target=_worker,
        name="delegate-subagent-policy-worker",
        daemon=True,
    )
    worker.start()

    try:
        while not done.wait(poll_interval_seconds):
            now = time.monotonic()
            with activity_lock:
                idle_seconds = now - last_activity[0]
            elapsed_seconds = now - started_at

            timeout = None
            if interrupt_after_seconds is not None and elapsed_seconds >= interrupt_after_seconds:
                timeout = _SubagentPolicyTimeout("interrupt", interrupt_after_seconds)
            elif idle_seconds >= stall_timeout_seconds:
                timeout = _SubagentPolicyTimeout("stall", stall_timeout_seconds)

            if timeout is not None:
                with outcome_lock:
                    if done.is_set():
                        continue
                    done.set()
                try:
                    child.interrupt()
                except Exception:
                    logger.debug("delegate_subagent: child interrupt failed", exc_info=True)
                raise timeout

        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("result")
    finally:
        child.tool_progress_callback = original_callback


def _resolve_model_creds(model_name: str, parent_agent) -> dict:
    """Resolve a model name to a credential bundle for a subagent.

    Uses the same switch_model pipeline as /model. Returns a dict with
    model, provider, base_url, api_key, api_mode — all resolved from
    the model name. When the resolved provider matches the parent's,
    provider/base_url/api_key/api_mode are None so _build_child_agent
    inherits from the parent.
    """
    name = (model_name or "").strip()
    if not name:
        return {"model": None, "provider": None, "base_url": None,
                "api_key": None, "api_mode": None}

    from hermes_cli.model_switch import switch_model

    parent_provider = getattr(parent_agent, "provider", "") or ""
    parent_model = getattr(parent_agent, "model", "") or ""
    parent_base_url = getattr(parent_agent, "base_url", "") or ""
    parent_api_key = getattr(parent_agent, "api_key", "") or ""

    user_providers = {}
    custom_providers = None
    try:
        from cli import CLI_CONFIG
        user_providers = CLI_CONFIG.get("providers") or {}
        custom_providers = CLI_CONFIG.get("custom_providers")
    except Exception:
        try:
            from hermes_cli.config import load_config
            _full = load_config()
            user_providers = _full.get("providers") or {}
            custom_providers = _full.get("custom_providers")
        except Exception:
            pass

    result = switch_model(
        raw_input=name,
        current_provider=parent_provider,
        current_model=parent_model,
        current_base_url=parent_base_url,
        current_api_key=parent_api_key,
        is_global=False,
        user_providers=user_providers,
        custom_providers=custom_providers,
    )

    if not result.success:
        raise ValueError(
            result.error_message
            or f"Could not resolve model '{name}' for this subagent."
        )

    creds = {
        "model": result.new_model or parent_model,
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
    }
    # Only override credentials when the provider actually differs
    if result.target_provider and result.target_provider != parent_provider:
        creds["provider"] = result.target_provider
        creds["base_url"] = result.base_url or None
        creds["api_key"] = result.api_key or None
        creds["api_mode"] = result.api_mode or None
    return creds


def _classify_subagent_result(result: Any) -> tuple[str, Optional[str]]:
    """Map core child-result statuses onto Herald's completion contract."""
    if not isinstance(result, dict):
        return "completed", None
    child_status = str(result.get("status") or "completed").lower()
    if child_status in {"completed", "success"}:
        return "completed", None
    error = result.get("error")
    if not isinstance(error, str) or not error.strip():
        error = f"Subagent ended with status '{child_status}'."
    return "failed", error


def _apply_soul_inheritance(child, inherit_soul: bool) -> None:
    """Apply per-call SOUL identity policy before the child's first model call."""
    setattr(child, "load_soul_identity", bool(inherit_soul))
    # _build_child_agent currently returns before first prompt build, but clear
    # defensively in case core starts warming the prompt eagerly.
    setattr(child, "_cached_system_prompt", None)


_INHERITED_CONTEXT_MESSAGE_LIMIT = 20
_INHERITED_CONTEXT_CHAR_LIMIT = 12_000
_NO_TOOLSETS_SENTINEL = "__herald_model_only__"


def _compose_subagent_context(parent_agent, explicit_context, inherit_context: bool):
    """Build opt-in bounded user/assistant context without copying tool state."""
    if not inherit_context:
        return explicit_context

    messages = getattr(parent_agent, "_session_messages", None)
    if not isinstance(messages, list):
        messages = []
    rendered: List[str] = []
    for message in messages[-_INHERITED_CONTEXT_MESSAGE_LIMIT:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            rendered.append(f"Parent {role}: {content}")

    transcript = "\n".join(rendered)
    if len(transcript) > _INHERITED_CONTEXT_CHAR_LIMIT:
        transcript = transcript[-_INHERITED_CONTEXT_CHAR_LIMIT:]
    sections = []
    if transcript:
        sections.append("Recent parent conversation (opt-in, text only):\n" + transcript)
    if explicit_context:
        sections.append("Explicit task context:\n" + str(explicit_context))
    return "\n\n".join(sections) or None


def _resolve_subagent_toolsets(toolsets, inherit_toolsets: bool):
    """Map public inheritance semantics onto core's truthy-list API."""
    if toolsets is None:
        return None if inherit_toolsets else [_NO_TOOLSETS_SENTINEL]
    if not isinstance(toolsets, list) or not all(
        isinstance(item, str) and item.strip() for item in toolsets
    ):
        raise ValueError("'toolsets' must be an array of non-empty strings.")
    if not toolsets:
        # Core treats [] as omitted. A truthy unknown name intersects to [],
        # which AIAgent correctly interprets as no enabled toolsets.
        return [_NO_TOOLSETS_SENTINEL]
    return toolsets


def _enforce_subagent_toolset_policy(child, requested_toolsets) -> None:
    """Remove core-added toolsets so an explicit child subset remains exact."""
    requested = {
        str(name) for name in (requested_toolsets or [])
        if str(name) != _NO_TOOLSETS_SENTINEL
    }
    built_toolsets = [
        str(name) for name in (getattr(child, "enabled_toolsets", None) or [])
    ]
    enabled = [name for name in built_toolsets if name in requested]
    disabled = list(getattr(child, "disabled_toolsets", None) or [])
    for name in built_toolsets:
        if name not in enabled and name not in disabled:
            disabled.append(name)

    from model_tools import get_tool_definitions

    child.enabled_toolsets = enabled
    child.disabled_toolsets = disabled
    child.tools = get_tool_definitions(
        enabled_toolsets=enabled,
        disabled_toolsets=disabled,
        quiet_mode=True,
    )
    child.valid_tool_names = {
        tool["function"]["name"] for tool in (child.tools or [])
    }
    child._cached_system_prompt = None


def _resolve_parent_agent(parent_agent=None, session_id: str = ""):
    """Resolve the live commissioning agent without guessing across sessions.

    Hermes injects ``parent_agent`` for direct callers in some contexts and
    exposes it through the classic CLI plugin manager. The desktop/TUI tool
    path currently supplies only its exact ``HERMES_UI_SESSION_ID``; resolve
    that ID through the TUI's own session registry. Never scan or fall back to
    another session, because model credentials and delivery ownership belong
    to the commissioning agent.
    """
    if parent_agent is not None:
        return parent_agent

    try:
        from hermes_cli.plugins import get_plugin_manager

        plugin_manager = get_plugin_manager()
        cli = getattr(plugin_manager, "_cli_ref", None)
        cli_agent = getattr(cli, "agent", None) if cli else None
        if cli_agent is not None:
            return cli_agent
    except Exception:
        pass

    try:
        from gateway.session_context import get_session_env
        from tui_gateway import server as tui_server

        ui_session_id = str(
            get_session_env("HERMES_UI_SESSION_ID", "") or ""
        ).strip()
        durable_session_id = str(session_id or "").strip()
        sessions_lock = getattr(tui_server, "_sessions_lock", None)

        def _resolve_from_sessions(sessions):
            if ui_session_id:
                session = sessions.get(ui_session_id)
                if isinstance(session, dict) and session.get("agent") is not None:
                    return session.get("agent")
            if not durable_session_id:
                return None
            matches = []
            for session in sessions.values():
                if not isinstance(session, dict):
                    continue
                agent = session.get("agent")
                if agent is None:
                    continue
                if (
                    str(session.get("session_key") or "") == durable_session_id
                    or str(getattr(agent, "session_id", "") or "")
                    == durable_session_id
                ):
                    matches.append(agent)
            return matches[0] if len(matches) == 1 else None

        if sessions_lock is None:
            return _resolve_from_sessions(getattr(tui_server, "_sessions", {}))
        with sessions_lock:
            return _resolve_from_sessions(getattr(tui_server, "_sessions", {}))
    except Exception:
        return None


def _capture_subagent_routing(parent_agent) -> dict:
    """Capture task-local completion ownership before spawning the child thread."""
    from .callback import capture_session_routing

    routing = capture_session_routing(parent_agent)
    return {
        "session_id": str(routing.get("session_id") or ""),
        "session_key": str(routing.get("session_key") or ""),
        "origin_ui_session_id": str(routing.get("origin_ui_session_id") or ""),
    }


def handle_delegate_subagent(args: dict, **kwargs) -> str:
    """Spawn an in-process subagent with per-call model and timeout policy.

    Runs asynchronously in a background thread. Returns immediately with
    a task_id. The result re-enters the conversation as a new message via
    process_registry.completion_queue when the subagent finishes.

    Uses the core delegate_task internals (_build_child_agent +
    _run_single_child) directly, injecting a per-call model that the
    core delegate_task schema doesn't expose. No core patches required.
    """
    import uuid

    goal = args.get("goal", "")
    model_name = args.get("model", "")
    inherit_soul = args.get("inherit_soul", False) is True
    inherit_context = args.get("inherit_context", False) is True
    inherit_toolsets = args.get("inherit_toolsets", True) is not False
    parent_agent = _resolve_parent_agent(
        kwargs.get("parent_agent"), kwargs.get("session_id", "")
    )

    if not goal.strip():
        return _tool_error("'goal' is required.")
    if not parent_agent:
        return _tool_error(
            "delegate_subagent requires a parent agent context "
            "(not available in this mode)."
        )
    if not _async_delivery_supported():
        return _tool_error(
            "delegate_subagent is asynchronous, but this session cannot receive "
            "detached results. No child was started. Use llm_call when a "
            "synchronous result is required."
        )

    context = _compose_subagent_context(
        parent_agent, args.get("context"), inherit_context,
    )
    try:
        toolsets = _resolve_subagent_toolsets(
            args.get("toolsets"), inherit_toolsets,
        )
    except ValueError as e:
        return _tool_error(str(e))

    # Resolve the model to a credential bundle
    try:
        creds = _resolve_model_creds(model_name, parent_agent)
    except ValueError as e:
        return _tool_error(f"Could not resolve model '{model_name}': {e}")
    except Exception as e:
        return _tool_error(f"Model resolution failed: {type(e).__name__}: {e}")

    # Import core delegation internals
    try:
        from tools.delegate_tool import (
            _build_child_agent,
            _run_single_child,
            _get_child_timeout,
            _load_config as _load_delegation_config,
            DEFAULT_MAX_ITERATIONS,
        )
    except ImportError as e:
        return _tool_error(f"Cannot import core delegation internals: {e}")

    cfg = _load_delegation_config()
    max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    try:
        stall_timeout_seconds, interrupt_after_seconds = _parse_subagent_timeout_policy(
            args,
            core_timeout_seconds=_get_child_timeout(),
        )
    except ValueError as e:
        return _tool_error(str(e))

    # Build the child agent with the resolved model + credentials
    try:
        child = _build_child_agent(
            task_index=0,
            goal=goal,
            context=context,
            toolsets=toolsets,
            model=creds["model"],
            max_iterations=max_iter,
            task_count=1,
            parent_agent=parent_agent,
            override_provider=creds["provider"],
            override_base_url=creds["base_url"],
            override_api_key=creds["api_key"],
            override_api_mode=creds["api_mode"],
            role="leaf",
        )
        # Core subagents deliberately skip context files and persona by
        # default. Opting in here uses Hermes's native identity-only seam:
        # SOUL.md is loaded from the active profile, while project context,
        # USER.md, memory, and parent conversation history remain excluded.
        _apply_soul_inheritance(child, inherit_soul)
        if toolsets is not None:
            _enforce_subagent_toolset_policy(child, toolsets)
    except Exception as e:
        return _tool_error(f"Failed to build subagent: {type(e).__name__}: {e}")

    task_id = f"subagent-{uuid.uuid4().hex[:12]}"
    effective_model = creds["model"] or getattr(parent_agent, "model", "?")
    start_time = time.time()

    # Capture task-local routing identifiers BEFORE spawning the thread.
    routing = _capture_subagent_routing(parent_agent)
    session_id = routing["session_id"]
    session_key = routing["session_key"]
    origin_ui_session_id = routing["origin_ui_session_id"]

    # Record in state file for provenance (task 5)
    with _state_lock:
        state = _load_state()
        state.setdefault("runs", []).append({
            "run_id": task_id,
            "profile": "in-process",
            "dispatched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message_preview": goal[:120],
            "session_id": session_id,
            "model": effective_model,
            "status": "running",
            "completed_at": "",
            "duration_seconds": None,
            "output_preview": "",
            "usage": {},
            "type": "subagent",
            "stall_timeout_seconds": stall_timeout_seconds,
            "interrupt_after_seconds": interrupt_after_seconds,
        })
        _save_state(state)

    # Run in a background thread — push result to completion_queue
    def _run_in_background():
        try:
            result = _run_child_with_timeout_policy(
                child=child,
                run_child=lambda: _run_single_child(
                    task_index=0,
                    goal=goal,
                    child=child,
                    parent_agent=parent_agent,
                ),
                stall_timeout_seconds=stall_timeout_seconds,
                interrupt_after_seconds=interrupt_after_seconds,
            )
            # Update state file with terminal provenance
            elapsed = time.time() - start_time
            delivery_status, child_error = _classify_subagent_result(result)
            output_preview = ""
            if isinstance(result, dict):
                preview_value = (
                    child_error
                    if delivery_status == "failed"
                    else result.get("summary", result.get("output", ""))
                )
                output_preview = str(preview_value or "")[:500]
            _update_run_status(
                task_id,
                status=delivery_status,
                output_preview=output_preview,
                duration_seconds=elapsed,
                model=effective_model,
            )
            # Deliver to parent session via completion_queue
            try:
                from tools.process_registry import process_registry
                evt = {
                    "type": "async_delegation",
                    "delegation_id": task_id,
                    "goal": f"[delegate_subagent] {goal[:80]}",
                    "context": context,
                    "toolsets": toolsets,
                    "role": "leaf",
                    "model": effective_model,
                    "status": delivery_status,
                    "summary": (
                        result.get("summary", str(result))
                        if isinstance(result, dict) and delivery_status == "completed"
                        else (str(result) if delivery_status == "completed" else None)
                    ),
                    "error": child_error,
                    "api_calls": result.get("api_calls", 0) if isinstance(result, dict) else 0,
                    "duration_seconds": round(elapsed, 2),
                    "dispatched_at": start_time,
                    "completed_at": time.time(),
                    "session_id": session_id,
                    "session_key": session_key,
                    "origin_ui_session_id": origin_ui_session_id,
                }
                process_registry.completion_queue.put(evt)
            except ImportError:
                logger.warning("delegate_subagent: process_registry unavailable, result lost")
        except Exception as e:
            elapsed = time.time() - start_time
            error_text, timeout_kind = _describe_subagent_error(e)
            _update_run_status(
                task_id,
                status="failed",
                output_preview=error_text[:500],
                duration_seconds=elapsed,
                model=effective_model,
            )
            try:
                from tools.process_registry import process_registry
                evt = {
                    "type": "async_delegation",
                    "delegation_id": task_id,
                    "goal": f"[delegate_subagent] {goal[:80]}",
                    "context": context,
                    "toolsets": toolsets,
                    "role": "leaf",
                    "model": effective_model,
                    "status": "failed",
                    "summary": None,
                    "error": error_text,
                    "timeout_kind": timeout_kind,
                    "api_calls": 0,
                    "duration_seconds": round(elapsed, 2),
                    "dispatched_at": start_time,
                    "completed_at": time.time(),
                    "session_id": session_id,
                    "session_key": session_key,
                    "origin_ui_session_id": origin_ui_session_id,
                }
                process_registry.completion_queue.put(evt)
            except ImportError:
                pass

    thread = threading.Thread(
        target=_run_in_background,
        name=f"delegate-subagent-{task_id}",
        daemon=True,
    )
    thread.start()

    return json.dumps({
        "task_id": task_id,
        "status": "dispatched",
        "model": effective_model,
        "stall_timeout_seconds": stall_timeout_seconds,
        "interrupt_after_seconds": interrupt_after_seconds,
        "message": (
            "Subagent running in background. Result will be delivered "
            "as a new message when it completes. Continue working."
        ),
    })


# ---------------------------------------------------------------------------
# llm_call — bare LLM inference without agentic overhead
# ---------------------------------------------------------------------------


def _strip_inline_reasoning_blocks(content: str) -> str:
    """Remove inline reasoning wrappers from model-visible output."""
    import re

    return re.sub(
        r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>"
        r".*?"
        r"</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _extract_content_or_reasoning_fallback(response: Any) -> str:
    """Compatibility fallback for Hermes cores without the shared extractor."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = choices[0].message
    content = getattr(message, "content", "") or ""
    if isinstance(content, str) and content.strip():
        cleaned = _strip_inline_reasoning_blocks(content)
        if cleaned:
            return cleaned

    parts = []
    for field in ("reasoning", "reasoning_content"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    details = getattr(message, "reasoning_details", None)
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                value = detail.get("summary") or detail.get("content") or detail.get("text")
                if isinstance(value, str) and value.strip() and value.strip() not in parts:
                    parts.append(value.strip())
    return "\n\n".join(parts)


def handle_llm_call(args: dict, **kwargs) -> str:
    """Make a bare LLM inference call through the host's provider routing.

    Routes through agent.auxiliary_client.call_llm — the same function
    used by ctx.llm, context compression, vision, and web extraction.
    No agent loop, no tool schemas, no subagent overhead.
    """
    messages = args.get("messages", [])
    system_prompt = args.get("system_prompt")
    model_override = args.get("model")
    provider_override = args.get("provider")
    temperature = args.get("temperature")
    max_tokens = args.get("max_tokens")
    json_mode = args.get("json_mode", False)

    if not isinstance(messages, list) or not messages:
        return _tool_error("'messages' is required (list of {role, content} objects).")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return _tool_error(f"messages[{index}] must be an object with role and content.")
        if message.get("role") not in {"system", "user", "assistant"}:
            return _tool_error(
                f"messages[{index}].role must be system, user, or assistant."
            )
        if not isinstance(message.get("content"), str):
            return _tool_error(f"messages[{index}].content must be a string.")

    if system_prompt is not None and not isinstance(system_prompt, str):
        return _tool_error("'system_prompt' must be a string when provided.")
    if model_override is not None and not isinstance(model_override, str):
        return _tool_error("'model' must be a string when provided.")
    if provider_override is not None and not isinstance(provider_override, str):
        return _tool_error("'provider' must be a string when provided.")
    model_override = model_override.strip() if model_override is not None else None
    provider_override = provider_override.strip() if provider_override is not None else None
    if not isinstance(json_mode, bool):
        return _tool_error("'json_mode' must be a boolean when provided.")
    if system_prompt is not None and any(
        message["role"] == "system" for message in messages
    ):
        return _tool_error(
            "Use either 'system_prompt' or a system-role message, not both."
        )
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0.0 <= temperature <= 2.0
    ):
        return _tool_error("'temperature' must be a number from 0.0 to 2.0.")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        return _tool_error("'max_tokens' must be a positive integer.")

    # Build an independent message list so JSON-mode instructions never mutate
    # caller-owned dictionaries.
    full_messages: list = []
    if system_prompt is not None:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(dict(message) for message in messages)

    # response_format is honored by OpenAI-compatible routes but ignored by
    # Codex Responses and native Anthropic adapters. A system instruction gives
    # those routes the same behavioral contract; output is validated below.
    extra_body = None
    if json_mode:
        json_instruction = (
            "Return only one valid JSON object. Do not use Markdown fences or "
            "include prose before or after the object."
        )
        system_index = next(
            (index for index, message in enumerate(full_messages)
             if message["role"] == "system"),
            None,
        )
        if system_index is None:
            full_messages.insert(0, {"role": "system", "content": json_instruction})
        else:
            existing = full_messages[system_index]["content"]
            separator = "\n\n" if existing else ""
            full_messages[system_index]["content"] = (
                f"{existing}{separator}{json_instruction}"
            )
        extra_body = {"response_format": {"type": "json_object"}}

    try:
        from agent.auxiliary_client import call_llm

        try:
            from agent.auxiliary_client import extract_content_or_reasoning
        except ImportError:
            extract_content_or_reasoning = _extract_content_or_reasoning_fallback

        response = call_llm(
            task=None,
            provider=provider_override or None,
            model=model_override or None,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120.0,
            extra_body=extra_body,
        )
    except ImportError as e:
        return _tool_error(
            f"Cannot import call_llm from agent.auxiliary_client: {e}. "
            "This tool requires running inside a Hermes agent session."
        )
    except Exception as e:
        return _tool_error(f"LLM call failed: {type(e).__name__}: {e}")

    # Extract text from the response
    text = ""
    requested_provider = provider_override or ""
    configured_provider = ""
    model = model_override or ""

    # Try response.model first (most providers return it)
    response_model = (
        response.get("model")
        if isinstance(response, dict)
        else getattr(response, "model", None)
    )
    if isinstance(response_model, str) and response_model.strip():
        model = response_model.strip()

    # Preserve the profile's configured provider independently from any
    # per-call override. call_llm may still fall back internally, so neither
    # value is reported as the provider that actually served the response.
    try:
        from agent.auxiliary_client import _read_main_provider
        configured_provider = (_read_main_provider() or "").strip()
    except Exception:
        pass

    # Only claim an actual provider when the returned response explicitly
    # reports one. Core call_llm does not currently annotate fallback responses.
    response_provider = (
        response.get("provider")
        if isinstance(response, dict)
        else getattr(response, "provider", None)
    )
    provider = (
        response_provider.strip()
        if isinstance(response_provider, str) and response_provider.strip()
        else ""
    )

    # Extract text from various response shapes
    if isinstance(response, dict):
        dict_text = response.get("content")
        if dict_text is None:
            dict_text = response.get("text")
        if isinstance(dict_text, str):
            dict_text = _strip_inline_reasoning_blocks(dict_text)
        if dict_text is None or (
            isinstance(dict_text, str) and not dict_text.strip()
        ):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        dict_text = message.get("content")
                        if isinstance(dict_text, str):
                            dict_text = _strip_inline_reasoning_blocks(dict_text)
                        if dict_text is None or (
                            isinstance(dict_text, str) and not dict_text.strip()
                        ):
                            reasoning_parts = []
                            for field in ("reasoning", "reasoning_content"):
                                value = message.get(field)
                                if isinstance(value, str) and value.strip():
                                    reasoning_parts.append(value.strip())
                            details = message.get("reasoning_details")
                            if isinstance(details, list):
                                for detail in details:
                                    if isinstance(detail, dict):
                                        value = (
                                            detail.get("summary")
                                            or detail.get("content")
                                            or detail.get("text")
                                        )
                                        if isinstance(value, str) and value.strip():
                                            reasoning_parts.append(value.strip())
                            if reasoning_parts:
                                dict_text = "\n\n".join(dict.fromkeys(reasoning_parts))
                    if dict_text is None or (
                        isinstance(dict_text, str) and not dict_text.strip()
                    ):
                        dict_text = choice.get("text")
        if isinstance(dict_text, str):
            text = dict_text
        else:
            text = json.dumps(
                response if dict_text is None else dict_text,
                default=str,
            )
    elif hasattr(response, "choices"):
        # OpenAI-compatible response
        choices = response.choices
        if choices and len(choices) > 0:
            choice = choices[0]
            if hasattr(choice, "message"):
                text = extract_content_or_reasoning(response)
            elif hasattr(choice, "text"):
                text = choice.text or ""
    elif hasattr(response, "content"):
        # Anthropic-style response
        text = response.content or ""
        if isinstance(text, list):
            text = " ".join(
                block_text
                for block in text
                if isinstance((block_text := getattr(block, "text", None)), str)
            )
    else:
        text = str(response)

    if json_mode:
        candidate = text.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                candidate = "\n".join(lines[1:-1]).strip()
        try:
            parsed_json = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _tool_error(
                "json_mode requested, but the model did not return valid JSON."
            )
        if not isinstance(parsed_json, dict):
            return _tool_error(
                "json_mode requested, but the model returned JSON that was not an object."
            )
        text = json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":"))

    # Extract usage info
    usage = {}
    if isinstance(response, dict):
        u = response.get("usage")
    else:
        u = getattr(response, "usage", None)
    if u is not None:
        if hasattr(u, "model_dump"):
            usage = u.model_dump()
        elif hasattr(u, "dict"):
            usage = u.dict()
        elif isinstance(u, dict):
            usage = u
        else:
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            }

    return json.dumps({
        "text": text,
        "provider": provider,
        "requested_provider": requested_provider,
        "configured_provider": configured_provider,
        "model": model,
        "usage": usage,
    })


# ---------------------------------------------------------------------------
# ping_profile — health check tool
# ---------------------------------------------------------------------------

def handle_ping_profile(args: dict, **kwargs) -> str:
    """Check if a target Hermes profile's API server is reachable."""
    profile = args.get("profile", "").strip()
    if not profile:
        return _tool_error("'profile' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    import urllib.request
    url = f"{pcfg['url']}/v1/health"
    headers = {"Authorization": f"Bearer {pcfg['api_key']}"}
    req = urllib.request.Request(url, headers=headers, method="GET")

    start = time.time()
    try:
        with urlopen(req, timeout=5.0) as resp:
            elapsed_ms = round((time.time() - start) * 1000)
            body = resp.read().decode("utf-8", errors="replace")
            # Try to extract model info if the endpoint returns it
            model = ""
            try:
                data = json.loads(body)
                model = data.get("model", "")
            except (json.JSONDecodeError, ValueError):
                pass
            return json.dumps({
                "profile": profile,
                "status": "up",
                "response_time_ms": elapsed_ms,
                "model": model,
                "url": pcfg["url"],
            })
    except HTTPError as e:
        elapsed_ms = round((time.time() - start) * 1000)
        # 401 means the server is up but needs auth — still "up"
        status = "up" if e.code in (401, 403) else "error"
        return json.dumps({
            "profile": profile,
            "status": status,
            "response_time_ms": elapsed_ms,
            "http_code": e.code,
            "url": pcfg["url"],
        })
    except URLError as e:
        return json.dumps({
            "profile": profile,
            "status": "down",
            "error": str(e.reason),
            "url": pcfg["url"],
        })
    except Exception as e:
        return json.dumps({
            "profile": profile,
            "status": "down",
            "error": f"{type(e).__name__}: {e}",
            "url": pcfg["url"],
        })


def handle_list_profile_models(args: dict, **kwargs) -> str:
    """Return the target default and verified dispatchable model route aliases."""
    profile = args.get("profile", "").strip()
    if not profile:
        return _tool_error("'profile' is required.")

    pcfg, err = _resolve_profile(profile)
    if err:
        return _tool_error(err)

    models_url = f"{pcfg['url']}/v1/models"
    try:
        response = _get_json(models_url, pcfg["api_key"], timeout=10.0)
    except RuntimeError as e:
        return _tool_error(
            f"Cannot list dispatchable models for {profile}: {e}"
        )
    except Exception as e:
        return _tool_error(
            f"Cannot list dispatchable models for {profile}: "
            f"invalid response from {models_url} ({type(e).__name__})"
        )

    entries = response.get("data") if isinstance(response, dict) else None
    if not isinstance(entries, list):
        return _tool_error(
            f"Cannot list dispatchable models for {profile}: the target does "
            f"not expose a compatible authenticated /v1/models route listing."
        )

    primary = next(
        (
            entry for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and not entry.get("parent")
        ),
        None,
    )
    dispatchable = sorted(
        (
            {
                "alias": entry["id"].strip(),
                "resolved_model": entry["root"].strip(),
            }
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and entry["id"].strip()
            and isinstance(entry.get("root"), str)
            and entry["root"].strip()
            and isinstance(entry.get("parent"), str)
            and entry["parent"].strip()
        ),
        key=lambda item: item["alias"],
    )
    default_model = primary["id"].strip() if primary else ""
    return json.dumps({
        "profile": profile,
        "advertised_primary": {
            "model": default_model,
            "dispatchable_as_override": False,
            "is_runtime_evidence": False,
        },
        "dispatchable_models": dispatchable,
        "dispatchable_model_count": len(dispatchable),
    })

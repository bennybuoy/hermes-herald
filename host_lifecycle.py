"""Polled host-only delegation through Hermes's public lifecycle facade.

No worker threads or completion injection: each RPC stays inside its caller's
active parent context. Capability-bearing handles never leave this adapter.
"""
from __future__ import annotations

import json
import secrets
import threading
from dataclasses import asdict, is_dataclass

MAX_TASKS = 128

HOST_DELEGATE_SUBAGENT_SCHEMA = {
    "name": "delegate_subagent",
    "description": (
        "Launch or control a Hermes-owned subagent from the isolated plugin host. "
        "Returns delivery='poll': use the returned next_action on this same tool in "
        "the owning session to retrieve the final result. No automatic completion "
        "message, background polling, or restart recovery. Children inherit Hermes's "
        "safe parent tool policy and normal delegation reasoning/timeout policy. "
        "Hermes determines child role and further delegation from its depth policy; "
        "exact tool restrictions (including model-only), conversation/SOUL inheritance "
        "options and per-call reasoning/timeouts are unsupported and rejected before "
        "launch in both execution modes. Provide self-contained goal/context. "
        "Model/provider are exact lifecycle routing values, not the lenient /model "
        "switch pipeline. Call list_profile_models() without a profile to see "
        "configured local providers. Always pass BOTH provider and model when "
        "overriding across providers (including OpenRouter or unregistered models); "
        "omitted provider uses delegation config before the parent. "
        "Task handles last only for this plugin host lifetime (128 retained tasks); "
        "Hermes terminal results have one-hour retention. Retrieve promptly. "
        "Cancellation is cooperative; accepted does not mean stopped. Steering queues "
        "text only while the child accepts it."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["launch", "status", "result", "cancel", "steer"],
                       "default": "launch", "description": "Omit for launch. Other actions require task_id."},
            "goal": {"type": "string", "minLength": 1, "maxLength": 16000,
                     "description": "Required on launch only; self-contained task."},
            "context": {"type": "string", "maxLength": 32000,
                        "description": "Explicit background brief, launch only; no parent history copy."},
            "model": {"type": "string", "minLength": 1,
                      "description": "Exact model slug (catalog or unregistered), launch only. Pair with provider= when crossing providers or using OpenRouter. Omission uses Hermes delegation routing."},
            "provider": {"type": "string", "minLength": 1, "maxLength": 64,
                         "description": "Exact configured provider slug from list_profile_models(), launch only. Pin with model for an exact cross-provider/OpenRouter pair."},
            "inherit_toolsets": {"type": "boolean", "enum": [True], "default": True,
                                 "description": "Only true is supported: Hermes's safe inherited tool policy."},
            "task_id": {"type": "string", "description": "Opaque Herald ID returned by launch; controls only."},
            "reason": {"type": "string", "maxLength": 500,
                       "description": "Optional cancellation reason, cancel only."},
            "text": {"type": "string", "minLength": 1, "maxLength": 32000,
                     "description": "Required steering text, steer only."},
        },
    },
}

_ACTION_FIELDS = {
    "launch": {"goal", "context", "model", "provider", "inherit_toolsets"},
    "status": {"task_id"},
    "result": {"task_id"},
    "cancel": {"task_id", "reason"},
    "steer": {"task_id", "text"},
}
_PUBLIC_FIELDS = {
    "status": ("state", "updated_at", "diagnostic"),
    "result": ("terminal_state", "ready", "summary", "structured_payload", "started_at",
               "completed_at", "error_classification", "error_message", "usage_metadata",
               "tool_execution_summary", "result_hash"),
    "cancel": ("accepted", "already_terminal", "unknown_handle", "unsupported", "state"),
}


def lifecycle_record(value):
    """Normalize native immutable records and isolated-host wire mappings."""
    return asdict(value) if is_dataclass(value) else dict(value)


class HostLifecycleAdapter:
    """One bounded, non-persistent handle table per Herald registration.

    Never evict a task silently to make room. Hermes remains the ownership
    authority on every control call, even if a caller forges session_id kwargs.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._tasks = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate(args):
        if not isinstance(args, dict):
            raise ValueError("Arguments must be an object.")
        action = args.get("action", "launch")
        if not isinstance(action, str) or action not in _ACTION_FIELDS:
            raise ValueError("action must be launch, status, result, cancel, or steer.")
        unsupported = set(args) - _ACTION_FIELDS[action] - {"action"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported {action} options: {names}. The public lifecycle cannot preserve these policies; "
                             "no child was launched. Omit options only if those policies are not needed.")
        if "inherit_toolsets" in args and args["inherit_toolsets"] is not True:
            raise ValueError("inherit_toolsets must be true. The public lifecycle cannot preserve model-only or exact toolsets.")
        HostLifecycleAdapter._validate_fields(args, action)
        return action

    @staticmethod
    def _validate_fields(args, action):
        required = {"launch": "goal", "steer": "text"}.get(action)
        if required and (not isinstance(args.get(required), str) or not args[required].strip()):
            raise ValueError(f"{required} must be a non-empty string for {action}.")
        limits = {"goal": 16000, "context": 32000, "provider": 64, "text": 32000, "reason": 500}
        for name in ("goal", "context", "model", "provider", "text", "reason"):
            if name not in args:
                continue
            value = args[name]
            if not isinstance(value, str) or len(value) > limits.get(name, 16000):
                raise ValueError(f"{name} must be a string within its host-mode length limit.")
            if name in {"model", "provider"} and not value.strip():
                raise ValueError(f"{name} must be non-empty; omit it to use delegation routing.")
        if action != "launch" and not isinstance(args.get("task_id"), str):
            raise ValueError("task_id is required for control actions.")

    @staticmethod
    def _safe_lifecycle_error(action, exc):
        """Return an audited canned error message without echoing raw exception text.

        RPC exceptions may contain serialized capability-bearing handles, so
        raw exception text is never printed, logged, or returned. On launch, no
        handle exists yet; classify common ValueError routing/auth failures into
        actionable canned messages.
        """
        default = (
            "Hermes lifecycle call failed. Retry from the owning active session; "
            "requires the public ctx.subagent_lifecycle facade with per-launch provider support. "
            "Upgrade Hermes and check delegation routing/policy. No legacy fallback."
        )
        if action != "launch" or not isinstance(exc, ValueError):
            return default
        msg = str(exc).lower()
        if "unknown request fields" in msg and "provider" in msg:
            return default
        if any(
            token in msg
            for token in (
                "cannot resolve delegation provider",
                "has no api key",
                "no api key",
                "missing credentials",
                "without a base_url",
                "missing_api_key",
                "not authenticated",
                "no usable model provider",
            )
        ):
            return (
                "Requested delegation provider is not configured or has no usable "
                "credentials on this profile. Call list_profile_models() to inspect "
                "configured providers, or omit provider to use default delegation routing."
            )
        if "disabled" in msg:
            return (
                "Requested delegation provider is disabled in this profile's "
                "configuration. Call list_profile_models() to choose an enabled provider."
            )
        if "not found on path" in msg:
            return (
                "Requested delegation provider requires an external CLI command "
                "that was not found on PATH."
            )
        if "no active hermes parent session" in msg or "parent_session_id does not match" in msg:
            return (
                "No matching active Hermes parent session is available; "
                "retry from the owning active session."
            )
        if "unknown toolsets" in msg or "broaden parent permissions" in msg:
            return "Requested subagent toolsets were rejected by Hermes policy; no child was launched."
        return default

    def __call__(self, args, **kwargs):
        try:
            action = self._validate(args)
            session = kwargs.get("session_id")
            if not isinstance(session, str) or not session:
                raise ValueError("An owning session_id is required; no child was launched. Retry in an active Hermes session.")
        except ValueError as exc:
            return json.dumps({"status": "error", "error": str(exc), "delivery": "poll"})
        try:
            payload = self._perform(action, args, session)
        except Exception as exc:
            # RPC exceptions may contain serialized capability-bearing arguments.
            # Never print, log, or echo them; leave the task available for retry.
            return json.dumps({"status": "error", "delivery": "poll",
                               "error": self._safe_lifecycle_error(action, exc)})
        return json.dumps(payload)

    def _perform(self, action, args, session):
        service = self._ctx.subagent_lifecycle
        if action == "launch":
            with self._lock:
                if len(self._tasks) >= MAX_TASKS:
                    return {"status": "error", "delivery": "poll",
                            "error": "Herald host task capacity reached (128 retained tasks). No child launched. "
                                     "Retrieve outstanding results before restarting; no task was evicted."}
                task = "herald-host-" + secrets.token_hex(16)
                request = {name: args[name] for name in ("goal", "context", "model", "provider") if name in args}
                request.update(role="leaf", parent_session_id=session, provider=args.get("provider"))
                handle = service.launch(request)
                public_handle = lifecycle_record(handle)
                self._tasks[task] = (session, handle)
            return {"status": "dispatched", "delivery": "poll", "task_id": task,
                    "model": public_handle.get("model"), "provider": public_handle.get("provider"),
                    "next_action": {"action": "result", "task_id": task}}
        task = args["task_id"]
        with self._lock:
            record = self._tasks.get(task)
        if record is None or record[0] != session:
            return {"status": "error", "delivery": "poll",
                    "error": "Unknown task or wrong owning session. Use the launch task_id in its original session; "
                             "host restart loses handles."}
        handle = record[1]
        # This core check is necessary even when the local session label matches.
        status = lifecycle_record(service.status(handle))
        if status.get("state") == "UNKNOWN":
            return {"status": "error", "delivery": "poll", "task_id": task,
                    "error": "Hermes rejected task ownership or the result expired. Use the owning active session; "
                             "terminal results are retained for one hour, not across restart."}
        if action == "status":
            raw = status
        elif action == "steer":
            raw = {"accepted": service.steer(handle, args["text"])}
        elif action == "cancel":
            raw = lifecycle_record(service.cancel(handle, reason=args.get("reason", "Herald caller requested cancellation")))
        else:
            raw = lifecycle_record(service.result(handle))
        fields = ("accepted",) if action == "steer" else _PUBLIC_FIELDS[action]
        public = {name: raw[name] for name in fields if name in raw}
        payload = {"task_id": task, "delivery": "poll", **public}
        if action != "result" or not public.get("ready"):
            payload["next_action"] = {"action": "result", "task_id": task}
        return payload

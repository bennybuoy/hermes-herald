"""In-process completion delivery around the public, host-owned lifecycle.

Herald never receives or mutates a child agent. Handles stay registration-local;
observer threads retain the commissioning context, not an ambient profile.
"""
from __future__ import annotations

import contextvars
import copy
import threading
import time
from types import SimpleNamespace

from .host_lifecycle import HostLifecycleAdapter, HOST_DELEGATE_SUBAGENT_SCHEMA, lifecycle_record

NATIVE_DELEGATE_SUBAGENT_SCHEMA = copy.deepcopy(HOST_DELEGATE_SUBAGENT_SCHEMA)
NATIVE_DELEGATE_SUBAGENT_SCHEMA["description"] = (
    "Launch a Hermes-owned subagent asynchronously with an exact model/provider pair. "
    "Returns task_id and delivery='callback': the final result (summary or error) is "
    "auto-delivered to the spawning session. Use action=status/result/cancel/steer "
    "on this tool with task_id for manual control. Requires an active parent session "
    "and ctx.subagent_lifecycle; no legacy fallback. Runs in-process rather than durable "
    "across restart. Cancellation is cooperative interruption, not proof of exit. "
    "Children use core delegation routing, safe inherited tools, reasoning and timeout "
    "policy. Per-call tool restrictions/model-only, reasoning/timeouts, SOUL and parent "
    "history inheritance are unsupported and rejected before launch. Supply explicit "
    "goal/context. Call list_profile_models() without a profile for configured "
    "local providers; pass BOTH provider and model when overriding across "
    "providers (including OpenRouter or unregistered models). Omitted provider "
    "uses delegation config before parent routing. "
    "Task handles last for this registration lifetime (128 retained tasks); core "
    "terminal results have one-hour retention."
)


class NativeLifecycleAdapter(HostLifecycleAdapter):
    """Public launch/control with one context-bound completion observer per child."""

    def _perform(self, action, args, session):
        if action != "launch":
            return super()._perform(action, args, session)
        from . import tools
        from .callback import capture_session_routing

        if not tools._async_delivery_supported():
            return {"status": "error", "error": "This session cannot receive detached results. "
                    "No child was started; use llm_call for synchronous inference."}
        # Snapshot before construction: core may change session ContextVars while
        # building a child. Never resolve the parent by scanning private registries.
        routing = capture_session_routing(SimpleNamespace(session_id=session))
        routing["session_id"] = session
        observer_context = contextvars.copy_context()
        # Construction side effects remain inside a separate context copy.
        payload = contextvars.copy_context().run(super()._perform, action, args, session)
        if payload.get("status") != "dispatched":
            return payload
        task = payload["task_id"]
        handle = self._tasks[task][1]
        started = time.time()
        # Persistence failure must not turn an accepted launch into a retryable
        # launch error. The child already exists; expose its control task either way.
        try:
            with tools._state_lock:
                state = tools._load_state()
                state.setdefault("runs", []).append({
                    "run_id": task, "profile": "in-process", "type": "subagent",
                    "backend": "lifecycle", "dispatched_at": tools._utc_now_iso(),
                    "message_preview": args["goal"][:120], "session_id": session,
                    "model": payload.get("model"), "status": "running",
                    "completed_at": "", "duration_seconds": None,
                    "output_preview": "", "usage": {},
                })
                tools._save_state(state)
        except Exception:
            payload["persistence_warning"] = "Child launched; Herald provenance could not be saved."
        try:
            threading.Thread(
                target=observer_context.run,
                args=(self._observe, task, handle, args, routing, started, payload.get("model")),
                name=f"delegate-subagent-{task}", daemon=True,
            ).start()
        except Exception:
            payload["message"] = "Child launched but automatic observation failed; use next_action to retrieve it."
            return payload
        payload.update(delivery="callback", message="Subagent running in background. Result will be delivered "
                       "to the spawning session when it completes. Continue working.")
        return payload

    def _observe(self, task, handle, args, routing, started, model):
        from . import tools
        from tools.process_registry import process_registry

        try:
            service = self._ctx.subagent_lifecycle
            while True:
                terminal = lifecycle_record(service.wait(handle, timeout_seconds=1))
                if terminal.get("state") == "UNKNOWN":
                    raise RuntimeError("Lifecycle ownership unavailable")
                if terminal.get("completed"):
                    result = lifecycle_record(service.result(handle))
                    break
            status = "completed" if result.get("terminal_state") == "SUCCEEDED" else "failed"
            summary = result.get("summary") if status == "completed" else None
            error = result.get("error_message") if status == "failed" else None
            if status == "failed" and not error:
                error = f"Subagent ended with state {result.get('terminal_state', 'UNKNOWN')}."
            api_calls = result.get("usage_metadata", {}).get("api_calls", 0)
        except Exception:
            # RPC exceptions may serialize capability-bearing arguments.
            status, summary, error, api_calls = "failed", None, "Hermes lifecycle observation failed; " \
                "use action=result in the owning session to check the child. No replacement was launched.", 0
        elapsed = time.time() - started
        try:
            tools._update_run_status(task, status=status, output_preview=str(summary or error or "")[:500],
                                     duration_seconds=elapsed, usage={"api_calls": api_calls}, model=model)
        except Exception:
            pass  # Provenance must never suppress the completion notification.
        process_registry.completion_queue.put({
            "type": "async_delegation", "delegation_id": task,
            "goal": f"[delegate_subagent] {args['goal'][:80]}", "context": args.get("context"),
            "model": model, "status": status, "summary": summary, "error": error,
            "api_calls": api_calls, "duration_seconds": round(elapsed, 2),
            "dispatched_at": started, "completed_at": time.time(), **routing,
        })

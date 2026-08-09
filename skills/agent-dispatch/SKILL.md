---
name: agent-dispatch
description: Configure, operate, troubleshoot, and choose between Hermes Herald's cross-profile dispatch, persistent chat, model-selectable subagents, bare LLM calls, run management, and approval relay tools.
version: 1.0.0
author: Ben
license: MIT
metadata:
  hermes:
    tags: [hermes-herald, multi-agent, profiles, subagents, llm-call]
---

# Hermes Herald / Agent Dispatch

Hermes Herald adds 11 tools for communicating with named Hermes profiles, running model-selectable local subagents, making bare LLM calls, and managing asynchronous runs. The plugin manifest and config namespace are both `hermes-herald` / `hermes_herald`.

## When to Use

Load this skill when you need to:

- configure or verify a Herald target profile;
- choose between `dispatch_agent`, `dispatch_chat`, `delegate_subagent`, core `delegate_task`, and `llm_call`;
- fan independent work out to named Hermes profiles;
- continue a persistent conversation with another profile;
- select a different model for one in-process subagent;
- recover, poll, collect, or cancel dispatched runs;
- resolve or troubleshoot a protected-command approval relay;
- test or update the Hermes Herald plugin.

Do not use Herald for non-Hermes A2A peers. Use Hermes's A2A tooling for Agent Card peers. Do not use an in-process subagent for work that must survive the parent process exiting.

## Tool Selection

| Need | Tool | Important boundary |
|---|---|---|
| Independent task on a named profile | `dispatch_agent` | Async; `callback` auto-delivers, `none` is truly detached |
| Persistent multi-turn profile conversation | `dispatch_chat` | Synchronous; one saved target session per configured profile name |
| Different-model local worker | `delegate_subagent` | Async in-process daemon thread; not durable |
| Same-model local worker | core `delegate_task` | Simpler when per-call model selection is unnecessary |
| Bare classification, translation, scoring, extraction | `llm_call` | Synchronous inference; no tools or agent loop |
| Poll one dispatched run | `check_dispatch` | Use the exact `{run_id, profile}` pair |
| Poll several runs once | `collect_dispatches` | Accepts a list of `{run_id, profile}` objects |
| Audit calls or inspect topology | `dispatch_status` | Queries SQLite; full task text is opt-in |
| Cancel a target run | `cancel_dispatch` | Cooperative; also retires the local listener after target stop succeeds |
| Check target reachability | `ping_profile` | Liveness, not model-route authorization |
| Discover safe model aliases | `list_profile_models` | Call before using either dispatch tool with `model=...` |
| Deny a relayed protected command | `approve_dispatch` | Deny-only in v1; requires the fresh Herald approval ID and human confirmation |

## Installation and Configuration

Install on profiles that originate Herald calls:

```bash
hermes plugins install bennybuoy/hermes-herald --enable
```

A target-only profile needs a configured Hermes API server, not Herald itself. Restart the origin process after installation or updates, then begin a fresh session so the new schemas load.

Configure targets in the origin profile's `config.yaml`:

```yaml
hermes_herald:
  # Optional stable audit identity for custom/Docker profile homes:
  # origin_name: reviewer-orchestrator
  allow_self: false
  profiles:
    reviewer:
      url: http://127.0.0.1:8651
      api_key: ${REVIEWER_API_KEY}
      capabilities: [dispatch, chat]
      # Optional exact model_routes alias for both dispatch tools:
      # model: reviewer-fast
  # Optional; defaults to <active HERMES_HOME>/hermes-herald-runs.json
  # state_file: /custom/private/path/hermes-herald-runs.json
  # Optional durable audit DB; share only across same-filesystem origins:
  # ledger_file: /custom/private/path/hermes-herald-network.db
  # Optional dispatch_chat activity-stall default, in seconds:
  chat_timeout: 600
```

Store bearer tokens in the origin profile's `.env`, for example with `hermes config set REVIEWER_API_KEY <value>`. Do not commit or print them. The API-server key is a transport credential; model-provider credentials remain on the target profile.

For remote targets, use a trusted private network or authenticated TLS. Herald does not add TLS.

## Cross-Profile Workflow

### 1. Verify the target

```text
ping_profile(profile="reviewer")
list_profile_models(profile="reviewer")
```

`ping_profile` establishes reachability. `list_profile_models` authenticates to the target and returns exact `model_routes` aliases that may be supplied to `dispatch_agent` or `dispatch_chat`.

### 2. Write a self-contained brief

A `dispatch_agent` run does not inherit the origin conversation. Include the goal, paths/URLs/errors, constraints, output format, and verification requirements in `message`.

### 3. Dispatch independent work together

```text
dispatch_agent(profile="reviewer", message="Review ...")
dispatch_agent(profile="researcher", message="Research ...")
```

Keep every returned `{run_id, profile}` pair. Do not busy-poll immediately; continue useful work and let completion arrive asynchronously.

### 4. Verify consequential claims

A target's final summary is a self-report. Read back files, fetch URLs, inspect Git state, or check external IDs before reporting side effects as successful.

## `dispatch_agent`

`dispatch_agent` POSTs a new run to a configured target and returns a `run_id` immediately. The target run has its own session, model, skills, memory, and tools.

- An explicit `model` must be an exact target `model_routes` alias advertised by authenticated `GET /v1/models`.
- A profile-level `model` is the default for both `dispatch_agent` and `dispatch_chat`.
- If neither is present, Herald omits the model field and preserves the target's normal default.
- Auto-delivery requires a commissioning session that supports detached results and a live origin process.
- `delivery="none"` requires neither: no listener or callback is created. Preserve incoming `trace_id`/`max_hops`, set `parent_edge_id`, and pass incoming `hop_count` as `parent_hop` when forwarding detached graph work.
- After origin restart, use `dispatch_status` and `check_dispatch`; Herald does not recreate old listeners.

## Directed Routes and Audit Ledger

- Routes are caller-side outbound policy. A target in A's config grants calls through Herald from A→B only; B→A requires a separate outbound entry in B's config.
- Standard profile names are inferred from `HERMES_HOME`; set `origin_name` for custom/Docker homes so ledger attribution is not merely `custom`.
- `capabilities: [dispatch]`, `[chat]`, or `[dispatch, chat]` gates each outbound route. Missing or malformed capabilities grant nothing.
- There is no wildcard "all profiles" route.
- Self-routing needs both a matching profile entry and `allow_self: true`; otherwise it fails before network contact.
- Async graphs have no mandatory depth cap. Optional `max_hops` is a per-trace loop brake for detached or callback chains; Herald increments `parent_hop` and refuses an over-budget edge before network contact. Ping-pong consumes one hop per edge. Treat this model-carried context as an operational guardrail, not a cryptographic security boundary.
- These are enforced Herald caller-side controls, not authenticated target-side caller identity. The target bearer key is the actual API authority; a key holder can bypass Herald through direct HTTP or terminal tools.
- SQLite stores full task text, instructions, origin/target, delivery mode, trace/hop lineage, model provenance, status, and timestamps. Existing bounded JSON history is imported once; migrated rows are labelled `legacy_state_cache` because only previews survived. Configured transport keys are excluded, but secrets placed inside task text are not detectable and will be stored.
- A recovery-state write failure after a remote side effect is non-fatal: the handle and ledger edge are retained and the tool response carries a warning.
- `dispatch_status(include_messages=false, include_topology=true)` is the safe default. Set `include_messages=true` only when full briefs are needed.
- The profile-local JSON state file remains bounded recovery state. A shared `ledger_file` is supported on one filesystem; a shared `state_file` is not.

SSE recovery uses five total connection attempts: the initial connection and four reconnects with 5/10/20/40-second delays. Before reconnecting, Herald checks authoritative run status. After exhausted connection attempts it polls every two seconds, resetting the stall timer when target `updated_at` advances. Silent runs notify after 600 seconds; approval waits use a 30-minute window. Five consecutive polling transport errors end local monitoring and require manual reconciliation.

## `dispatch_chat`

Use `dispatch_chat` for synchronous dialogue where later messages depend on prior replies.

- Calls reuse one target session per configured profile name.
- `new_session=true` creates and stores a fresh target session.
- The target streams assistant deltas and tool lifecycle events; activity resets the stall timer and is surfaced to the parent UI when it exposes a progress callback.
- `stall_timeout_seconds` overrides `hermes_herald.chat_timeout`. It is an inactivity threshold, not a flat wall-clock cap; productive calls may run longer.
- `instructions` is sent as a system message for the current request; resend it when later turns still require it.
- An explicit or profile-level `model` must be an exact target `model_routes` alias verified through authenticated `/v1/models`. Omission preserves the target default.
- Persistent chat returns one final reply and blocks the current origin agent turn; it does not create a separately pollable Herald run.
- If a saved target session returns 404, retry deliberately with `new_session=true`.

## `delegate_subagent`

Use `delegate_subagent` for an isolated in-process worker that needs a per-call model:

```text
delegate_subagent(
  goal="Review the current diff. Do not edit.",
  model="gpt-5",
  context="Repo: /path; constraints: ...",
  inherit_context=true,
  inherit_soul=true,
  inherit_toolsets=false,
  toolsets=["file", "terminal"],
  stall_timeout_seconds=600,
  interrupt_after_seconds=1800
)
```

Key contracts:

- It works in the classic interactive CLI and desktop/TUI, where Herald resolves the exact commissioning UI session. Gateway and API dispatch paths do not expose the required parent-agent context and fail closed.
- It returns a `task_id` immediately and auto-delivers the final summary or error.
- It runs in a daemon background thread and dies with the parent process.
- Bare and full model names pass through Hermes's model-switch pipeline.
- `inherit_context=true` copies only a bounded recent parent user/assistant text window (20 messages, 12,000 characters). System prompts, tool calls/results, memory, and hidden state are excluded. Explicit `context` is always included.
- `inherit_soul=true` loads the active profile's full `SOUL.md` as primary identity. It remains off by default and does not inherit conversation history, `USER.md`, memory, or project context files.
- `inherit_toolsets=true` is the default when `toolsets` is omitted. Set it false plus `toolsets=[]` for a model-only child, or request an explicit subset; requested tools are intersected with parent capabilities and blocked child surfaces remain unavailable. An explicit list is exact: parent MCP toolsets are stripped unless named in that list, overriding core's global MCP-preservation default for this child.
- `stall_timeout_seconds` is activity-based and defaults to 600 seconds.
- `interrupt_after_seconds` is an optional wall-clock threshold that requests cooperative interruption, stops waiting, and reports timeout. It cannot immediately terminate a blocking provider or tool call.
- Hermes core's global `delegation.child_timeout_seconds` must be unset or `0`; otherwise Herald refuses to launch because core could preempt the per-call policy.

## `llm_call`

Use `llm_call` when you need only inference, not an agent loop:

- `messages` must contain at least one `{role, content}` item.
- Use either `system_prompt` or a system-role message, not both.
- `model` and `provider` are optional routing requests. A model-only request is
  pinned to the active provider; it does not infer a different endpoint from a
  vendor-prefixed model name.
- Model overrides pass through Hermes's `/model` resolver and must match the
  selected provider's advertised authenticated inventory before the inference
  request. The inventory check may itself query that provider. Human forms such
  as `gpt5.6sol` normalize to `gpt-5.6-sol`; a family
  form such as `gpt-5.6` reuses the active advertised family variant. Unknown
  models fail before a provider fallback can run.
- Use exact provider IDs. Hermes maps bare `openai` to OpenRouter, so Herald
  refuses that ambiguous alias. Use `openai-codex` for ChatGPT Codex OAuth or
  `openrouter` only when OpenRouter is intentional.
- `max_tokens` is a best-effort provider-dependent hint, not a universal cap.
- `json_mode=true` adds a JSON-only instruction, requests structured output where supported, and rejects output that is not a valid JSON object.
- `provider` is reported only when the response identifies the serving provider. `requested_provider` and `configured_provider` describe routing intent, not necessarily the fallback route that served the response.
- `requested_model`, `resolved_provider`, and `resolved_model` expose route
  resolution separately from the response-reported `model`.

## Run Recovery and Cancellation

- `dispatch_status()` lists durable ledger calls and credential-free configured/observed topology, including prior sessions.
- The target API remains authoritative for live status; follow a recovered record with `check_dispatch(run_id, profile)`.
- `cancel_dispatch` sends the target's stop request first. Cancellation is cooperative and may take a few seconds while the target finishes its current step.
- Never guess a run's profile. The same `{run_id, profile}` pair is required for polling, approval, and cancellation.

## Approval Relay

For autonomous meshes, prefer the target's `smart` approval mode. Use `off` only for deliberately trusted or sandboxed targets. Manual relay is an advanced model-mediated fallback.

When a target emits `approval.request`, Herald queues the notice to the originating session. The model may propose `approve_dispatch(choice="deny")`, but Herald then calls Hermes's human-owned elicitation surface and sends no denial unless the human accepts.

Safety and scope:

- The displayed `approval_id` is a fresh Herald delivery nonce bound to the origin session and `{profile, run_id}`. It is not a target-side approval-entry selector.
- Hermes's current run-approval endpoint resolves pending commands FIFO. Herald shows and advances one local FIFO head at a time.
- Stale nonces, wrong profiles, and other sessions fail before the target is contacted.
- Herald v1 accepts only `choice="deny"`. Positive choices (`once`, `session`, `always`) fail before profile resolution or network contact because current Hermes core does not expose an immutable target approval ID.
- Target-supplied command and description text is untrusted display data.
- If there is no human surface, confirmation fails closed.
- `resolve_all=true` is deny-only and transactionally applies to the current pending snapshot.

## Troubleshooting

### Tool or skill missing

1. Run `hermes plugins list` and confirm `hermes-herald` is enabled.
2. Restart the process that loads the plugin.
3. Start a fresh session; existing sessions retain old schemas.
4. The bundled skill is opt-in and namespaced: `skill_view("hermes-herald:agent-dispatch")`.

### Target unreachable

1. Run `ping_profile`.
2. Check the configured URL and target gateway process.
3. Separate connection failure from 401/403 authentication failure.
4. Confirm the target API server is enabled and bound to the intended interface.

### Completion does not arrive

1. Confirm the origin process that commissioned the run is still alive.
2. Run `check_dispatch` with the original profile.
3. Inspect `dispatch_status` for persisted provenance.
4. Reconcile the existing run; do not redispatch blindly.

### Persistent chat rejects a model alias

Call `list_profile_models(profile=...)` and use one of the exact advertised route aliases. The target's primary identity or an arbitrary provider/model slug is not sufficient unless it is also declared as a route alias. Omit `model` to use the target default.

### Approval notice does not arrive

Use `check_dispatch`. If the target reports `waiting_for_approval`, the origin listener may have been lost or the manual relay may be delayed. Do not approve an unseen command. Deny or cancel when safe recovery is unavailable.

## Verification

From the plugin repository:

```bash
cd tests
HERMES_HERALD_PLUGIN_DIR=../ HERMES_SOURCE_DIR=/path/to/hermes-agent \
  python3 -m pytest -v
```

Also verify:

- `plugin.yaml` advertises the same 11 tools registered in `__init__.py`;
- the plugin registers the bundled `agent-dispatch` skill;
- `plugin.yaml`, README, release card, and hero all say v1.0.0;
- the working tree is understood and `git diff --check` passes;
- running Hermes processes were restarted before live verification.

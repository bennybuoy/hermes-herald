---
name: agent-dispatch
description: Configure, operate, troubleshoot, and choose between Hermes Herald's cross-profile dispatch, persistent chat, model-selectable subagents, bare LLM calls, run management, and approval relay tools.
version: 1.2.0
author: Ben
license: MIT
metadata:
  hermes:
    tags: [hermes-herald, multi-agent, profiles, subagents, llm-call]
---

# Hermes Herald / Agent Dispatch

Hermes Herald adds 12 tools for communicating with named Hermes profiles, running model-selectable local subagents, making bare LLM calls, and managing asynchronous runs. The plugin manifest and config namespace are both `hermes-herald` / `hermes_herald`.

## Supported execution boundary

**Full Herald delegation requires `plugins.isolation: in_process`.** This includes
automatic completion delivery, exact child tool restrictions, and per-call
conversation/SOUL/reasoning/timeout controls. Isolated `host` execution has an
experimental polling-only adapter, not equivalent support. Never launch a hosted
child expecting an automatic completion message, and never drop requested
restrictions just to make a launch succeed.

Plugin-process isolation is distinct from shared-gateway/multi-profile scoping.
Herald supports separate profile routes, credentials, state and ledgers inside
the shared gateway; that does not provide an OS/filesystem sandbox. Do not change
a user's isolation setting or restart their gateway without authorization.

Version 1.2.0 also supports current Hermes YAML dependencies, resolves local
models in the active profile, pins explicitly selected child runtime bundles,
and retains the original durable parent identity for completion delivery.

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
| Different-model local worker | `delegate_subagent` | Supported in `in_process` with auto-delivery; experimental host subset is poll-only; neither durable |
| Same-model local worker | core `delegate_task` | Simpler when per-call model selection is unnecessary |
| Bare classification, translation, scoring, extraction | `llm_call` | Synchronous inference; no tools or agent loop |
| Direct endpoint research | `llm_direct` | Separate opt-in; configured endpoint only; bypasses host routing |
| Poll one dispatched run | `check_dispatch` | Use the exact `{run_id, profile}` pair |
| Poll several runs once | `collect_dispatches` | Accepts a list of `{run_id, profile}` objects |
| Audit calls or inspect topology | `dispatch_status` | Queries SQLite; full task text is opt-in |
| Cancel a target run | `cancel_dispatch` | Cooperative; also retires the local listener after target stop succeeds |
| Check target reachability | `ping_profile` | Liveness, not model-route authorization |
| Discover safe model routes | `list_profile_models` | Omit `profile` for local `llm_call`; name a profile for remote dispatch |
| Deny a relayed protected command | `approve_dispatch` | Deny-only in v1; requires the fresh Herald approval ID and human confirmation |

## Installation and Configuration

Install on profiles that originate Herald calls:

```bash
hermes plugins install bennybuoy/hermes-herald --enable
```

A target-only profile needs a configured Hermes API server, not Herald itself. Restart the origin process after installation or updates, then begin a fresh session so the new schemas load.

For the supported local delegation workflow, merge this into the origin's
configuration, retaining existing plugin enablement and trust settings:

```yaml
plugins:
  isolation: in_process
```

This is a profile-wide plugin trust decision, not a per-tool switch. Use only
trusted plugins in-process. An isolation-required deployment must retain its
policy and accept the experimental subset or avoid local Herald delegation.

Configure targets in the origin profile's `config.yaml`:

```yaml
hermes_herald:
  # Optional stable audit identity for custom/Docker profile homes:
  # origin_name: reviewer-orchestrator
  allow_self: false
  profiles:
    reviewer:
      url: http://127.0.0.1:8642/p/reviewer
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

For a shared gateway, use the default listener's actual port plus
`/p/<profile>` (without `/v1`); Herald preserves the prefix when appending API
paths. Store the named target's inbound key as its `API_SERVER_KEY`, and use
that same key on the origin route. The default listener's key does not
authenticate a named prefix. Do not enable a second API listener or start a
second gateway for a multiplexed secondary profile. A standalone/legacy
target instead uses its own unprefixed base URL, such as `http://127.0.0.1:8651`.

Dispatchable models are **not** configured in that origin block. They are
aliases on the **API-listener owner**. At tested shared-gateway revision
`1f2b9e08644c6de4a5b5c50d0e4676a19c5577c5`, this means the default gateway's
configuration, not a separate alias table in each secondary profile. For a
standalone target, use that target's configuration:

```yaml
# Listener-owning profile's config.yaml
platforms:
  api_server:
    extra:
      model_routes:
        reviewer-fast:
          provider: ollama-cloud
          model: glm-5.2
        reviewer-reasoning:
          provider: openai-codex
          model: gpt-5.6-sol
```

`list_profile_models(profile="reviewer")` authenticates to the configured base URL plus `/v1/models`: URL-selected profile identity plus the listener's aliases only. Herald does not list the target's full provider catalog. Verify that exact listing; a short listing means check aliases on the listener owner and restart that gateway when safe. Pin `provider` on each alias. Origin `hermes_herald.profiles.reviewer.model` may name one of those aliases as a default; it cannot invent new ones. Listener-wide aliases are not per-profile allowlists. See the [shared-gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/multi-profile-gateways).

Copy an exact alias from `pass_as_model` into `model`. If the listing advertises
a provider, pass that provider; it must match a pinned alias. If provider metadata
is absent, verify the listener's alias with its operator rather than inventing a
route. Both dispatch tools accept `reasoning_effort` for the target request; the
target's policy can ignore, clamp or drop it.

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

### Deliberate retry contract (remote runs only)

- For a retryable task, optionally supply `idempotency_key` **before the first
  call**. It must be 1–255 visible ASCII characters, no spaces; never place secrets
  in it. Otherwise Herald generates a key and returns it on success or ambiguous
  failure with `retry_instruction`. Keep that key and every original argument.
  If the entire response is lost before you see a generated key, you have no
  caller-known identity; a pre-supplied key avoids that gap.
- To deliberately retry the same work, send the exact key and all original
  arguments again from the owning origin/durable session. Do not change the
  brief, instructions, delivery, model/provider, reasoning, trace or hop fields.
  Herald persists identity and the exact POST body (including generated edge/trace
  IDs) before transport, and surfaces the original `run_id`. `/new`, a changed
  URL/configured model, or a rotated bearer key fails closed. Sessionless detached
  callers are origin-scoped. The ledger stores a one-way credential fingerprint,
  not the bearer token itself.
- **Omitting the key means NEW intentional work**, even with identical messages.
  Never retry an ambiguous failure by dropping or regenerating its key.
- Concurrent same-key calls get `idempotency_in_progress`: wait, then reuse the
  same key unchanged. An abandoned attempt's lease lasts up to 120 seconds.
  Already recorded acceptance returns the handle without another POST, ledger
  edge, listener, or callback. Retry does not transfer ownership or restore an
  old process's listener; poll with `check_dispatch` after restart.
- Ambiguous replay requires the exact target's authenticated `/v1/capabilities`
  to advertise `features.runs_idempotency` with `supported=true`, `durable=true`,
  and valid `retention_seconds` at original submission and retry. The deadline
  is original reservation time + advertised retention − 60 seconds; retries
  never extend it. The inspected API advertises 24 hours, not a universal promise.
  Older/non-durable targets still allow initial submission with a warning, but
  unsafe or expired replay returns `idempotency_unsafe_replay` without a POST.
- A target HTTP 409 is `idempotency_server_conflict`; local changed-request,
  routing, or ownership conflicts are `idempotency_conflict`. Reconcile known
  work with the target/operator. Do not change the key to bypass a conflict.
- This is not unconditional exactly-once execution: the target must honor its
  advertised persistence/namespace/retention. Database replacement/deletion can
  break that guarantee. Keep Herald's ledger, which retains identity tombstones
  and original bodies without automatic pruning; deleting them makes used keys
  appear new. A crash during local bookkeeping can leave audit/cache/monitoring
  incomplete; Herald returns the known handle with a warning, not duplicate
  delivery. Task text remains persisted private data; do not include secrets.

```text
# First submission, with identity retained by the caller:
dispatch_agent(profile="reviewer", message="Review ...", delivery="none",
               idempotency_key="review-release-01")
# Deliberate retry ONLY: exactly the same key and original arguments:
dispatch_agent(profile="reviewer", message="Review ...", delivery="none",
               idempotency_key="review-release-01")
```

This does not add retry identity to `dispatch_chat`, `delegate_subagent`,
approval/cancellation, or inference tools.

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
- On `chat_timeout`, keep the returned `profile` and `session_id`: the remote turn may still be running. Inspect that target session before sending more work; never automatically resend the chat. Socket waits respect the configured inactivity window without a separate 60-second cap.

Remote callback and approval notices use `API calls: ?` when no call count is
available. Run token usage is preserved separately; do not infer call counts
from tokens.

## `delegate_subagent`

### Experimental host mode: poll, do not await a completion message

Minimum tested Hermes prerequisite: transport-fix [PR #133762](https://github.com/NousResearch/hermes-agent/pull/133762), exact candidate
**`c2a785b4e3a59ba49dc564e23e5bffab89dd9cce`** (or a later upstream revision
preserving its fix and lifecycle contract), not a released-version claim.
Herald adapts to the existing public `ctx.subagent_lifecycle`, without core edits.
CI now pins tested revision **`c72f29e0a3608e2276e311c1c9a891aedf2acb17`**.
Passing this lane proves the polling adapter, not automatic host completion.
Inside `plugins.isolation: host`, the same tool has a smaller schema:

```text
delegate_subagent(action="launch", goal="Review ...", context="Self-contained brief ...",
                  provider="openai-codex", model="gpt-5")
delegate_subagent(action="result", task_id="<exact returned task_id>")
delegate_subagent(action="status", task_id="<exact returned task_id>")
delegate_subagent(action="steer", task_id="<exact returned task_id>", text="Updated brief ...")
delegate_subagent(action="cancel", task_id="<exact returned task_id>", reason="Stop ...")
```

- `action` defaults to `launch`. Keep the opaque Herald `task_id` and follow
  returned `next_action` on `delegate_subagent` in the **owning active session**.
  `delivery="poll"` means **no automatic completion**. `ready=false` means not
  finished; do useful work before retrying. Final summaries/errors and usage
  metadata are preserved on `result`; do not use `check_dispatch` for these IDs.
- Exact `provider`/`model` values are passed unchanged, **not** through `/model`
  name matching. Pin both for an exact pair. Omitted provider uses delegation
  config before the parent; omitted model uses normal delegation routing.
- Omit tool policy options, or use only `inherit_toolsets=true`. Hermes controls
  its safe inherited parent tools, including normal MCP inheritance. **All
  explicit `toolsets` lists (including `[]`) and `inherit_toolsets=false` fail
  before launch**, because the current API cannot preserve Herald's exact
  tool-subset/model-only guarantees. Never retry a rejected restricted task by
  dropping its restrictions unless the user explicitly approves that change.
- `inherit_context`, `inherit_soul`, `reasoning_effort`, per-call stall/interrupt
  or other timeout fields are unsupported whenever supplied (even false/null).
  They fail before construction, not silently ignored. Use explicit `context`
  and Hermes delegation defaults, or choose `in_process` for those policies.
  No role, working-directory or per-tool-block overrides are supported.
- Cancel is cooperative; `accepted` is not proof the child stopped. Steering
  returns whether text was accepted; use status/result to observe termination.
- Core ownership is checked on every action. Capability tokens stay private to
  the adapter. No background RPC is attempted after tool return, and no
  consent-gated `ctx.inject_message` workaround is enabled.
- IDs are memory-only and bounded to **128 tasks per registration**; capacity
  refusal starts no child and evicts no prior results. Hermes retains terminal
  results for **one hour**. Poll promptly. Host restart/unload loses IDs but may
  leave Hermes-owned children running: retrieve/cancel before restart. No
  restart recovery or dispatch-ledger persistence; use `dispatch_agent` for
  durable target work.

### Supported in-process mode

Use `delegate_subagent` for an in-process worker that needs a per-call model:

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
- Delivery is bound to the original durable parent session, captured before
  child construction. `/new` must not redirect an old completion to a replacement
  conversation. Keep that session and process alive; this is not durable recovery.
- It runs in a daemon background thread and dies with the parent process.
- Bare and full model names pass through Hermes's model-switch pipeline.
- Resolution reads the active profile's configuration, not the launch profile's
  global CLI snapshot. Unchanged transport bundles inherit the live parent route.
  Changed provider/endpoint/key/API-mode bundles are pinned on the child, including
  same-provider changes; startup pool leasing and per-turn env refresh cannot
  overwrite that explicit selection. Parent and normal inherited-child policies
  stay unchanged. Never pass credentials in prompts.
- `inherit_context=true` copies only a bounded recent parent user/assistant text window (20 messages, 12,000 characters). System prompts, tool calls/results, memory, and hidden state are excluded. Explicit `context` is always included.
- `inherit_soul=true` loads the active profile's full `SOUL.md` as primary identity. It remains off by default and does not inherit conversation history, `USER.md`, memory, or project context files.
- `inherit_toolsets=true` is the default when `toolsets` is omitted. Set it false plus `toolsets=[]` for a model-only child, or request an explicit subset; requested tools are intersected with parent capabilities and blocked child surfaces remain unavailable. An explicit list is exact: parent MCP toolsets are stripped unless named in that list, overriding core's global MCP-preservation default for this child.
- `stall_timeout_seconds` is activity-based and defaults to 600 seconds.
- `interrupt_after_seconds` is an optional wall-clock threshold that requests cooperative interruption, stops waiting, and reports timeout. It cannot immediately terminate a blocking provider or tool call.
- Hermes core's global `delegation.child_timeout_seconds` must be unset or `0`; otherwise Herald refuses to launch because core could preempt the per-call policy.

`reasoning_effort` optionally sets the child's thinking budget for this call:
`minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`, or `none`. Omission
keeps Hermes's normal delegation-then-parent resolution. The host may still clamp,
drop or substitute the budget; this is not a strict every-request guarantee.

## `llm_call`

Use `llm_call` when you need only inference, not an agent loop:

- `messages` must contain at least one `{role, content}` item.
- Use either `system_prompt` or a system-role message, not both.
- Call `list_profile_models()` before choosing a route. Its local mode returns
  the calling profile's configured default plus every provider explicitly
  configured by the user as exact provider/model slugs. Virtual MoA routes,
  ambient or auto-discovered credentials, and fallback-only routes are not
  routing authority.
- With no routing arguments, Hermes uses the host's active configured route.
- Prefer that configured default. Do not choose another provider merely because
  it advertises the same model; override only when the task or user explicitly
  calls for that endpoint.
- `model` and `provider` are optional only as a pair: supply both by copying one
  exact `available_routes` entry, or omit both.
- Herald checks that pair against configured discovery and passes it unchanged
  to Hermes's public `ctx.llm.complete()` facade. Hermes owns credentials,
  transport, provider resolution, timeout behavior, and fallback policy.
- Explicit selection also requires the host's
  `plugins.entries.hermes-herald.llm` trust policy to permit provider and model
  overrides. Missing trust configuration fails closed.
- `max_tokens` is a best-effort provider-dependent hint, not a universal cap.
- `json_mode=true` adds a JSON-only instruction and rejects output that is not a valid JSON object.
- A route failure is returned noisily. Herald itself does not retry the call;
  host behavior remains governed by Hermes.
- Results report the requested pair separately from the provider/model returned
  by the host facade.

For example, merge a least-privilege allowlist into the origin configuration:

```yaml
plugins:
  entries:
    hermes-herald:
      llm:
        allow_provider_override: true
        allowed_providers: [openai-codex]
        allow_model_override: true
        allowed_models: [gpt-5.6-sol]
```

These are illustrative models, not discovery authority. Select a pair returned by
the calling profile's `list_profile_models()` and allow only that intended route.

## `llm_direct`

Use this separate research lane only when direct OpenAI-compatible parameter
control is intended. It bypasses Hermes's host LLM routing and is disabled until
explicitly enabled in the origin configuration:

```yaml
hermes_herald:
  llm_direct:
    enabled: true
    default_endpoint: lab
    endpoints:
      lab:
        base_url: https://llm.example.com/v1
        api_key: ${LAB_API_KEY}
        default_model: lab-model
        allowed_models: [lab-model]
```

```text
llm_direct(messages=[{"role": "user", "content": "Reply with OK"}],
           endpoint="lab", temperature=0, seed=7, extra_body={"top_k": 40})
```

- Endpoint URLs and credentials belong in configuration, never tool arguments.
  API keys must be environment references such as `${LAB_API_KEY}`.
- Choose a configured endpoint and a model allowed by that endpoint's policy.
- Use named parameters for reserved OpenAI fields; `extra_body` accepts only
  vendor-specific additions. Redirects are refused to protect bearer credentials.
- Treat provider output as untrusted. A direct call is not an agent and has no tools.

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
2. Check the configured URL and the listener-owning gateway process.
3. Separate connection failure from 401/403 authentication failure.
4. Confirm the listener is enabled and bound to the intended interface. For a
   shared gateway, check `/p/<profile>` routing and the named profile's key;
   do not enable a second API listener on that profile.

### Completion does not arrive

First distinguish the handle type:

1. A remote `dispatch_agent` run uses `{profile, run_id}`. Confirm the origin
   process is alive, query `check_dispatch`, and inspect `dispatch_status` for
   persisted provenance. `delivery="none"` never schedules a callback.
2. An in-process `delegate_subagent` child uses a local `task_id`. Keep the
   original durable parent session alive; `/new` and process exit are not recovery.
   Do not send that ID to `check_dispatch`.
3. An experimental host child returns `delivery="poll"`. No notification will
   arrive: call `delegate_subagent(action="result", task_id=...)` from the owning
   active session. Restart loses its memory-only handle; terminal results expire.

Reconcile existing work before redispatching. If automatic completion or exact
tool restrictions are essential, use supported in-process execution after the
user accepts that profile-wide trust setting; do not weaken the task silently.

### Persistent chat rejects a model alias, or the listing is missing a model

Call `list_profile_models(profile=...)` and use one of the exact advertised route aliases. The target's primary identity or an arbitrary provider/model slug is not sufficient unless it is also declared as a route alias. Omit `model` to use the target default.

If the listing is shorter than expected, check the **listener owner's**
`platforms.api_server.extra.model_routes`, restart that gateway when safe, and
list again. On the tested shared gateway this is the default profile; on a
standalone gateway it is the target. Origin `hermes_herald.profiles` does not
define the catalog.

### Approval notice does not arrive

Use `check_dispatch`. If the target reports `waiting_for_approval`, the origin listener may have been lost or the manual relay may be delayed. Do not approve an unseen command. Deny or cancel when safe recovery is unavailable.

## Verification

Use the canonical Hermes runner and isolated environment described in the
repository's README and CI workflow, never a live profile's state. CI pins
legacy `9fc80ac70f6b97e36556c33bc4895b75558d4136` (Python 3.11), shared-gateway
`1f2b9e08644c6de4a5b5c50d0e4676a19c5577c5` (Python 3.14), and experimental
host-candidate `c72f29e0a3608e2276e311c1c9a891aedf2acb17` (Python 3.14).
The manifest's general Hermes floor is not optional-host certification.

With the host-candidate checkout at `hermes-agent/` and its isolated environment
built as in CI, run the following from the plugin repository. For legacy or
shared-gateway, use the README/workflow's file selection that omits only
`test_host_subagent_lifecycle.py`; those pins cannot run the real host transport
tests. Do not present that omission as a host-mode pass.

```bash
scripts/ci-python scripts/check_release_contract.py
HERMES_PYTHON="$PWD/scripts/ci-python" \
  bash hermes-agent/scripts/run_tests.sh "$PWD/tests" -j 2 --file-retries 0 \
  -p no:cacheprovider -o addopts= --confcutdir="$PWD/tests" --rootdir="$PWD/tests" -q
```

Also verify:

- `plugin.yaml` advertises the same 12 tools registered in `__init__.py`;
- the plugin registers the bundled `agent-dispatch` skill;
- the manifest, skill frontmatter, README badge and any version-bearing release
  artwork agree on the version;
- the working tree is understood and `git diff --check` passes;
- live verification, if separately authorized, uses a fresh process/session;
  do not restart a running gateway merely to run these synthetic tests.

<img src="assets/hero-banner.png" alt="Hermes Herald — Multi-Agent Dispatch for Hermes Agent" width="100%">

# Hermes Herald

<p>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Hermes%20Agent-compatible-8B5CF6" alt="Hermes Agent"></a>
  <img src="https://img.shields.io/badge/version-1.2.0-22C55E" alt="Version 1.2.0">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/tools-12-orange" alt="12 Tools">
</p>

> **Your agents should not just talk to you. They should be able to talk to each other.**

Hermes Herald turns one Hermes Agent into a **conductor of specialist agents and models**. It adds three capabilities that Hermes users otherwise have to wire together themselves:

| Herald adds | Why you would want it |
|---|---|
| **📡 A network of named Hermes agents** | Ask another profile a question, continue the conversation later, or send independent work to several specialists in parallel. Each target keeps its own identity, memory, skills, credentials, tools, and runtime. |
| **🔀 A model switchboard for subagents** | Spawn a local child with the model, context, SOUL identity, and tool access appropriate to this one task—without changing the parent agent. |
| **🧠 A lightweight inference lane** | Use a model for classification, extraction, translation, scoring, or JSON generation without paying for an agent loop and a large tool schema. |

These are independent. You can use Herald only for `llm_call`, only for model-selectable subagents, or build a multi-host network of persistent specialist profiles.

> [!IMPORTANT]
> **Herald's full delegation workflow requires `plugins.isolation: in_process`.**
> Isolated-host mode does not support automatic completion delivery to the parent
> conversation. An experimental adapter offers launch, status, result retrieval,
> cancellation and steering, but requires explicit polling and is not equivalent
> to the supported workflow. This is **plugin-process isolation**, not shared-gateway
> or multi-profile isolation: Herald supports profile-scoped use in a shared gateway.

## What's new in 1.2.0

- Compatibility with current Hermes YAML dependencies, without requiring PyYAML
  on hosts that use `ruamel.yaml`.
- Profile-scoped model resolution and preservation of explicitly selected child
  credentials/endpoints across startup pool leasing and first-turn env refresh.
- Completion events retain the original durable parent-session identity, captured
  before child construction, so Hermes can reject stale-session delivery.
- Shared-gateway setup guidance and a pinned compatibility CI matrix.
- An **experimental, polling-only** adapter for Hermes's isolated plugin host.
  No Hermes core changes or live configuration changes.

## Why Herald?

A single giant agent prompt is convenient until different jobs need different identities, memories, credentials, tools, costs, or trust boundaries.

Herald lets you build systems such as:

- a coordinator that sends code, security, and documentation reviews to separate specialists in parallel;
- a teacher agent that maintains an ongoing conversation with a curriculum expert;
- a local reasoning agent that delegates mechanical inspection to a faster model;
- several Hermes installations connected over Tailscale, each retaining its own environment and credentials;
- an autonomous workflow whose calls, model choices, lineage, status, and results remain queryable after the originating turn;
- a high-volume classifier that uses bare inference rather than loading terminal, browser, and file tools for every item.

Herald is not a replacement for Hermes profiles or provider configuration. It is the **control plane between them**.

---

## Choose the right execution mode

This is the most important distinction in Herald:

| What you need | Use | Returns | Context and lifetime | Model choice |
|---|---|---|---|---|
| A conversation with a named specialist | `dispatch_chat` | Final reply; blocks the current agent turn | Reuses a persistent target session; streamed activity resets the stall timer | Target default or a verified target `model_routes` alias |
| Independent work on another Hermes profile | `dispatch_agent` | `run_id` immediately | Fresh target session; monitored by SSE/polling or explicitly detached | Target default or a verified target `model_routes` alias |
| A local child agent with a different model | `delegate_subagent` | `task_id` immediately; automatic completion in supported `in_process` mode | Non-durable; explicit conversation, identity and tool inheritance controls | Hermes model-switch resolution; see the experimental host subset below |
| Classification, extraction, translation, scoring | `llm_call` | Text synchronously | No agent loop and no tools | Hermes provider/model routing |
| Direct endpoint experiments | `llm_direct` | Text synchronously | No agent loop; separately opt-in | Pre-configured endpoint and allowed model; bypasses host LLM routing |

**Use `dispatch_chat` when your next decision depends on the target’s reply.**

**Use `dispatch_agent` when the work is self-contained and the origin should remain available.**

```python
# Continue a real conversation with a named profile.
dispatch_chat(profile="tutor", message="Explain decorators with one example")
dispatch_chat(profile="tutor", message="Now adapt that example to cache results")

# Fan independent work out without blocking the origin.
dispatch_agent(profile="reviewer", message="Review the current diff. Do not edit.")
dispatch_agent(profile="researcher", message="Find authoritative sources for the design.")

# In plugins.isolation: in_process, choose a local model and inheritance.
delegate_subagent(
    goal="Classify the test failures by likely root cause",
    model="gpt-5",
    inherit_context=True,
    inherit_soul=True,
    toolsets=[],
)

# Skip the agent machinery for a small inference job.
llm_call(
    messages=[{"role": "user", "content": "Return the sentiment: The fix worked perfectly."}],
)
```

---

## Five-minute setup

### 1. Install Herald on an origin profile

```bash
hermes plugins install bennybuoy/hermes-herald --enable
```

Install Herald on profiles that will **originate** calls. A target-only profile needs Hermes’s API server, not the plugin. Restart the process that loads the origin profile after installation.

Manual installation is also supported, but user plugins are opt-in:

```bash
git clone https://github.com/bennybuoy/hermes-herald.git ~/.hermes/plugins/hermes-herald
hermes plugins enable hermes-herald
```

For a named origin, prefix the commands below with `hermes -p <origin>`.

For the full local delegation workflow, retain or set this in the origin's
`config.yaml` (merge it with any existing `plugins` settings):

```yaml
plugins:
  isolation: in_process
```

This setting is profile-wide, not a sandbox exemption for just Herald. Only use
in-process plugins you trust. Do not switch an isolation-required deployment
without reviewing that trust decision. Start a fresh origin process and session
after changing isolation; do not restart an active gateway just to try an example.

### 2. Prepare a target profile

On current Hermes, one default gateway serves named profiles through a shared
API listener. Do not start another listener for each named profile. The example
below uses port `8642`; use the actual default gateway's configured API port.

```bash
hermes profile create tutor --description "A patient teaching specialist"
hermes -p tutor setup

HERALD_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
hermes -p tutor config set API_SERVER_KEY "$HERALD_KEY"

# Store the same transport key in the origin profile's .env.
hermes config set TUTOR_API_KEY "$HERALD_KEY"
unset HERALD_KEY
```

Enable/configure the default gateway's API server using Hermes setup if it is
not already enabled. Its own listener key is separate from the named profile's
key. `/p/tutor/` authenticates with **tutor's `API_SERVER_KEY`**, never the default
listener's key. A named profile must be served by the multiplexer and must not
enable its own `api_server`; unknown/unserved prefixes fail closed.

Herald does **not** invent the target's model catalog. `list_profile_models(profile="tutor")` authenticates to the configured base URL plus `/v1/models`, which advertises the profile identity plus configured aliases — not every model its providers can run, and not anything in origin `hermes_herald.profiles`.

**Alias ownership depends on topology.** At the tested shared-gateway revision
`1f2b9e08644c6de4a5b5c50d0e4676a19c5577c5`, `model_routes` belongs to the
default listener's API adapter; `/p/tutor/v1/models` does not load a separate
alias table from tutor's YAML. Configure aliases in the **listener-owning
profile's** `config.yaml`, then verify the authenticated listing on the exact
target URL. For a standalone target, configure its own `config.yaml` instead:

```yaml
platforms:
  api_server:
    extra:
      # ... existing listener host, port and key ...
      model_routes:
        tutor-fast:
          provider: ollama-cloud
          model: glm-5.2
        tutor-reasoning:
          provider: openai-codex
          model: gpt-5.6-sol
```

Use one alias per `(provider, model)` pair you want callable from Herald. Pin `provider` on each alias so the target cannot pick a subscription-less route for the same model name. Reload by restarting the listener-owning gateway when safe. A short listing means those aliases were not advertised; do not invent a Herald-side catalog. Listener-owned aliases are not a separate per-profile allowlist.

The shared service is the default gateway (`hermes -p default gateway run`, or
its installed service). If it is already running, do not start a duplicate.
See Hermes's [multi-profile gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/multi-profile-gateways).

**Standalone/legacy topology:** an older independent target gateway, or a
current target explicitly configured with `gateway.standalone: true`, owns its
own API listener, port, key and `model_routes`. Configure that target's API
server and use its unprefixed base URL (for example `http://127.0.0.1:8652`).
`hermes -p tutor gateway run` is appropriate for that topology, not for starting
another listener inside the shared gateway.

> [!IMPORTANT]
> The API key is a **bearer credential for a terminal-capable Hermes agent**, not a model-provider key. Use a unique random key per target trust boundary. Keep it out of Git and logs. Expose the API only on loopback, a trusted private network, or behind authenticated TLS.

### 3. Declare the origin’s outbound route

Add this to the origin profile’s `config.yaml`:

```yaml
hermes_herald:
  # Optional when a custom/Docker HERMES_HOME cannot reveal the profile name.
  # origin_name: coordinator
  allow_self: false

  profiles:
    tutor:
      url: http://127.0.0.1:8642/p/tutor
      api_key: ${TUTOR_API_KEY}
      # Required explicit outbound grants. Missing/malformed means no calls.
      capabilities: [dispatch, chat]
      # Optional exact target model_routes alias used by both dispatch tools.
      # model: tutor-fast

  # Activity-stall timeout for dispatch_chat, not a flat wall-clock deadline.
  chat_timeout: 600

  # Optional paths; defaults live under the active HERMES_HOME.
  # state_file: /private/path/hermes-herald-runs.json
  # ledger_file: /private/path/hermes-herald-network.db
```

`capabilities` is deliberately fail-closed. Use `[dispatch]`, `[chat]`, or `[dispatch, chat]`.

Use a base URL without `/v1`: Herald appends API paths while preserving
`/p/tutor`. A standalone target uses its own unprefixed port instead.

Each route's `api_key` is the **target profile's own inbound API token** — the
same value you stored as the target's `API_SERVER_KEY` in step 2. A dispatch
failing with `401 Unauthorized` almost always means this
route carries a stale or wrong copy of that key, not that the target is down:
check the target's inbound key and update the origin's secret reference without
printing either value before debugging further.

### 4. Restart and verify

Start a fresh origin session and look for:

```text
hermes-herald: registered 12 tools
```

Then verify reachability and authenticated model discovery:

```python
ping_profile(profile="tutor")
list_profile_models(profile="tutor")
```

For local `llm_call` routing, omit `profile`. The result is a fail-closed
allowlist built from Hermes's explicit-only inventory for the calling profile:
the configured default plus every provider explicitly configured by the user.
It returns the exact provider and model slugs the agent can copy into
`llm_call`. Virtual MoA routes, ambient or auto-discovered credentials, and
fallback-only routes are excluded:

```python
list_profile_models()
```

Hermes executes the selected pair through its public `ctx.llm` plugin service,
so credentials and transport stay host-owned. Explicit overrides are also
subject to Hermes's per-plugin trust policy. Allow only the providers and
models this plugin should be able to select:

```yaml
plugins:
  entries:
    hermes-herald:
      llm:
        allow_provider_override: true
        allowed_providers: [openai-codex, ollama-cloud]
        allow_model_override: true
        allowed_models: [gpt-5.6-sol, glm-5.2]
```

Without that trust block, `llm_call` can still omit both routing fields and use
the host's active configured route; explicit provider/model selection fails
closed.

The plugin also bundles an opt-in operating skill:

```python
skill_view(name="hermes-herald:agent-dispatch")
```

### 5. Talk to the target

```python
dispatch_chat(
    profile="tutor",
    message="Explain Python decorators with a simple example",
)
```

A follow-up to `tutor` reuses the stored target session unless `new_session=True`.

---

## Upgrading from Agent Dispatch 1.x

Hermes Herald renames the plugin identity from `agent-dispatch` to
`hermes-herald`. The tool names are unchanged, but Hermes's plugin allow-list
uses the manifest identity. Replace the old entry before restarting:

```yaml
plugins:
  enabled:
    # Remove: agent-dispatch
    - hermes-herald
```

If the old checkout is installed at `~/.hermes/plugins/agent-dispatch`, move it
to `~/.hermes/plugins/hermes-herald` or reinstall it with the command above. Do
not keep both checkouts: they advertise the same capabilities and make plugin
discovery ambiguous.

---

## Pillar 1 — A network of named Hermes agents

A Herald target is a real Hermes profile, not an anonymous prompt wrapper. It can have its own:

- SOUL and operating instructions;
- model and provider credentials;
- skills and memory;
- tools and filesystem;
- Home Assistant, browser, GitHub, email, or other integrations;
- machine, container, or network location.

### Persistent inter-agent conversation

`dispatch_chat` sends one synchronous turn over a streaming `/v1/chat/completions` connection and preserves continuity with `X-Hermes-Session-Id`.

```text
Coordinator                         Tutor profile
    │                                    │
    ├─ "Explain decorators" ────────────▶│ turn 1
    │◀──────────────────────── "Here's…" ┤
    │                                    │
    ├─ "Show a caching version" ────────▶│ turn 2, same history
    │◀──────────────────── "Use lru_cache"┤
```

While the call is synchronous, target assistant deltas and tool lifecycle events keep the activity timer alive. Tool progress is relayed to the origin UI when that surface exposes a progress callback. SSE keepalives prove the transport is connected but do not count as agent activity.

```python
dispatch_chat(
    profile="tutor",
    message="Challenge my proposed explanation",
    model="tutor-reasoning",       # exact verified model_routes alias
    instructions="Be direct.",    # current turn only
    stall_timeout_seconds=900,     # activity-based
    new_session=False,
)
```

Important contracts:

- the call blocks the **current origin agent turn**, not every Hermes session or process;
- later calls to the configured profile name reuse its saved target session;
- `new_session=True` creates and stores a fresh session;
- `instructions` applies to the current turn and should be resent when still needed;
- `model` is verified against authenticated `/v1/models`; omission uses the target default;
- the stall timer resets on actual assistant or tool activity, so productive long turns can exceed it indefinitely;
- persistent chat is synchronous and does not return a separately pollable Herald `run_id`.

A chat transport timeout returns `error_code="chat_timeout"` with the target
`profile` and `session_id`. The remote turn may still be running: inspect that
session before deliberately sending more work. Herald does not automatically
resend a timed-out chat. The socket wait uses the requested inactivity window,
without an additional 60-second cap.

### Parallel or detached work

`dispatch_agent` starts a target `/v1/runs` task and returns a `run_id` immediately.

Remote callback and approval notices show `API calls: ?` because the run API
reports token usage, not an inference-call count. Token usage remains in the
run state and dispatch ledger; it is never presented as a call count.

```python
# Default: monitor through SSE and deliver the result back to this session.
dispatch_agent(
    profile="reviewer",
    message="Review the repository for release blockers. Do not edit.",
    model="reviewer-fast",
)

# Detached: no local listener or callback. Inspect it later.
dispatch_agent(
    profile="researcher",
    message="Continue this workflow independently.",
    delivery="none",
    max_hops=6,
)
```

With `delivery="callback"` (default), Herald monitors structured target events and reinjects the terminal result into the commissioning session. With `delivery="none"`, the target continues without a local listener; use `check_dispatch` or `dispatch_status` later.

Herald also provides:

- batch polling with `collect_dispatches`;
- cooperative cancellation with `cancel_dispatch`;
- durable call and model provenance in SQLite;
- trace, parent-edge, and hop lineage for forwarded agent graphs;
- optional `max_hops` to stop repeated A→B→A forwarding before another edge is created;
- protected-command notice relay with human-confirmed remote denial.

### Safe deliberate retries of `dispatch_agent`

Each call sends `Idempotency-Key` and commits a SQLite reservation **before**
POST. Supply an optional `idempotency_key` before the first call when you need
to retain identity even if the entire tool response is lost:

```python
# First attempt, with an identity chosen for this one piece of work.
dispatch_agent(profile="reviewer", message="Review the diff. Do not edit.",
               delivery="none", idempotency_key="review-release-2026-10-11-01")
# Only if deliberately retrying that SAME work: identical original arguments.
dispatch_agent(profile="reviewer", message="Review the diff. Do not edit.",
               delivery="none", idempotency_key="review-release-2026-10-11-01")
```

- Keys are 1–255 visible ASCII characters, without spaces; Herald never trims
  or repairs them. Do not put secrets in keys or task text.
- Omit the key for **new intentional work**. Herald generates and returns a key
  on success and ambiguous failure, with `retry_instruction`. Identical messages
  without a key remain separate tasks. Retain the returned key; if the entire
  origin/tool response disappears before you receive a generated key, there is
  no caller-known retry identity. Prefer a supplied key in that situation.
- A deliberate retry must reuse the exact key and all original supplied arguments
  (including delivery, instructions, model/provider, reasoning, and lineage),
  from the same origin home/profile and owning durable session. Detached callers
  without a session are scoped to their origin. `/new` cannot transfer ownership.
  URL, configured default model, or credential rotation fails closed; credentials
  themselves are not stored, only a one-way credential fingerprint.
- After a lost acceptance response, Herald reuses the **original full body**,
  including generated edge/trace IDs, and surfaces the original target `run_id`.
  A locally recorded success returns that handle without another POST, ledger
  edge, listener, or callback. Concurrent same-key callers get
  `idempotency_in_progress`; wait and retry unchanged. An abandoned caller holds
  its reservation for up to 120 seconds before safe recovery can proceed.
- Ambiguous replay requires authenticated `GET /v1/capabilities` to advertise
  `features.runs_idempotency` with `supported=true`, `durable=true`, and a valid
  `retention_seconds`, both at original submission and retry. Herald uses the
  original reservation time plus advertised retention, minus a 60-second safety
  margin, never extends the horizon on retries, and refuses expired or unverifiable
  replay. The inspected
  Hermes API reports 24-hour retention; this is not a universal server promise.
  Older/non-durable targets can still accept initial work, with a warning, but
  cannot safely recover ambiguous acceptance through Herald.
- HTTP 409 is a distinct `idempotency_server_conflict`, not a connectivity error.
  Changed local arguments/routing/ownership produce `idempotency_conflict`.
  Do **not** remove/change the key to bypass either error or unsafe replay:
  reconcile existing work with the target operator and known run handles first.
- This is not unconditional exactly-once execution. It relies on the target
  honoring its advertised namespace, persistence, and retention; replacing or
  deleting its database breaks that boundary. Preserve Herald's ledger as well:
  identity tombstones and original bodies are retained, not automatically pruned.
  Removing them can make a previously used key look new. Task text remains private
  persisted data. A crash after acceptance but during local bookkeeping can leave
  audit/cache/monitoring incomplete; retries return the handle with a warning,
  not a second listener. Origin restart does not restore callbacks. Poll with
  `check_dispatch`; replay identity does not make target execution restart-proof.

This contract applies only to `dispatch_agent`, not chat, local delegation,
approval, cancellation, or inference tools.

<details>
<summary><strong>Async monitoring and recovery details</strong></summary>

The listener consumes `/v1/runs/{run_id}/events`. If SSE drops, Herald checks authoritative run status before reconnecting, then retries with 5/10/20/40-second backoff. After five total failed connections, it switches to authenticated polling.

SSE activity and advancing target `updated_at` timestamps reset the normal 600-second stall timer. Approval waits use 30 minutes. Five consecutive polling transport failures end monitoring with instructions to reconcile using `check_dispatch`.

Auto-delivery is process-bound: restarting the origin loses its active listener. The durable SQLite ledger retains provenance and the bounded JSON cache retains live recovery data, but Herald does not recreate an old listener or inject a result into a dead session. Recover the handle with `dispatch_status`, then query the target with `check_dispatch`.

</details>

---

## Pillar 2 — A model switchboard for local subagents

Hermes core’s `delegate_task` uses normal delegation routing. `delegate_subagent`
adds **per-call model choice**. The supported workflow, including automatic
completion delivery and inheritance/reasoning controls, requires
`plugins.isolation: in_process`. Shared-gateway profile scoping does not change
this plugin-execution requirement.

### Experimental isolated host: explicit polled lifecycle

**Minimum tested prerequisite:** Hermes transport-fix [PR #133762](https://github.com/NousResearch/hermes-agent/pull/133762), exact
candidate **`c2a785b4e3a59ba49dc564e23e5bffab89dd9cce`** (or a later upstream
revision retaining that fix and lifecycle contract). This is a PR candidate,
not a claimed released-version requirement. Herald adapts to Hermes's existing
`ctx.subagent_lifecycle`; no Hermes core patch is included here.

The CI lane pins the later tested revision
**`c72f29e0a3608e2276e311c1c9a891aedf2acb17`**. Neither revision supplies Herald's
automatic host-completion workflow. A passing adapter suite proves only the
explicitly polled subset; it does not certify full isolated-host support.

When Hermes loads Herald inside `plugins.isolation: host`, registration selects
a host-specific schema and handler for the same `delegate_subagent` tool:

```python
delegate_subagent(action="launch", goal="Review the diff. Do not edit.",
                  context="Repo: /srv/app; focus on token handling.",
                  provider="openai-codex", model="gpt-5")
# Response includes delivery="poll", opaque task_id, and:
# next_action={"action": "result", "task_id": "<returned task_id>"}
delegate_subagent(action="status", task_id="<returned task_id>")
delegate_subagent(action="result", task_id="<returned task_id>")
delegate_subagent(action="steer", task_id="<returned task_id>", text="Focus on expiry handling")
delegate_subagent(action="cancel", task_id="<returned task_id>", reason="No longer needed")
```

- Omit `action` for launch. `goal` is required for launch; controls require the
  exact returned Herald `task_id`, and steering requires `text`. Control calls
  must occur in the **owning active Hermes session**. Hermes rechecks ownership
  on every action; a caller-supplied session label cannot transfer ownership.
- Delivery is **poll only**. Follow the returned `next_action` on this same
  tool, not `check_dispatch`. A not-ready result returns `ready=false`; a ready
  result retains the terminal summary/error and usage metadata and is repeatable.
  Do other useful work between polls. No completion message is promised.
- Herald performs no background RPC/polling after tool return: that path loses
  the active parent in the current host transport. `ctx.inject_message` is
  consent-gated user-input scheduling, not native session-safe completion, and
  is neither enabled nor used by this adapter.
- `model` and `provider` pass unchanged to the lifecycle API. Pin both for an
  exact provider/model pair; this is **not** the lenient `/model` pipeline.
  Omitted provider uses delegation config first, then the parent; omitted model
  uses Hermes's delegation route. Do not assume omission always follows the
  live parent's model/provider.
- Tools use Hermes's normal safe inherited parent policy (including its MCP
  inheritance rules). Only omitted or `inherit_toolsets=true` is supported.
  **All explicit toolset lists, including `[]`, and `inherit_toolsets=false`
  fail before launch.** The current lifecycle API turns an empty list into
  inheritance, can append MCP tools to explicit lists, and does not ensure
  Herald's exact-subset contract. Herald never substitutes inherited tools for
  a requested model-only/restricted child.
- `inherit_context`, `inherit_soul`, per-call `reasoning_effort`,
  `stall_timeout_seconds`, `interrupt_after_seconds`, or other timeout fields
  are rejected when supplied, even if false/null/default-valued. No option is
  silently ignored. Provide explicit `context`; normal Hermes delegation
  reasoning/timeout and child identity defaults apply. Use `in_process` for
  Herald's legacy per-call policies. Role overrides, per-tool blocking and
  working-directory overrides are also unsupported. Hermes determines the
  actual child role from its delegation depth policy; Herald does not promise
  a non-delegating leaf when the host permits further delegation.
- Cancellation requests cooperative interruption, not confirmed termination.
  Steering queues text only while supported by the child; check `accepted`.
- Herald stores at most **128 tasks per registration**, in memory only. At
  capacity, new launches fail without evicting prior tasks or creating a child.
  Capability-bearing core handles are private, never returned or logged by the
  adapter. No handles/results are persisted to Herald's dispatch ledger/cache.
  Host unload/restart loses these IDs; children may still be running in Hermes.
  Retrieve results and cancel unfinished work before restarting. Hermes's
  terminal retention is **one hour**; retrieve promptly. Neither host nor parent
  restart recovery is supported. Use `dispatch_agent` for durable target work.

### Supported in-process mode

```python
delegate_subagent(
    goal="Review auth.py for security issues. Do not edit.",
    model="gpt-5",
    context="Repository: /srv/app; focus on token handling.",
    inherit_context=True,
    inherit_soul=True,
    inherit_toolsets=False,
    toolsets=["file", "terminal"],
    stall_timeout_seconds=600,
    interrupt_after_seconds=1800,
)
```

### Decide exactly what the child inherits

| Input | Default | Behaviour |
|---|---:|---|
| `context` | none | Explicit task facts supplied regardless of inheritance mode |
| `inherit_context` | `false` | Opt-in copy of the latest 20 parent user/assistant text messages, capped at 12,000 characters; excludes system, tools, results, memory, and hidden state |
| `inherit_soul` | `false` | Loads the active parent profile’s full `SOUL.md` as the child identity |
| `inherit_toolsets` | `true` | Inherits the parent’s safe child tool subset when `toolsets` is omitted |
| `toolsets=[]` | omitted | Creates a model-only child |
| `toolsets=[...]` | omitted | Requests an exact subset, intersected with parent capabilities; inherited MCP toolsets are stripped unless explicitly named |

The child runs asynchronously in a daemon thread and returns a `task_id` immediately. Streamed text and tool events reset the stall timer. `interrupt_after_seconds` requests cooperative interruption after a wall-clock threshold; it cannot instantly kill a provider or blocking tool call.

Because the child is in-process, it is not durable across parent-process exit. Use `dispatch_agent` when process isolation or durable target execution matters.

The final summary/error is submitted to Hermes's native completion pipeline for
the **original durable parent session**. It is not retargeted to a replacement
conversation after `/new`. Keep the commissioning session and process alive;
automatic delivery is not restart recovery.

Model selection reads the active profile's configuration, not a launch-profile
global snapshot. An unchanged provider/endpoint/key/API-mode bundle inherits the
parent's live runtime. A changed bundle is pinned for that child, including
same-provider endpoint or key changes; the child's credential pool and per-turn
env refresh cannot replace it. Parent and ordinary inherited-child policies are
unchanged. Never put provider credentials in the task brief.

`delegate_subagent` requires the live parent-agent context exposed by classic interactive CLI and desktop/TUI sessions. Gateway/API plugin-dispatch paths fail closed rather than guessing another session.

> Hermes core's global `delegation.child_timeout_seconds` must be `0` or unset.
> Herald refuses launch otherwise because core could preempt the per-call policy.

### Per-call reasoning effort

`delegate_subagent` accepts `reasoning_effort` to set the child's thinking budget for this one call, without touching core's `delegation.reasoning_effort` config:

```python
delegate_subagent(
    goal="Deep-dive the auth flow for security issues",
    model="gpt-5",
    reasoning_effort="high",   # minimal..ultra; 'none' disables thinking
)
```

Omitted, the child keeps the normal resolution order (core `delegation.reasoning_effort` when configured, otherwise the parent's level). Herald assigns `child.reasoning_config` after build via the same post-build seam as SOUL inheritance. Downstream host policy can still clamp, drop, or substitute the requested budget (mandatory-reasoning rejection, length recovery, provider limits); do not treat the parameter as a strict every-request guarantee against the host.

---

## Pillar 3 — A lightweight inference lane

`llm_call` delegates one completion to Hermes's public plugin LLM service,
without an agent loop, tool schemas, terminal access, or subagent overhead.
Hermes owns authentication, transport, provider behavior, timeouts, and routing
policy; Herald never handles provider credentials.

```python
# Classification
llm_call(messages=[{
    "role": "user",
    "content": "Classify as urgent or routine: Production database is unavailable.",
}])

# Structured extraction
llm_call(
    messages=[{
        "role": "user",
        "content": "Extract name and age: John Smith is 32.",
    }],
    json_mode=True,
)

# Translation on a selected route
llm_call(
    messages=[{"role": "user", "content": "Translate to French: Good morning."}],
    provider="openai-codex",
    model="gpt-5.6-sol",
)
```

Use it for tasks where tools and iterative reasoning would be overhead: routing, scoring, rewriting, extraction, translation, compact summarisation, and schema-constrained JSON.

With no routing arguments, `llm_call` uses the host's active configured route.
Before choosing an override, call `list_profile_models()` and copy one exact
returned `{provider, model}` pair. Provider and model must be supplied together.

Prefer `configured_default`. Do not select another provider merely because it
advertises the same model; override the route only when the task or user
explicitly calls for that endpoint.

Herald checks that the exact pair appears in the configured inventory, then
passes it unchanged to `ctx.llm.complete()`. Hermes applies the plugin trust
allowlists and owns the actual provider resolution. Unknown pairs fail before
the host call; policy or provider failures are returned noisily without a
Herald-side retry. Results expose the requested pair alongside the provider and
model reported by the host facade.

`llm_direct` is a separate, **opt-in** research lane. It does not use host
routing. Endpoints are named in config (`hermes_herald.llm_direct.enabled: true`,
explicit `http(s)` `base_url`, `api_key: ${ENV_VAR}` only). The tool never
accepts a URL or credential as an argument. `extra_body` is for vendor-specific
fields only — reserved OpenAI fields must go through the matching parameters.
Redirects are refused so a bearer token cannot follow a 3xx to another origin.

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

```python
llm_direct(
    messages=[{"role": "user", "content": "Reply with the single word OK"}],
    temperature=0,
    seed=7,
    extra_body={"top_k": 40},
)
```

---

## The 12 tools

| Tool | Purpose |
|---|---|
| `dispatch_chat` | Persistent synchronous conversation with streamed activity and verified target model selection |
| `dispatch_agent` | Async cross-profile run with callback or detached delivery |
| `delegate_subagent` | Supported local delegation in `in_process`; experimental polling-only subset in `host` |
| `llm_call` | Bare model inference without an agent loop |
| `llm_direct` | Opt-in direct OpenAI-compatible call to a pre-configured endpoint (full parameter control; credentials stay in config) |
| `check_dispatch` | Query one target run using its exact `{profile, run_id}` |
| `collect_dispatches` | Query several run handles in one call |
| `dispatch_status` | Read durable call history and credential-free configured/observed topology |
| `cancel_dispatch` | Request cooperative target cancellation and suppress late local delivery |
| `approve_dispatch` | Deny the exact visible protected-command notice through human confirmation |
| `ping_profile` | Check target API reachability |
| `list_profile_models` | Discover configured local `llm_call` routes, or exact remote `model_routes` aliases accepted by both dispatch modes |

---

## Routing policy is not caller authentication

This distinction is deliberate and non-negotiable.

A route such as:

```yaml
profiles:
  reviewer:
    capabilities: [dispatch]
```

means:

> “Calls made through Herald on **this origin profile** may dispatch to `reviewer`, but may not open persistent chat.”

Herald enforces that rule before network contact. Missing or malformed capabilities grant nothing. Adding reviewer to coordinator creates coordinator→reviewer only; the reverse requires a separate outbound entry on reviewer.

It does **not** mean:

> “The reviewer API server can cryptographically identify coordinator and reject every other bearer-key holder.”

Hermes’s API server currently authenticates the target bearer key. It does not map different client keys to verified Hermes profile principals. Anyone who can reach the endpoint and possesses that key has the key’s authority, including callers that bypass Herald with HTTP or terminal tools.

Therefore:

- Herald’s route table is a useful, enforced **caller-side policy and accidental-recursion guard**;
- it is not a bilateral network ACL, sandbox, or target-side identity firewall;
- `origin_name` is audit attribution, not authenticated caller identity;
- use a unique target/key per trust boundary, network ACLs/Tailscale, containers, or a credential-aware reverse proxy when callers need different authority;
- true “only profile A may call profile B” enforcement requires per-caller principals in Hermes core or equivalent infrastructure.

Self-routing requires both a matching route entry and `allow_self: true`. Async `max_hops` is an operational loop brake, not a cryptographic control; a malicious caller can discard model-carried lineage or bypass Herald.

---

## Model routing

Both `dispatch_agent` and `dispatch_chat` support target-controlled model choice.

**Where aliases come from.** They live under `platforms.api_server.extra.model_routes` on the **API-listener owner**: the default gateway for shared `/p/<profile>` URLs at the tested revision, or the target itself for standalone gateways. The origin Herald route (`hermes_herald.profiles.<name>`) is only `url`, `api_key`, `capabilities`, and an optional default `model` that must already be advertised by that endpoint. Herald never writes `model_routes`.

**What `/v1/models` lists.** The URL-selected profile identity (not a valid override) plus the listener's aliases. Provider catalogs, fallbacks, and ambient credentials are excluded on purpose. If a model is missing from `list_profile_models(profile=...)`, check the listener owner's aliases and restart that gateway when safe. Do not assume that a secondary profile's YAML supplies a separate alias catalog.

When `model` is supplied explicitly or configured on the Herald profile route, Herald:

1. authenticates to the target’s `GET /v1/models`;
2. requires an exact `model_routes` alias with a resolved target model;
3. refuses unknown, unverifiable, or primary-identity-only names before sending work;
4. records requested and resolved model provenance in the ledger.

Copy one alias from `pass_as_model` (or an entry's `pass_as`) into `model=`.
Copy the advertised `provider` when present; an explicit provider must match a
pinned alias. If the listing omits it, verify the listener's alias configuration
with its operator rather than guessing a provider. An unpinned alias can let the
target choose a route with no subscription. Both dispatch tools also accept
per-call `reasoning_effort`; the target may clamp, ignore or drop that request
under its own reasoning policy.

Omit `model` to preserve the target’s normal default. Call `list_profile_models(profile=...)` before selecting an alias.

This is intentionally stricter than calling the native API directly: a typo must not silently fall back to a different model.

---

## Persistence, audit, and graph lineage

Herald uses two local stores:

- a bounded JSON state cache for live run handles, target chat sessions, listeners, and approval recovery;
- a SQLite ledger for durable call provenance, status, model resolution, timing, delivery mode, trace lineage, and optional full task text.

`dispatch_status(include_messages=False, include_topology=True)` is the safe default. Set `include_messages=True` deliberately: the ledger excludes configured bearer credentials, but it stores task text exactly as supplied. Do not put secrets in dispatch briefs.

Same-filesystem origins may share one `ledger_file` for a combined observed graph. Never share one JSON `state_file` between processes.

A post-side-effect state-cache failure does not hide a remote handle: Herald returns the handle, records the ledger edge when possible, and includes an explicit warning.

---

## Approval relay

For autonomous target profiles, prefer Hermes’s `smart` approval mode. Use manual Herald relay only when a human-attended origin is intentionally part of the workflow.

When the target emits `approval.request`, Herald delivers the redacted request to the commissioning session. `approve_dispatch` can deny that request and requires:

- the exact target `{profile, run_id}`;
- a fresh Herald `approval_id` bound to the originating session;
- the visible FIFO head;
- explicit human confirmation through Hermes’s owned approval surface.

Cross-session, cross-profile, stale, replayed, or unseen requests fail before the target is contacted. **Herald v1 is deny-only:** `choice="once"`, `"session"`, and `"always"` are rejected before profile resolution or network contact. Current Hermes targets resolve approvals by FIFO position rather than an immutable target request ID, so positive remote approval cannot be proven to authorize the command that was shown. `resolve_all=True` is available only with `choice="deny"`.

Target command descriptions remain untrusted display data. If the listener was lost before the command preview arrived, do not approve it—reconcile or cancel the run.

---

## Architecture

```text
                         HERMES HERALD

 Origin profile
 ├─ dispatch_chat ───── streaming persistent turn ───▶ Named target profile
 ├─ dispatch_agent ─── async /v1/runs task ─────────▶ Named target profile
 │                       │
 │                       └─ SSE → reconnect → polling → callback/ledger
 ├─ delegate_subagent ─ in-process child → native parent completion
 │                     experimental host adapter → explicit result polling
 ├─ llm_call ────────── Hermes-owned plugin LLM service
 └─ llm_direct ──────── opt-in configured OpenAI-compatible endpoint

 Target profiles retain their own SOUL, skills, memory, credentials and sessions.
 In a shared gateway, API model aliases belong to the listener owner;
 profiles are not separate OS/filesystem sandboxes.
```

Reliability properties include redirect refusal for credentialed HTTP, atomic state replacement, WAL-backed SQLite, bounded previews, session-owned approval nonces, cooperative cancellation, model-route verification, activity-aware stalls, and zero Hermes core patches.

---

## Trust boundaries

- A valid target transport key is powerful. Protect it like remote execution access.
- Herald does not provide TLS. Use loopback, a trusted private network, or authenticated TLS.
- Target provider credentials, skills, memory, and tool credentials remain on the target.
- Target responses are untrusted input when reinjected into the origin. Verify consequential side effects independently.
- Async callback delivery is not a durable queue; recover target state with `dispatch_status` and `check_dispatch` after origin restart.
- Persistent chat stores one target session handle per configured profile name; use `new_session=True` to start fresh.
- A target gateway restart can interrupt active work even when the origin still knows the handle.

### Multiplex support

Herald origins may run in separate single-profile processes or inside a multiplexed gateway.
Configuration and the default state and ledger paths follow Hermes' active profile home;
the config cache is keyed by canonical home. `${VAR}` credentials resolve at call time
through the active profile's secret scope, not another origin's process environment.
Single-profile processes retain their environment-based configuration and credentials.

An A→B→A integration test through the real `PluginManager`, gateway profile scope, and
registered `dispatch_status` handler proves separate route tables, ledger rows, credentials,
and recovered sessions and approvals for two origins in one process. Background listeners
and delegated subagent threads carry the origin's profile context across thread hops
(`contextvars.copy_context()` at spawn), and custom `state_file`/`ledger_file` paths expand
`${HERMES_HOME}`/`$HERMES_HOME` (and other variables) within the active profile scope.

This is profile scoping, not a sandbox for untrusted plugins. An explicitly configured
shared ledger still exposes the combined history to its participating origins. Keep JSON
state files private to each origin. Missing scoped credentials never borrow another
origin's key, and route capability grants and target authentication remain required.

---

## Development and tests

The repository workflow is the executable compatibility contract:

| Lane | Exact Hermes revision | Python | Delegation coverage |
|---|---|---|---|
| `legacy` | `9fc80ac70f6b97e36556c33bc4895b75558d4136` | 3.11 | Supported `in_process` |
| `shared-gateway` | `1f2b9e08644c6de4a5b5c50d0e4676a19c5577c5` | 3.14 | Supported `in_process` |
| `host-candidate` | `c72f29e0a3608e2276e311c1c9a891aedf2acb17` | 3.14 | Experimental poll-only host adapter |

The manifest's `requires_hermes: ">=0.21.0"` is the general installation floor,
not a claim that every intermediate host revision or optional isolation mode is
certified. Build the isolated environment for the chosen pin using
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). With its Hermes checkout at
`hermes-agent/` and test interpreter at `$HOME/.ci-venv/bin/python`, run the release
check and the workflow's test selection:

```bash
scripts/ci-python scripts/check_release_contract.py

# legacy/shared-gateway: omit only the real host-transport test file.
# host-candidate: select every test file, including real host subprocesses.
export HERALD_CI_ISOLATION=in_process  # use host for the host-candidate lane
tests=()
for test in "$PWD"/tests/test_*.py; do
  if [[ "$HERALD_CI_ISOLATION" == in_process && "${test##*/}" == test_host_subagent_lifecycle.py ]]; then
    continue
  fi
  tests+=("$test")
done
HERMES_PYTHON="$PWD/scripts/ci-python" \
  bash hermes-agent/scripts/run_tests.sh "${tests[@]}" -j 2 --file-retries 0 \
  -p no:cacheprovider -o addopts= --confcutdir="$PWD/tests" --rootdir="$PWD/tests" -q
```

For a custom isolated layout, the equivalent runner invocation is:

```bash
HERMES_PYTHON=/path/to/herald-test-python \
  bash /path/to/hermes-agent/scripts/run_tests.sh /path/to/herald/tests \
  --rootdir=/path/to/herald/tests --confcutdir=/path/to/herald/tests -q
```

Use the Hermes PM-managed test interpreter behind a small `herald-test-python`
wrapper that exports `HERMES_SOURCE_DIR`, `HERMES_HERALD_PLUGIN_DIR`, and candidate
`PYTHONPATH` on each invocation: the canonical runner deliberately sanitizes the
environment. Use a synthetic `HOME` and scratch-backed temporary directories;
never run against live profile state. Herald's conftest gives each per-file test
process its own temporary `HERMES_HOME`. The host regression uses a real plugin
subprocess and synthetic child execution only, with no paid inference.

The suite covers YAML compatibility, scoped route and credential pinning,
spawning-session ownership, host lifecycle launch/return/poll, foreign-session rejection,
pre-launch policy refusal, capacity/error privacy, and unchanged in-process
registration, alongside streaming chat, model routes, approval relay, ledger
and release contracts. The CI matrix pins the exact host snapshots;
isolated-host tests require the transport candidate described above and are
not evidence of host-isolation support on the legacy or shared-gateway pins.

### Troubleshooting completion delivery

- **In-process local child:** retain the original session; `/new` is a new
  durable conversation even when the UI tab is unchanged. Process exit ends
  local children. Do not use `check_dispatch` for local `task_id` values.
- **Experimental host child:** no completion notification is expected. Use
  `delegate_subagent(action="result", task_id=...)` in the owning active session.
  If automatic completion or exact tool restrictions are required, use the
  supported in-process mode rather than dropping requirements.
- **Remote dispatch:** query the original `{profile, run_id}` with
  `check_dispatch`, and use `dispatch_status` to recover persisted provenance.
  Reconcile the existing run before redispatching; callback delivery is not a
  durable queue.

## License

MIT — see [LICENSE](LICENSE).

Built by Ben Kamholtz for [Hermes Agent](https://github.com/NousResearch/hermes-agent), created by [Nous Research](https://nousresearch.com).
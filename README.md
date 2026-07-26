<img src="assets/hero-banner.png" alt="Hermes Herald — Multi-Agent Dispatch for Hermes Agent" width="100%">

# Hermes Herald

<p>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Hermes%20Agent-compatible-8B5CF6" alt="Hermes Agent"></a>
  <img src="https://img.shields.io/badge/version-1.0.0-22C55E" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/tools-11-orange" alt="11 Tools">
</p>

> **Your agents shouldn't just talk to you. They should talk to each other.**

Hermes Herald turns Hermes Agent from a solo operator into a **conductor of agents**. It adds three independent capabilities:

| Capability | What it gives you |
|---|---|
| **📡 Agent-to-agent conversation** | Talk to named, separately configured Hermes profiles over authenticated API endpoints. Keep a multi-turn session or fan tasks out asynchronously. |
| **🔀 Per-call subagent models** | Spawn an isolated in-process subagent with the model best suited to one task, without changing the parent agent's runtime. |
| **🧠 Bare LLM inference** | Run classification, translation, summarisation, extraction, and scoring without loading an agent loop or tool schemas. |

The headline feature is genuine inter-agent dialogue. Agent A can ask Agent B a question, receive an answer, ask a follow-up, and get a response with the earlier session history. For independent work, asynchronous dispatch returns immediately and delivers the result later through Hermes's Runs API and SSE event stream.

Why use separate agents instead of one enormous prompt?

- **Isolation:** each target keeps its own model, skills, memory, tool permissions, credentials, and operating context.
- **Continuity:** `dispatch_chat` gives a named profile a reusable conversation instead of a fresh anonymous worker every turn.
- **Parallelism:** dispatch several independent runs and keep the origin agent available while they work.
- **Fit-for-purpose models:** reserve an expensive reasoning model for the tasks that need it and use faster models for reviews, extraction, or classification.
- **Operational control:** discover models, inspect persisted run metadata, poll, batch-collect, cancel, and relay protected-command approvals.
- **Credential separation:** Herald sends a transport bearer token to the target API; the target's upstream model credentials stay on the target.

All three pillars work independently. Use `llm_call` without configuring another profile, use `delegate_subagent` without dispatching anywhere, or build a network of specialist Hermes agents that can talk to one another. Herald does not replace Hermes profile or provider setup: every target must already be a working Hermes profile.

---

## Quick Start

### 1. Install and enable Herald on the origin profile

```bash
hermes plugins install bennybuoy/hermes-herald --enable
```

Install Herald only on profiles that will **originate** calls. A target-only profile needs Hermes's API server, not this plugin. Restart the active Hermes process after installation: reopen a CLI/TUI session, or run `hermes gateway restart` for a gateway process.

The unprefixed `hermes ...` commands below act on the current origin profile. If your origin is a named profile, prefix those commands with `hermes -p <origin>`.

### 2. Prepare a target Hermes profile

Create and set up the target normally. Skip creation if the profile already exists.

```bash
hermes profile create tutor --description "A patient teaching specialist"
hermes -p tutor setup
```

Generate a dedicated transport key, enable the target API server, and store the same key in the origin profile under a profile-specific name:

```bash
HERALD_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

# Target profile. The explicit `extra` path works across Hermes releases.
hermes -p tutor config set --force gateway.platforms.api_server.enabled true
hermes -p tutor config set --force gateway.platforms.api_server.extra.host 127.0.0.1
hermes -p tutor config set --force gateway.platforms.api_server.extra.port 8652
hermes -p tutor config set --force gateway.platforms.api_server.extra.key "$HERALD_KEY"

# Origin profile — custom *_API_KEY names are stored in its .env
hermes config set TUTOR_API_KEY "$HERALD_KEY"
unset HERALD_KEY
```

Start the target gateway in another terminal:

```bash
hermes -p tutor gateway run
```

For a background service, use `hermes -p tutor gateway install` followed by `hermes -p tutor gateway start` instead.

> [!IMPORTANT]
> The API-server key is a **transport bearer token**, not a model-provider key. It authorizes access to a terminal-capable Hermes agent. Use a unique random value per target, protect both profiles' local config files, never commit them, and expose the API only on loopback, a trusted private network, or behind TLS. Configure the target's model/provider credentials separately with `hermes -p tutor setup`.

### 3. Map the target in the origin config

Add a `hermes_herald` block to the origin profile's `config.yaml`:

```yaml
hermes_herald:
  # Optional stable ledger identity for custom/Docker HERMES_HOME layouts.
  # origin_name: marie
  # Self-routing is blocked unless this is explicitly true.
  allow_self: false
  profiles:
    tutor:
      url: http://127.0.0.1:8652
      api_key: ${TUTOR_API_KEY}
      capabilities: [dispatch, chat]
  # Optional: default timeout for dispatch_chat (seconds, default 600)
  chat_timeout: 900
  # Optional: omit for <active HERMES_HOME>/hermes-herald-runs.json
  # state_file: /custom/private/path/hermes-herald-runs.json
  # Durable audit ledger. Profiles on one filesystem may share this path.
  # ledger_file: /custom/private/path/hermes-herald-network.db
```

Do **not** put `model:` on a profile you intend to use with `dispatch_chat`. A profile-level model is an async `dispatch_agent` default, and Herald rejects it on the synchronous persistent-chat path rather than silently applying the wrong runtime.

### 4. Restart and verify

Restart the origin Hermes process so it loads the plugin. Look for:

```
hermes-herald: registered 11 tools
```

Herald also bundles an opt-in operating skill. Load it explicitly when you
want the agent to configure, choose between, or troubleshoot Herald's tools:

```python
skill_view(name="hermes-herald:agent-dispatch")
```

Then verify both reachability and authenticated model discovery:

```python
ping_profile(profile="tutor")
list_profile_models(profile="tutor")
```

`ping_profile` is a liveness check. `list_profile_models` also exercises bearer authentication and reports the target's advertised primary identity plus any exact aliases Herald can safely send to `dispatch_agent(model=...)`.

### 5. Use it

```python
# Ask the tutor a question
dispatch_chat(profile="tutor", message="Explain Python decorators with a simple example")

# Ask a follow-up — the tutor remembers the prior turn
dispatch_chat(profile="tutor", message="Now show me one that caches results")

# Fire off independent work — result arrives via SSE while this process is alive
dispatch_agent(profile="tutor", message="Draft three practice questions about decorators")

# Truly detached graph work — no callback/listener; inspect it later
dispatch_agent(profile="tutor", message="Pass this down the chain", delivery="none")
dispatch_status(include_topology=True, include_messages=False)

# Spawn a subagent with a different model
delegate_subagent(goal="Summarise this codebase", model="gemma4:31b")

# Quick classification — no agent loop, just inference
llm_call(messages=[{"role": "user", "content": "Is this email spam? Subject: 'WIN FREE iPhone'"}])
```

---

## The Three Pillars

Hermes Herald equips your agent with 11 tools across three independent pillars. Each works standalone — you can use `llm_call` without ever dispatching, or dispatch without ever spawning a subagent.

### 📡 Pillar 1: Cross-Profile Dispatch

Hermes core's `delegate_task` is excellent for anonymous in-process workers. Herald adds a tool-level control plane for **named profiles exposed through Hermes's authenticated API server**: reusable conversations, async runs, persisted run metadata, cancellation, model-alias discovery, and approval relay. Targets can be separate local profiles or remote Hermes installations.

#### `dispatch_chat` — agents talking to agents

`dispatch_chat` opens a synchronous connection to a target profile and blocks until the reply arrives. The critical difference: **subsequent calls to the same profile continue the same stored session.** The target receives the prior conversation history, subject to that target's normal retention and context-management behaviour.

```
Your Agent                    Tutor Profile
   │                              │
   ├─ "Explain decorators" ─────▶│  (turn 1)
   │                              │
   │ ◀──────────────── "Here's…" ┤
   │                              │
   ├─ "Show me a caching one" ───▶│  (turn 2 — remembers turn 1)
   │                              │
   │ ◀────────── "Use @lru_cache"┤
   │                              │
   ├─ "What about timeout?" ─────▶│  (turn 3 — same stored session)
   │                              │
```

Set `new_session=true` to start fresh. Configurable per-call `timeout` (default 600s).

#### `dispatch_agent` — monitored async or truly detached

POSTs to the target's `/v1/runs` endpoint. With the default `delivery="callback"`, Herald opens an SSE stream and **automatically delivers** the terminal result to the current origin session. With `delivery="none"`, it starts the run without a listener or callback; use `check_dispatch` or `dispatch_status` later. Multiple dispatches in one turn run in parallel.

Auto-delivery is process-bound: the origin Hermes process must remain alive. The durable SQLite ledger retains call provenance, while the bounded JSON state cache retains live recovery data such as session handles and approvals. If that recovery cache becomes unwritable after a remote side effect, Herald still returns the remote handle and records the ledger edge with an explicit warning. Herald does not recreate an old SSE listener or retroactively inject a result into a dead session.

Detached calls receive a small `[HERALD ROUTING CONTEXT]` containing a generated `trace_id`, `edge_id`, and `hop_count`; callback calls receive it when a guarded chain opts into lineage. A forwarding agent preserves the trace and delivery mode, sets `parent_edge_id` to the incoming edge, and passes the incoming count as `parent_hop`; Herald performs the increment. Set optional `max_hops` on the first call to stop A→B→A ping-pong before the next edge exceeds the budget. Omit it for unlimited depth. This creates explicit graph lineage and an optional normal-flow loop brake without a Hermes core patch.

```
Your Agent                      Reviewer Profile
   │                                 │
   ├─ dispatch_agent ──────────────▶│ POST /v1/runs
   │                                 │ (task starts)
   │                                 │
   │  ◀───────────── SSE stream ────┤ /v1/runs/{id}/events
   │                                 │
   │  [ASYNC DELEGATION COMPLETE]    │
   │  result auto-delivered          │
```

If the SSE connection drops mid-run, the listener attempts to reconnect with exponential backoff:

| Before connection attempt | Delay | Cumulative backoff |
|---|---:|---:|
| 2 | 5s | 5s |
| 3 | 10s | 15s |
| 4 | 20s | 35s |
| 5 | 40s | 75s |

Before each reconnection, Herald checks the run status via `GET /v1/runs/{run_id}` — if the run already reached a terminal state while disconnected, the result is delivered immediately without resubscribing to the non-replayable SSE queue.

After five total failed connection attempts — the initial connection plus four reconnections, with ~75 seconds of cumulative backoff — the listener switches to authenticated `GET /v1/runs/{run_id}` polling. The polling fallback uses activity-based stall detection: each poll checks the target's `updated_at` timestamp, and if it has advanced since the last poll, the stall timer resets — just like activity events on the SSE path. A run whose target status keeps reporting activity should not hit the stall during a long polling session. If no activity is observed for 600 seconds (or 30 minutes while awaiting approval), a stall notification is delivered. Five consecutive polling transport failures also end local monitoring and tell you to reconcile with `check_dispatch`. If the stream is lost before a protected-command preview arrives, Herald refuses to approve the unseen command.

While SSE is healthy, activity events (message deltas, tool calls, reasoning, or approval state changes) reset the 600-second stall timer. While waiting for approval, a separate 30-minute approval timeout applies. A silent tool can still outlive the listener; use `check_dispatch` to reconcile target state.

#### `dispatch_agent` vs `dispatch_chat`

| | `dispatch_agent` | `dispatch_chat` |
|---|---|---|
| **Mode** | Async; callback-monitored or explicitly detached | Sync (blocks until reply) |
| **Session** | New target session per run; not reused by Herald | Persistent — continues the same stored conversation |
| **Context** | Self-contained task; no Herald continuity across runs | Prior session history, subject to target retention/context limits |
| **Result delivery** | `callback`: SSE auto-delivery with polling fallback; `none`: no callback | Return value — response is immediate |
| **Model selection** | Optional exact `model_routes` alias, verified before dispatch | Herald v1 intentionally uses the target's default runtime |
| **Parallelism** | Multiple dispatches run in parallel | Blocks the current turn |
| **Best for** | Independent background tasks | Multi-turn inter-agent dialogue |

**Use `dispatch_chat` for a back-and-forth conversation.** **Use `dispatch_agent` to fire off a self-contained task and keep working.**

### 🔀 Pillar 2: In-Process Subagents

Hermes core's `delegate_task` inherits the parent model — you can't pick a different model per subagent. The maintainers explicitly closed [issue #62731](https://github.com/NousResearch/hermes-agent/issues/62731) requesting this as "not-planned".

`delegate_subagent` fills that gap. It accepts a `model` parameter — bare names like `opus`, `gpt-5`, `glm` work, as do full `vendor/model` slugs. If model resolution selects a different provider, Herald uses the resolved provider credentials; the same resolved provider reuses the parent's credentials.

```python
# Review code with a fast model while parent uses a reasoning model
delegate_subagent(goal="Review auth.py for security issues", model="gemma4:31b")
```

The subagent runs in a daemon background thread with its own isolated context, terminal session, and toolset. Set `inherit_soul=true` to load the active parent profile's full `SOUL.md` as the child's primary identity. It is off by default and does not inherit conversation history, `USER.md`, memory, or project context files. Activity-based stall detection defaults to 600 seconds; tool events and streamed text reset it. An optional `interrupt_after_seconds` threshold requests cooperative interruption, stops waiting, and reports a timeout. It cannot guarantee immediate termination of a blocking provider or tool call. Local subagents are in-process and not durable across parent-process exit.

`delegate_subagent` works in the classic interactive CLI and desktop/TUI. Herald resolves the exact commissioning TUI session rather than scanning for another live agent. Gateway and API plugin-dispatch paths do not expose a parent-agent context, so this tool fails closed there; cross-profile dispatch and `llm_call` remain available on those surfaces.

> **Note:** Disable Hermes core's older global hard timeout so it can't preempt the plugin's per-call policy:
> ```yaml
> delegation:
>   child_timeout_seconds: 0
> ```

### 🧠 Pillar 3: Bare LLM Inference

Completely separate from dispatch. `llm_call` doesn't touch profiles, API servers, SSE streams, or subagents. It's a direct inference call through Hermes's provider routing — no agent loop, no tool schemas, no terminal access, no subagent overhead.

```python
# Classification
llm_call(messages=[{"role": "user", "content": "Is this email spam? Subject: 'WIN FREE iPhone'"}])

# Translation
llm_call(messages=[{"role": "user", "content": "Translate to French: 'Hello, how are you?'"}])

# JSON extraction
llm_call(
    messages=[{"role": "user", "content": "Extract name and age from: 'John Smith is 32'"}],
    json_mode=True
)
```

**When to use which tool:**

| Need | Tool |
|-----|------|
| Quick classification, translation, summarisation, scoring | `llm_call` |
| Agent with tools and terminal access, different model | `delegate_subagent` |
| Fire-and-forget task on a separate profile | `dispatch_agent` |
| Multi-turn conversation with a named agent | `dispatch_chat` |

---

## Tool Reference

### Dispatch Tools

#### `dispatch_agent`

Async dispatch to a named profile. Returns a `run_id` immediately; result auto-delivered via SSE when complete if the origin process is still running.

```python
dispatch_agent(
    profile="reviewer",           # required — target profile name
    message="Review this PR",     # required — task message
    instructions="You are a...",  # optional — system prompt override
    model="reviewer-fast",        # optional — verified model_routes alias
    delivery="none",              # optional — callback (default) or none
    trace_id="existing-trace",    # optional — preserve when forwarding
    parent_edge_id="incoming-edge", # optional — graph parent
    parent_hop=2,                  # optional — incoming hop; Herald increments
    max_hops=6                     # optional — omit for unlimited graph depth
)
```

- Multiple `dispatch_agent` calls in one turn run in parallel.
- `delivery="none"` is real fire-and-forget: it does not create a local listener and works from sessions that cannot accept async callbacks.
- Every call is recorded in the SQLite ledger. Detached forwarding can preserve lineage to form a directed call graph; optional `max_hops` terminates repeated forwarding and ping-pong loops.
- If `model` is specified, it must be an exact alias configured under the target's `model_routes`. The plugin verifies via `GET /v1/models` before starting the task. Unverifiable aliases fail closed.
- If the origin restarts, use `dispatch_status` to recover the handle and `check_dispatch` to query the target; the previous listener is not recreated.

#### `dispatch_chat`

Sync dispatch with persistent session. Subsequent calls continue the same conversation.

```python
dispatch_chat(
    profile="tutor",              # required — target profile name
    message="Explain decorators", # required — message to send
    instructions="You are a...",  # optional — system prompt for this turn
    new_session=False,            # optional — start fresh conversation
    timeout=60                    # optional — per-call timeout in seconds
)
```

- Calls to the same profile reuse the same target session until `new_session=True`.
- `instructions` applies to the current turn. Resend it on later turns if it remains required.
- Herald v1 always uses the target profile's default runtime on this path; a configured or supplied `model` override is rejected before the request is sent.
- Timeout resolution: per-call `timeout` → config `chat_timeout` (default 600s) → built-in default.

#### `delegate_subagent`

In-process subagent with per-call model override. Result auto-delivered on completion.

```python
delegate_subagent(
    goal="Summarise this codebase",  # required
    model="gemma4:31b",              # optional — model override
    context="The repo is at /path",  # optional — background info
    inherit_soul=True,                # optional — load parent profile SOUL.md
    toolsets=["web", "terminal"],    # optional — restrict toolsets
    stall_timeout_seconds=600,       # optional — activity-based stall timer
    interrupt_after_seconds=1800     # optional — cooperative interrupt threshold
)
```

### Management Tools

| Tool | What it does |
|------|-------------|
| `check_dispatch` | Poll the status of a single dispatched run |
| `collect_dispatches` | Batch-poll multiple dispatched runs at once |
| `dispatch_status` | Query durable call history, optionally reveal full task text, and show configured/observed directed topology |
| `cancel_dispatch` | Request cooperative target cancellation and suppress late local delivery |
| `ping_profile` | Liveness check — is the target profile's API server reachable? |
| `list_profile_models` | List models a target profile can safely accept for `dispatch_agent(model=...)` |

### Approval Relay

For autonomous agent meshes, configure trusted targets with Hermes's `smart` approval mode. Use `off`/YOLO only where the target's tools and execution environment are deliberately trusted or sandboxed. Manual relay is an advanced, model-mediated fallback for workflows that intentionally keep a human available; it is not required for normal Herald dispatch.

When the target Hermes instance leaves a protected command pending and emits an
`approval.request` event, Herald relays that request back to your session:

```
[DISPATCH APPROVAL REQUIRED]
Profile: reviewer
Run: abc123...
Herald approval ID: 4f8c...
Command: <redacted command preview>
Choices: once | session | always | deny
```

```python
approve_dispatch(
    run_id="abc123",
    profile="reviewer",
    approval_id="fresh-id-from-the-notice",
    choice="once",
    resolve_all=False,
)
```

- **Model-mediated relay:** the current implementation injects a pending request into the originating agent as a synthetic turn. The origin model may then call `approve_dispatch`; the original approval request is not presented directly to the human.
- **Target policy runs first:** in `smart` mode, Hermes may automatically approve or deny a flagged command without emitting a pending request, so nothing reaches Herald or the origin session. This is the recommended mode for autonomous agent-to-agent operation. `manual` mode makes the target wait, but does not by itself guarantee that the originating human receives a timely, usable prompt.
- **Scoped locally:** Herald binds the decision to the originating session, `{profile, run_id}`, and a fresh delivery nonce. Cross-session, cross-profile, and stale/replayed attempts are rejected before contacting the target. The displayed `approval_id` is Herald's local ownership/anti-replay token, not an opaque target-side approval-entry ID.
- **FIFO-bound at the target:** Hermes's current run-approval endpoint resolves pending commands FIFO, so Herald preserves that order locally. If several requests arrive, only the current head is shown; the next notice is promoted after the visible request resolves.
- **Defensive display:** the target redacts the command preview and Herald wraps it in explicit delimiters. Treat the preview as untrusted data and base the decision on the current FIFO head and fresh Herald `approval_id` shown in that notice.
- **Permanent scope is explicit:** `choice="always"` is refused unless the target supplies the approval `pattern_key`; that scope is shown on the human confirmation surface.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Origin Hermes Instance (your profile)                   │
│                                                          │
│  ┌───────────────┐    ┌───────────────────────────────┐  │
│  │ dispatch_chat │───▶│ POST /api/sessions/{id}/chat  │  │
│  │ (sync,        │    │ (persistent session —          │  │
│  │  back-and-    │    │  target remembers prior turns) │  │
│  │  forth)       │◀───│                               │  │
│  └───────────────┘    └───────────────────────────────┘  │
│                                                          │
│  ┌───────────────┐    ┌───────────────────────────────┐  │
│  │ dispatch_agent│───▶│ POST /v1/runs                  │  │
│  │ (async,       │    │ (task starts on target)        │  │
│  │  fire-and-    │    └───────────────────────────────┘  │
│  │  forget)      │           │                           │
│  └───────────────┘           ▼                           │
│                     ┌───────────────────────────────┐    │
│                     │ SSE Listener Thread            │    │
│                     │ • authenticated poll fallback │    │
│                     │ • no redirect credential leak │    │
│                     │ • stall detection              │    │
│                     │ • approval relay               │    │
│                     └──────────┬────────────────────┘    │
│                          │ result via                     │
│                          ▼ completion_queue                │
│                     ┌──────────────┐                     │
│                     │ Your Session │                     │
│                     │ (auto-delivery)│                    │
│                     └──────────────┘                     │
│                                                          │
│  ┌───────────────┐              ┌──────────────┐         │
│  │delegate_      │              │ llm_call     │         │
│  │subagent       │              │ (direct      │         │
│  │ (in-process   │              │  inference,  │         │
│  │  thread,      │              │  no dispatch)│         │
│  │  per-call     │              └──────────────┘         │
│  │  model)       │                                       │
│  └───────────────┘                                       │
└──────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────┐
│ Target Profile  │          │ Target Profile  │
│ reviewer (:8651)│          │ tutor (:8652)   │
│ (separate API   │          │ (separate API   │
│  server, own    │          │  server, own    │
│  model+memory)  │          │  model+memory)  │
└─────────────────┘          └─────────────────┘
```

### Reliability

- **Two persistence layers** — the bounded profile-local JSON state cache uses atomic replacement for live recovery; the uncapped SQLite ledger stores durable call provenance and supports concurrent readers/writers through WAL. Existing bounded JSON run history is imported into SQLite once when `dispatch_status` first sees it.
- **Share the ledger, not the state cache** — same-filesystem origins may point `ledger_file` at one database for a combined observed graph. Never point multiple origins at one JSON `state_file`.
- **Session and approval recovery** — stored chat session IDs and pending approval metadata are restored when the plugin registers after a restart.
- **Protected shared registries** — locks guard state-file updates, live listeners, chat session IDs, and pending approvals.
- **Credential-safe HTTP** — authenticated API and SSE requests reject redirects, preventing a target-controlled redirect from forwarding its bearer token to another origin.
- **Cooperative cancellation** — `cancel_dispatch` requests target interruption and suppresses stale local callback delivery; the target stops at a safe interruption point.
- **Zero core patches** — uses stdlib HTTP (`urllib`) and core delegation internals as importable library functions.

---

## Config

```yaml
hermes_herald:
  # Optional when Hermes cannot infer a named profile from HERMES_HOME.
  # origin_name: reviewer-orchestrator
  # Block self-dispatch/self-chat unless explicitly enabled.
  allow_self: false
  profiles:
    reviewer:
      url: http://localhost:8651
      api_key: ${REVIEWER_API_KEY}
      capabilities: [dispatch]  # this origin may delegate, but not chat
      # Optional async default: exact model_routes alias used by dispatch_agent.
      # Omit this if the profile will also be used with dispatch_chat.
      model: reviewer-fast
    tutor:
      url: http://localhost:8652
      api_key: ${TUTOR_API_KEY}
      capabilities: [dispatch, chat]
  # Optional: default timeout for dispatch_chat in seconds (default 600).
  chat_timeout: 900
  # Optional: defaults under the active profile's HERMES_HOME.
  # Do not share one custom state file across multiple origin processes.
  state_file: /custom/private/path/hermes-herald-runs.json
  # Durable SQLite history. Share this path across same-filesystem origins
  # when you want one observed network graph.
  ledger_file: /custom/private/path/hermes-herald-network.db
```

API keys support `${VAR_NAME}` environment interpolation. Prefer `hermes config set REVIEWER_API_KEY <value>` so the secret is written to the origin profile's `.env`; do not commit bearer tokens to `config.yaml`.

### Directed topology: forward, reverse, self, and "all agents"

Routes are **outbound and per origin**. Registering `ada` in Marie's config creates Marie→Ada; it does not create Ada→Marie. To allow the reverse direction, add Marie to Ada's own config. `capabilities` independently gates asynchronous `dispatch` and persistent `chat`; an omitted list preserves the v1-compatible default of both. There is no wildcard route to every registered Hermes profile. Herald infers the origin from standard profile homes; set `origin_name` when using a custom or containerized home that would otherwise appear as `custom`.

```yaml
# Marie's config: one-way delegation to Ada; no reverse authority implied.
hermes_herald:
  allow_self: false
  ledger_file: /srv/hermes/herald-network.db
  profiles:
    ada:
      url: http://127.0.0.1:8652
      api_key: ${ADA_API_KEY}
      capabilities: [dispatch]

# Ada's separate config: no Marie entry, so Ada cannot delegate back.
hermes_herald:
  ledger_file: /srv/hermes/herald-network.db
  profiles: {}
```

Self-routing requires **both** an explicit entry whose name matches the active profile and `allow_self: true`. This prevents accidental recursion. These controls govern Herald's tools at the origin; they are not a sandbox or target-side identity firewall. The bearer key remains the target API's actual authority, and a profile with other network/terminal tools may be able to call that API outside Herald.

There is no mandatory global layer limit. For a bounded async workflow, set `max_hops` on the initial dispatch and preserve it while forwarding. A call at `hop_count == max_hops` may finish normally but cannot create the next Herald edge in that trace. This catches A→B→A loops as repeated hops in either detached or callback mode. The context is model-carried rather than cryptographically signed, so it is an operational guardrail, not protection against a malicious agent that discards the trace or bypasses Herald.

`dispatch_status()` returns the current origin's declared outbound routes plus observed `origin_profile → target_profile` call counts from the ledger. Pass `include_messages=true` to retrieve full stored task text and instructions; it defaults to false because the ledger may contain sensitive briefs. The DB is created mode `0600` and never stores configured transport credentials. Task text is stored exactly as supplied, so do not put secrets in a brief.

### Cross-profile model routing

`dispatch_agent` sends no `model` field when neither the tool call nor the profile config specifies one, preserving the target's normal default. When a model is specified, it must be an exact alias configured under the target API server's `gateway.platforms.api_server.extra.model_routes`. The plugin authenticates to `GET /v1/models` and verifies the alias before starting the task. Arbitrary model strings, the advertised primary identity, or any unverifiable response fails closed.

Call `list_profile_models(profile=...)` before selecting an override — it returns the target's dispatchable aliases and their resolved models.

This is intentionally stricter than calling Hermes's native API directly: Herald v1 exposes only predeclared aliases for async dispatch. `dispatch_chat` uses the target profile's default runtime, and a profile-level model default makes that profile ineligible for chat.

### Trust boundaries and limitations

- **Target access is powerful.** A valid transport key can invoke a Hermes agent with whatever tools the target exposes to the `api_server` platform, including terminal access.
- **Herald does not provide TLS.** Use loopback for same-host profiles. For remote hosts, prefer a private network such as Tailscale or put the endpoint behind authenticated TLS; avoid exposing a plain HTTP listener to the public internet.
- **Targets need their own provider setup.** Herald never copies upstream model keys, skills, memory, or tool credentials from the origin.
- **Target output is untrusted input.** Async results are reinjected into the origin as a new turn. Use only trusted target profiles, treat hostile documents/web content they process as prompt-injection capable, and do not grant sensitive side effects based solely on a target's prose.
- **Manual remote approvals are model-mediated and fail closed.** The originating agent receives the notice and may propose `approve_dispatch`; Hermes still requires confirmation through the owning TUI/gateway surface before forwarding a positive choice. No available human surface, a stale request, timeout, or decline sends no approval to the target.
- **Approvals are session-owned.** Persisted runs remain visible within the origin profile for recovery, but a pending protected command can only be resolved from its originating session with the fresh `approval_id` delivered in that notice. Use separate Hermes profiles when different users must not see one another's run metadata.
- **Async delivery is not a durable queue.** Call provenance is durable, but an origin restart loses the active SSE listener. Recover with `dispatch_status` and `check_dispatch`, or choose `delivery="none"` when no callback is intended.
- **Ledger visibility is sensitive.** Full task text and instructions are stored locally for auditability. Keep the DB private and use `include_messages=true` deliberately. Configured bearer keys are never written to it, but secrets embedded by a caller in task text would be.
- **Bulk positive approval is refused.** `resolve_all=true` is available only with `choice="deny"`; positive commands must be reviewed one at a time.
- **Target restarts have their own semantics.** A target gateway restart can interrupt an active run even though the origin still knows its `run_id`.
- **Persistent chat is one session per configured profile name.** Use `new_session=True` to discard Herald's saved handle and start another conversation.

---

## Tests

```bash
cd tests
HERMES_HERALD_PLUGIN_DIR=../ HERMES_SOURCE_DIR=~/.hermes/hermes-agent \
  python3 -m pytest -v
```

151 tests covering: durable SQLite call provenance, legacy-history migration, private ledger permissions, run/chat post-side-effect recovery failures, detached and callback graph routing with optional hop budgets, directed route policy, stable origin attribution, SSE reconnection with exponential backoff, activity-based polling stall detection, transactional cancellation, session-owned nonce-bound approval relay, redirect credential isolation, profile-local recovery state, chat-session recovery, model resolution, bundled-skill registration, exact TUI parent-agent resolution, delegate_subagent async-delivery and cooperative-interrupt semantics, SOUL inheritance, llm_call validation, ping_profile, and more.

---

## License

MIT — see [LICENSE](LICENSE).

Built by Ben Kamholtz for [Hermes Agent](https://github.com/NousResearch/hermes-agent), created by [Nous Research](https://nousresearch.com).
<img src="assets/hero-banner.png" alt="Hermes Herald — Multi-Agent Dispatch for Hermes Agent" width="100%">

# Hermes Herald

<p>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Hermes%20Agent-compatible-8B5CF6" alt="Hermes Agent"></a>
  <img src="https://img.shields.io/badge/version-1.0.0-22C55E" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/tools-11-orange" alt="11 Tools">
</p>

> **Your agents should not just talk to you. They should be able to talk to each other.**

Hermes Herald turns one Hermes Agent into a **conductor of specialist agents and models**. It adds three capabilities that Hermes users otherwise have to wire together themselves:

| Herald adds | Why you would want it |
|---|---|
| **📡 A network of named Hermes agents** | Ask another profile a question, continue the conversation later, or send independent work to several specialists in parallel. Each target keeps its own identity, memory, skills, credentials, tools, and runtime. |
| **🔀 A model switchboard for subagents** | Spawn a local child with the model, context, SOUL identity, and tool access appropriate to this one task—without changing the parent agent. |
| **🧠 A lightweight inference lane** | Use a model for classification, extraction, translation, scoring, or JSON generation without paying for an agent loop and a large tool schema. |

These are independent. You can use Herald only for `llm_call`, only for model-selectable subagents, or build a multi-host network of persistent specialist profiles.

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
| A local child agent with a different model | `delegate_subagent` | `task_id` immediately; result returns asynchronously | In-process and non-durable; context, SOUL, and tools are controlled per call | Any model resolvable through the parent Hermes runtime |
| Classification, extraction, translation, scoring | `llm_call` | Text synchronously | No agent loop and no tools | Hermes provider/model routing |

**Use `dispatch_chat` when your next decision depends on the target’s reply.**

**Use `dispatch_agent` when the work is self-contained and the origin should remain available.**

```python
# Continue a real conversation with a named profile.
dispatch_chat(profile="tutor", message="Explain decorators with one example")
dispatch_chat(profile="tutor", message="Now adapt that example to cache results")

# Fan independent work out without blocking the origin.
dispatch_agent(profile="reviewer", message="Review the current diff. Do not edit.")
dispatch_agent(profile="researcher", message="Find authoritative sources for the design.")

# Choose a different local model and exactly what it inherits.
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

### 2. Prepare a target profile

```bash
hermes profile create tutor --description "A patient teaching specialist"
hermes -p tutor setup

HERALD_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
hermes -p tutor config set --force gateway.platforms.api_server.enabled true
hermes -p tutor config set --force gateway.platforms.api_server.extra.host 127.0.0.1
hermes -p tutor config set --force gateway.platforms.api_server.extra.port 8652
hermes -p tutor config set --force gateway.platforms.api_server.extra.key "$HERALD_KEY"

# Store the same transport key in the origin profile's .env.
hermes config set TUTOR_API_KEY "$HERALD_KEY"
unset HERALD_KEY
```

Start the target gateway:

```bash
hermes -p tutor gateway run
```

For a background service, use `hermes -p tutor gateway install` and `hermes -p tutor gateway start`.

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
      url: http://127.0.0.1:8652
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

### 4. Restart and verify

Start a fresh origin session and look for:

```text
hermes-herald: registered 11 tools
```

Then verify reachability and authenticated model discovery:

```python
ping_profile(profile="tutor")
list_profile_models(profile="tutor")
```

For local `llm_call` routing, omit `profile`. The result is a fail-closed
allowlist built from Hermes's explicit-only inventory for the calling profile:
the configured default plus every provider explicitly configured by the user.
Herald then keeps only concrete provider/model pairs that round-trip through
Hermes's resolver as a complete executable route. Virtual MoA routes, synthetic
picker identities that cannot round-trip, ambient or auto-discovered credentials,
and fallback-only routes are excluded:

```python
list_profile_models()
```

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

### Parallel or detached work

`dispatch_agent` starts a target `/v1/runs` task and returns a `run_id` immediately.

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

<details>
<summary><strong>Async monitoring and recovery details</strong></summary>

The listener consumes `/v1/runs/{run_id}/events`. If SSE drops, Herald checks authoritative run status before reconnecting, then retries with 5/10/20/40-second backoff. After five total failed connections, it switches to authenticated polling.

SSE activity and advancing target `updated_at` timestamps reset the normal 600-second stall timer. Approval waits use 30 minutes. Five consecutive polling transport failures end monitoring with instructions to reconcile using `check_dispatch`.

Auto-delivery is process-bound: restarting the origin loses its active listener. The durable SQLite ledger retains provenance and the bounded JSON cache retains live recovery data, but Herald does not recreate an old listener or inject a result into a dead session. Recover the handle with `dispatch_status`, then query the target with `check_dispatch`.

</details>

---

## Pillar 2 — A model switchboard for local subagents

Hermes core’s `delegate_task` is the right tool when the child should inherit the parent model. `delegate_subagent` adds **per-call model choice** and explicit inheritance controls.

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

`delegate_subagent` requires the live parent-agent context exposed by classic interactive CLI and desktop/TUI sessions. Gateway/API plugin-dispatch paths fail closed rather than guessing another session.

> If Hermes core’s global `delegation.child_timeout_seconds` is enabled, it can preempt Herald’s per-call policy. Set it to `0` or leave it unset when using Herald timeout controls.

---

## Pillar 3 — A lightweight inference lane

`llm_call` goes directly to one preflighted Hermes provider route without an
agent loop, tool schemas, terminal access, subagent overhead, or cross-provider
fallback.

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

With no routing arguments, `llm_call` pins the calling profile's configured
default provider and model. Before choosing an override, call
`list_profile_models()` and use an exact returned `{provider, model}` pair. A
provider-only override is accepted only for the active provider; selecting a
different provider requires an explicit model.

Prefer `configured_default`. Do not select another provider merely because it
advertises the same model; override the route only when the task or user
explicitly calls for that endpoint.

Model overrides are resolved through the same model-switch pipeline as Hermes's
`/model` command, but Herald then requires the resolved wire slug to appear in
the selected provider's authenticated model inventory **before the inference
request**. The inventory check may itself query the selected provider. A
model-only request stays on the active provider; it cannot silently
select OpenRouter from a vendor-prefixed slug. Family shorthand such as
`gpt-5.6` resolves to the active advertised family variant (`gpt-5.6-sol` here).
Unadvertised models and failed inference routes return a noisy error; Herald
does not retry them through OpenRouter, Nous Portal, or another provider.

Use exact provider IDs. In Hermes, bare `openai` is an alias for OpenRouter, so
Herald refuses that ambiguous alias: use `openai-codex` for ChatGPT Codex OAuth
or `openrouter` when OpenRouter is intentional. Results expose
`requested_model`, `resolved_provider`, and `resolved_model` separately from the
response-reported `model` and serving `provider` metadata.

---

## The 11 tools

| Tool | Purpose |
|---|---|
| `dispatch_chat` | Persistent synchronous conversation with streamed activity and verified target model selection |
| `dispatch_agent` | Async cross-profile run with callback or detached delivery |
| `delegate_subagent` | In-process child with per-call model and inheritance controls |
| `llm_call` | Bare model inference without an agent loop |
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

When `model` is supplied explicitly or configured on the Herald profile route, Herald:

1. authenticates to the target’s `GET /v1/models`;
2. requires an exact `model_routes` alias with a resolved target model;
3. refuses unknown, unverifiable, or primary-identity-only names before sending work;
4. records requested and resolved model provenance in the ledger.

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
 ├─ delegate_subagent ─ in-process model-selected child
 └─ llm_call ────────── direct provider inference

 Target profiles retain their own model routes, SOUL, skills, memory,
 credentials, tools, sessions, filesystem, and network location.
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

---

## Development and tests

```bash
cd tests
HERMES_HERALD_PLUGIN_DIR=../ HERMES_SOURCE_DIR=/path/to/hermes-agent \
  python3 -m pytest -v
```

The release suite currently contains **194 tests** covering streaming persistent chat, local and remote model-route discovery, fail-closed no-fallback inference, activity-aware stalls, subagent inheritance controls, async SSE recovery, polling fallback, transactional cancellation, session-owned deny-only approval relay, durable ledger migration, graph lineage and hop budgets, redirect credential isolation, exact TUI parent resolution, bare inference validation, and release contracts.

## License

MIT — see [LICENSE](LICENSE).

Built by Ben Kamholtz for [Hermes Agent](https://github.com/NousResearch/hermes-agent), created by [Nous Research](https://nousresearch.com).
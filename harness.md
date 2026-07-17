# Agentic AI Harness Engineering — A 4-Week Deep Curriculum

**One honest note first:** 30 days gets you complete conceptual coverage, a working harness built from raw HTTP calls (no framework crutches), and fluency in the vocabulary/tradeoffs senior interviewers probe. The judgment that separates senior from strong-junior — knowing *which* pillar to over-invest in for a given product — compounds over years. This plan front-loads every concept so nothing is missing; depth keeps building after day 30.

---

## Part 1 — The 14 Pillars (the map before the territory)

| # | Pillar | One-line definition |
|---|---|---|
| 0 | LLM & API mechanics | The raw substrate every harness sits on: tokens, roles, sampling, streaming |
| 1 | The agent loop | The control structure that turns one LLM call into autonomous multi-step behavior |
| 2 | Tool / function calling | How the model requests the world act on its behalf, and how you validate/execute that |
| 3 | Context engineering | What goes into the finite context window at each turn, and why |
| 4 | Memory systems | What survives *beyond* one context window, and how it's written/retrieved |
| 5 | Planning & decomposition | How a vague goal becomes an ordered sequence of tool-executable steps |
| 6 | Sandboxing & execution environments | Where model-generated actions actually run, and how they're contained |
| 7 | Permissions & safety models | Who/what approves an action before it's allowed to happen |
| 8 | Error recovery & self-correction | What happens when a step fails, and how the loop doesn't die or spiral |
| 9 | Sub-agent orchestration | When and how one agent becomes many, and how they don't step on each other |
| 10 | State management | What persists across turns, sessions, and process restarts |
| 11 | Streaming & interruption | How output arrives incrementally and how a user steers a running agent |
| 12 | Evaluation & observability | How you know the harness works, and where it breaks, with numbers not vibes |
| 13 | Domain specializations | How the same 12 pillars get reshaped for coding, browser, and research agents |

Dependency order matters: you can't design context engineering (3) before the loop (1) and tool calls (2) that fill it. You can't design planning (5) before memory (4) — plans need somewhere to live across steps. Sub-agent orchestration (9) needs context windows (3) and state (10) understood first, since isolation is the whole point. Evaluation (12) is listed last but you'll build a tiny eval harness in week 1 and grow it — flying blind for three weeks builds bad habits.

---

## Part 2 — Full Granular Topic Taxonomy

### Pillar 0 — LLM & API Mechanics (prerequisite layer)
- Tokenization: BPE mechanics, why token count ≠ word count, counting tokens for budgeting (tiktoken / provider-specific counters)
- Chat message format: system/user/assistant/tool roles; a multi-turn conversation is literally a growing array of role-tagged messages sent fresh each call — LLMs are stateless, **the harness is the state**
- Sampling parameters and their effect on agent behavior: temperature, top-p, top-k, stop sequences — why production agents run near-deterministic for tool-calling steps and higher temp only for open-ended reasoning
- Structured output, three flavors: native tool/function-calling (typed structured block), constrained decoding/grammars (mechanically forced schema), prompted-JSON (weakest, most fragile)
- Streaming protocol basics: SSE vs WebSockets, chunked token deltas, how a streamed response differs structurally from a blocking one
- Provider API shape differences that matter for portability: Anthropic Messages API vs OpenAI Chat Completions/Responses API vs self-hosted (vLLM, TGI, SGLang) — same concepts, different field names, different tool-call block formats
- Context window sizes and cost/latency scaling — why "just put everything in context" breaks down at scale

### Pillar 1 — The Agent Loop (core control architecture)
- Base loop: **observe → think → act → observe**
- ReAct pattern: interleaving a visible reasoning trace ("Thought:") with an action ("Action:") and the tool's result ("Observation:") in one transcript
- What counts as "one turn": one LLM call plus the tool executions it triggered, vs one user-visible chat message — conflating these is a common design bug
- Loop termination needs more than one condition: explicit "done" signal from the model, external verifier confirming goal state, max turn/token/cost budget as hard backstop, stuck-detector (Pillar 8) as soft backstop
- Formalizing the loop as a state machine: `PLANNING`, `AWAITING_MODEL`, `EXECUTING_TOOL`, `AWAITING_APPROVAL`, `ERROR`, `DONE`, with explicit transition rules — this is what ships in production, not a `while True`
- Deterministic (harness-decided) vs model-decided control flow — this split is one of the highest-leverage design decisions in a harness
- Synchronous vs asynchronous loop execution: blocking one-tool-at-a-time vs concurrent dispatch
- **Build now**: implement this loop in ~100 lines of raw Python with one tool (a calculator), no framework

### Pillar 2 — Tool / Function Calling Systems
- Tool schema anatomy: `name`, `description`, `input_schema` (JSON Schema: types, required fields, enums, nested objects)
- **The tool description is the interface** — this is prompt engineering, not documentation
- Native tool-calling (model emits a structured `tool_use` block) vs prompt-based/ReAct-style parsing (necessary without native support, and clarifies what native calling is built on)
- Parsing/validating tool calls: JSON Schema validation, and a **repair strategy** for malformed arguments (retry with the validation error fed back, not a silent crash)
- Serializing tool *results* back into context: format, truncation policy for huge outputs, a distinct format for error vs success results
- Parallel vs sequential tool calls: when the model can request multiple calls in one turn, how the harness executes them (`asyncio.gather` vs one-at-a-time), and when parallelism is unsafe
- Tool registry/dispatcher pattern: name→callable mapping, namespacing (`file.read` vs `web.search` vs `shell.exec`)
- Tool-set size and selection quality degrades past a few dozen tools; mitigations include namespacing, dynamic subsetting, and (2026 practice) **just-in-time tool discovery** via a "search tools" meta-tool
- Idempotent (read-only) vs side-effecting tools — feeds directly into Pillar 7 and Pillar 8
- **Build now**: a dispatcher plus 5 real tools (file read/write, shell exec, web search, calculator) with schema validation and repair-on-malformed-call

### Pillar 3 — Context Window & Context Engineering
- Context budget accounting: system prompt + tool schemas + history + retrieved content, against a hard model max — build a literal token counter/budget tracker first
- System prompt architecture as a checklist: identity/role, capabilities and boundaries, explicit tool-use policy, output-format contract, worked examples if needed
- Compaction techniques: rolling summarization, hierarchical/recursive summarization, selective forgetting
- Pruning: truncating/dropping old large tool outputs once used, keeping only the last N full turns verbatim
- "Just-in-time" vs "eager" context loading — a central fork: file tree + on-demand reads (scales to huge repos, more round trips) vs pre-loading everything (fewer round trips, blows budget fast)
- Prompt caching mechanics: cache breakpoints/prefix caching push you toward **append-only, stable-prefix context construction**
- Structured delimiters (XML tags, markdown headers) vs flat concatenation — improves the model's ability to distinguish tool results from instructions, which also matters for prompt-injection defense
- "Lost in the middle": instructions/facts in the middle of long context are attended to less reliably than at start/end
- Context overflow handling: what the harness actually does when a task is about to exceed the window mid-execution

### Pillar 4 — Memory Systems
- Working memory = the context window itself: bounded, short-term, gone at session end unless persisted
- Why context alone is insufficient: any task longer than one window, or any fact needing cross-session survival, needs external storage
- Cognitive-architecture vocabulary (useful in interviews): **episodic** (specific past events), **semantic** (distilled facts, no longer tied to when learned), **procedural** (learned routines/skills)
- RAG as agent memory: embeddings, vector stores, chunking strategy (fixed-size vs semantic/structural), similarity search, re-ranking
- Memory *write* policy: end-of-session summarization, explicit "remember this" triggers, importance-scored write
- Memory *read* policy: relevance vs recency-weighted vs blended, top-k selection, de-duplication
- File-based external memory — arguably the single most important *practical* pattern in modern coding harnesses: a persistent scratch file (`AGENTS.md`/`CLAUDE.md`, `todo.md`) read at session start and written to as the agent works, functioning as durable working memory with no vector database at all
- Memory consolidation: compressing many episodic entries into fewer semantic summaries over time
- Cross-session vs single-session memory, and multi-user memory isolation
- **Build now**: add a small embedding-backed retrieval store, and separately a scratchpad file the agent reads/writes across runs

### Pillar 5 — Planning & Task Decomposition
- Explicit upfront planning (inspectable, cheap to verify before spending tool calls) vs emergent/implicit planning (adapts as reality diverges, but harder to verify)
- Plan representations: flat todo list (simplest, what most production coding agents use), hierarchical task tree, full dependency DAG (for parallel/non-linear steps)
- Decomposition: recursive top-down breakdown until each leaf is directly tool-executable
- Named frameworks worth knowing by mechanism:
  - **Plan-and-Solve** — separate "make a plan" call from "execute step N" calls
  - **ReWOO** — generate the full plan and all tool calls up front, decoupled from intermediate observations
  - **ADaPT** — only decompose further if a first direct attempt fails
  - **Tree of Thoughts** — explore multiple candidate plan branches and self-evaluate under high uncertainty
- Re-planning triggers: a step invalidates an assumption, a tool fails repeatedly, or new information changes the goal
- Plan verification before execution: a cheap check that the plan actually addresses the goal, run before spending tool calls
- Planning-depth-vs-cost tradeoff: over-planning burns tokens/latency for trivial tasks — know when *not* to plan explicitly at all

### Pillar 6 — Sandboxing & Execution Environments
- Why sandbox at all: any tool executing model-generated code/commands is executing *untrusted* input, regardless of usual model behavior
- Isolation mechanisms, weakest to strongest: bare subprocess with limits, OS-level sandboxing (seccomp, chroot/namespaces), containers (Docker), microVMs (Firecracker, gVisor), full VMs
- Filesystem isolation: read-only mounts for reference material, a scratch/working directory, an explicit output directory copied out before the sandbox is destroyed
- Network isolation: default-deny egress with an explicit domain allowlist (proxy-based) is the dominant real-world pattern over blocklists, which are trivially incomplete
- Resource limits as first-class design: CPU quota, memory cap, wall-clock timeout per call, output size cap
- Reversibility design: dry-run/preview modes, snapshotting before risky actions, transactional execution where possible
- Ephemeral sandboxes (fresh container per task) vs persistent dev environments (long-lived, returned to across a multi-day task)
- **Build now**: move shell-exec into a real Docker container with resource limits, scratch dir, network disabled by default

### Pillar 7 — Permission & Safety Models
- Action risk tiering: read-only, mutating-but-reversible, irreversible/destructive — each needs a different default policy
- Human-in-the-loop approval gates: which tiers require confirmation, how the harness blocks and waits, how the UI surfaces "wants to do X, approve?" clearly enough to act on
- Autonomy levels as a spectrum: fully-manual, semi-autonomous (auto-approve safe tiers, gate risky ones), fully autonomous ("yolo mode") — a very common system-design interview question
- Allow-lists/deny-lists at multiple granularities: which tools exist at all, and within a tool, which parameters are permitted
- Prompt injection as a first-class threat: malicious instructions embedded in content the agent reads. Mitigations: privilege separation, treating retrieved content as *data* never *instructions*, sanitizing/flagging suspicious tool outputs
- Audit logging: every tool call, approval/denial, and decision, timestamped and attributable
- Rate limits and hard cost/action caps as a backstop independent of the model's own judgment
- **Build now**: approval-gate wrapper around write/delete tools, a full audit log, one deliberate prompt-injection resistance test

### Pillar 8 — Error Recovery & Self-Correction
- Error taxonomy: malformed tool call, tool execution failure, environment failure (timeout/OOM/crash), semantic failure (ran fine but didn't help), hallucinated success
- Retry strategy: immediate retry, exponential backoff, hard bounded retry count, circuit breakers
- **Reflexion** pattern specifically: after a failed attempt, have the model produce a short verbal self-critique of why it failed, feed that back as a new observation, retry — measurably beats blind retries
- Grounding/verification as discipline: after a code edit, actually run the tests; after a write, actually re-read the file — "no error" is necessary but not sufficient evidence of success
- Fallback strategies: alternate tool/approach on repeated failure, graceful degradation over total failure
- Stuck-state detection: catching repeated failing actions with no progress, escalating to a human rather than looping until budget exhausts

### Pillar 9 — Sub-Agent Orchestration & Multi-Agent Systems
- Three real reasons to decompose into multiple agents: **context isolation** (messy subtask exploration doesn't pollute the main context), **specialization** (narrowly-scoped agents are more reliable than one generalist), **parallelism** (independent subtasks run concurrently)
- Orchestrator-worker pattern, the dominant real-world topology: a lead agent decomposes and delegates, workers execute, results are synthesized
- What a good delegation contract requires: clear objective, expected output format, guidance on tools/sources, explicit task boundaries — vague objectives cause drift
- Context isolation mechanics: sub-agent gets a fresh context window scoped to its subtask; only a compact synthesized result crosses back, not the full transcript
- Fan-out/fan-in: launching N sub-agents concurrently, aggregating results
- Coordination problems: two sub-agents editing the same file, redundant work, an orchestrator over-spawning sub-agents for simple tasks — mitigated by explicit "effort scaling" rules in the orchestrator's prompt
- Cost reality: multi-agent research-style systems can burn ~10–15x the tokens of a single-agent conversation — this pattern is for tasks that decompose into independent, parallelizable threads, not a default upgrade
- **Model Context Protocol (MCP)**: host/client/server architecture, the three primitives a server exposes (tools, resources, prompts). As of mid-2026, MCP is donated to a vendor-neutral foundation and supported across every major provider
- **Build now**: a minimal orchestrator spawning 2–3 isolated sub-agent contexts, merging only summarized results; separately, stand up one real MCP server and connect as a client

### Pillar 10 — State Management Across Turns
- Session state vs durable/persistent state (must survive a restart, disconnect, or resumed-tomorrow task)
- Serializing full agent state — history, tool-call log, plan/todo state, memory references — so a long task can pause and resume exactly
- Checkpointing granularity: after every tool call (safest, most overhead) vs every N turns vs time-based
- Idempotency of state transitions: replaying from a checkpoint must not double-execute a side-effecting action
- Concurrency control for shared state: new user input mid-loop, or parallel sub-agents touching the same file
- State store choices: in-memory (fastest, gone on crash), SQLite/local file (durable, single-machine), external KV/document store (multi-process/multi-machine)

### Pillar 11 — Streaming & Interruption Handling
- Token-level streaming to the UI: SSE/WebSocket delivery, incremental rendering
- Streaming *tool-call arguments*: parsing partial/incomplete JSON gracefully until the call completes
- Clean interruption/cancellation: aborting generation or a running tool without leaving inconsistent state (half-written file, hung subprocess)
- Steering mid-task: queue new input for next iteration vs inject immediately — both legitimate, different UX
- Concurrency model: single-threaded async event loop vs multi-process workers, backpressure handling
- UX signaling tied directly to the Pillar-1 state machine, not a separately-maintained guess

### Pillar 12 — Evaluation & Observability
- Tracing as non-negotiable foundation: log every LLM call, every tool call (input/output/latency/cost), every decision point, as structured data
- Tooling landscape: OpenTelemetry conventions for LLM/agent tracing, dedicated platforms (LangSmith, Langfuse, W&B Weave, native provider tracing)
- Building an eval harness: tasks with *verifiable* success criteria, a held-out set you don't tune against
- Core metrics: task success rate, steps-to-completion, tool-call efficiency, cost per task, latency percentiles (p50/p95), human-preference win-rate
- LLM-as-judge: rubric design, calibration against human ratings, known biases (length bias, self-style bias, run-to-run inconsistency)
- Public benchmarks as reference designs for your own internal evals: **SWE-bench Verified**, **GAIA**, **WebArena**/**OSWorld**, **τ²-Bench**, **Terminal-Bench**, **METR's time-horizon/HCAST** work — study each one's task format and verification method, not just the leaderboard number
- Regression testing on every meaningful prompt/model change, not just at the end
- A/B testing and gradual rollout for production changes; failure clustering by root cause to fix the highest-leverage thing first

### Pillar 13 — Domain Specializations (same 12 pillars, different tools and environment)
- **Coding agents**: repo-scale context (on-demand file reads, AST/symbol-aware search), diff/patch editing vs whole-file rewrites, test-execution as the grounding loop, git-aware checkpoints/branches
- **Browser/computer-use agents**: raw pixel+coordinate actions vs semantic accessibility-tree actions (generally more robust), screenshot-based visual verification, dynamic-content handling as a distinct error class
- **Research agents**: iterative search→read→synthesize loops, source credibility weighting, citation tracking, map-reduce summarization to avoid context flooding from long sources
- The unifying insight: every domain is the same 12-pillar skeleton wearing a different tool set and execution environment — this cross-domain fluency is exactly what senior harness roles select for

---

## Part 3 — The 4-Week Schedule

**Week 1 — Foundations, the Loop, Tool Calling**
- Days 1–2: Pillar 0 — raw API calls, your own token counter, a working streaming response
- Days 3–4: Pillar 1 — bare ReAct loop, ~100 lines, one tool, explicit state machine
- Days 5–6: Pillar 2 — tool registry with 5 real tools, schema validation, repair path
- Day 7: Integrate into one working CLI agent; write 5 hand-crafted pass/fail test tasks
- *Reading*: ReAct, Toolformer, Anthropic's "Building Effective Agents"

**Week 2 — Context, Memory, Planning**
- Days 8–9: Pillar 3 — context budget tracker, rolling summarization, structured system prompt
- Days 10–11: Pillar 4 — embedding-backed retrieval store + scratchpad file
- Days 12–13: Pillar 5 — explicit plan generation, persistent todo-list, re-planning on failure
- Day 14: Integrate, refactor, expand eval suite to ~10 tasks
- *Reading*: Reflexion, ADaPT, Plan-and-Solve, Generative Agents

**Week 3 — Sandboxing, Permissions, Recovery, Sub-Agents**
- Days 15–16: Pillar 6 — real Docker sandbox, resource limits, network off by default
- Day 17: Pillar 7 — approval gates, audit log, prompt-injection resistance test
- Days 18–19: Pillar 8 — structured retry, self-critique-then-retry, stuck-loop escalation
- Days 20–21: Pillar 9 — minimal orchestrator with isolated sub-agent contexts; stand up an MCP server and connect as client
- *Reading*: MCP spec/docs, Anthropic's multi-agent research system post, SWE-agent's ACI writeup

**Week 4 — State, Streaming, Evaluation, Capstone**
- Day 22: Pillar 10 — serialized state to SQLite, checkpoint every tool call, verify no double-execution on resume
- Days 23–24: Pillar 11 — full streaming, clean mid-run cancellation, partial tool-call JSON parsing
- Days 25–27: Pillar 12 — full structured tracing, 15–20 task eval suite (containerized + programmatic checks), success rate/cost/latency, then confirm a deliberate prompt change is caught as a regression
- Day 28+: Capstone polish — architecture README, recorded demo, optionally go deeper into one Pillar 13 specialization

---

## Part 4 — Canonical Reading List

**Papers**: Yao et al. *ReAct* (P1) · Schick et al. *Toolformer* (P2) · Shinn et al. *Reflexion* (P8) · Wang et al. *Plan-and-Solve* (P5) · Xu et al. *ReWOO* (P5) · *ADaPT* (P5) · Yao et al. *Tree of Thoughts* (P5) · Park et al. *Generative Agents* (P4) · Jimenez et al. *SWE-bench*/*SWE-bench Verified* (P12/13) · Mialon et al. *GAIA* (P12) · Zhou et al. *WebArena* (P12/13) · Barres et al. *τ²-Bench* (P12) · Merrill et al. *Terminal-Bench* (P12)

**Blog posts**: Anthropic's ["Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents) (shared field vocabulary, when *not* to build an agent) · Anthropic's ["How we built our multi-agent research system"](https://www.anthropic.com/engineering/multi-agent-research-system) (P9, concrete cost multipliers and prompting failures) · the current [MCP spec](https://modelcontextprotocol.io) — a major revision finalizes July 28, 2026, so read the live spec · Simon Willison's blog for ongoing, concrete writing on tool use and prompt injection

---

## Part 5 — Codebases to Read Line-by-Line

- **mini-SWE-agent** — ~100-line core agent class, strong SWE-bench Verified score; the best single codebase for the irreducible minimum of a coding agent loop
- **SWE-agent** — study its Agent-Computer Interface (ACI) concept: a purpose-built, agent-friendly command interface instead of a raw shell
- **OpenHands** — event-stream architecture (`Agent → Actions → Environment → Observations → Agent`), Docker-per-session sandboxing
- **Aider** — diff-based, git-native editing; commits as natural checkpoints
- **smolagents** (Hugging Face) — agents write and execute actual Python as tool calls instead of JSON blocks
- **browser-use** — reference for accessibility-tree-driven browser action design
- [**best-of-Agent-Harnesses**](https://github.com/RyanAlberts/best-of-Agent-Harnesses) — actively updated map of 100+ projects by which harness concern they address

---

## Part 6 — Capstone Project Spec

A from-scratch CLI coding harness demonstrating every pillar: formalized loop with explicit states (1) · 6+ validated tools (2) · context budgeting with compaction (3) · persistent scratchpad memory (4) · explicit planning with re-planning (5) · Docker sandbox, network off by default (6) · approval gates + audit log (7) · structured retry + self-critique + stuck-loop escalation (8) · one orchestrated sub-agent (9) · checkpointed session resume (10) · streamed output with clean cancellation (11) · full tracing + scored eval suite with a demonstrated regression catch (12). Not a framework you configured — a system you can explain at every layer because you built every layer.

---

## Part 7 — What Top AI Labs Actually Probe

- **System design**: "design a harness for X" — a strong answer touches most pillars and justifies tradeoffs rather than reciting a framework's default
- **Live implementation**: a tool-calling loop from scratch, including streaming and partial-JSON parsing
- **Tradeoffs**: autonomy vs safety, context-budget allocation under pressure, when multi-agent orchestration earns its cost multiplier
- **Trace debugging**: given a failed transcript, classify the root cause against the Pillar 8 taxonomy — practice on your own capstone's failures
- **Currency**: know the current state of MCP, current open-source coding-agent architectures, and the current benchmark landscape — reciting 2023-era patterns as state of the art is itself a signal

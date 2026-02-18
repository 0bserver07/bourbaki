# Bourbaki Architecture

An AI agent for mathematical reasoning and theorem proving. Named after Nicolas Bourbaki, the collective pseudonym of mathematicians who wrote foundational treatises.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         TUI (Bun + React + Ink)                      │
│                                                                      │
│  Input ──→ CLI ──→ useAgentRunner ──→ fetch POST /query              │
│                         │                    │                       │
│                   useModelSelection     SSE stream                   │
│                   useInputHistory       ← AgentEvents                │
│                         │                    │                       │
│            ProviderSelector            HistoryItemView               │
│            ModelSelector               ProofDisplay                  │
│            ApiKeyPrompt                EventListView                 │
│                                        ReasoningIndicator            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP + SSE
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Python Backend (FastAPI + Pydantic AI)             │
│                                                                      │
│  Routes ──→ Agent Core ──→ Pydantic AI agent.iter()                  │
│                │                                                     │
│           Scratchpad (dedup + limits)                                 │
│                │                                                     │
│  ┌─────────┬──┴────┬──────────┬──────────┬──────────┬──────────┐    │
│  │SymPy    │Lean 4 │OEIS      │arXiv     │Exa       │Skills    │    │
│  │native   │subproc│httpx     │httpx     │httpx     │SKILL.md  │    │
│  └─────────┴───────┴──────────┴──────────┴──────────┴──────────┘    │
│                                                                      │
│  Sessions ──→ .bourbaki/sessions/                                    │
│  Autonomous ──→ Strategy queue + checkpoint/resume                   │
│  Problems ──→ 13 classic theorems                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Backend (Python)

### Tech Stack

- **Runtime:** Python 3.11+
- **Framework:** FastAPI with uvicorn
- **Agent:** Pydantic AI (`agent.iter()` for node-by-node control)
- **Streaming:** SSE via `sse-starlette`
- **Settings:** Pydantic Settings (loads from `.env`)
- **HTTP Client:** httpx (async, for external APIs)
- **Math:** SymPy (native symbolic computation)
- **Verification:** Lean 4 + Mathlib (async subprocess)

### Directory Layout

```
backend/bourbaki/
├── main.py                    # FastAPI app factory, CORS, router registration
├── config.py                  # Pydantic Settings, API key export
├── events.py                  # AgentEvent models (SSE wire format)
├── agent/
│   ├── core.py                # run_agent() — Pydantic AI agent loop
│   ├── context.py             # AgentDependencies (injected into tools)
│   ├── scratchpad.py          # Tool call tracking, limits, dedup
│   ├── prompts.py             # System prompt + iteration prompt builders
│   └── event_mapper.py        # Helper factories for AgentEvent creation
├── tools/
│   ├── symbolic_compute.py    # 30+ SymPy operations
│   ├── lean_prover.py         # Lean 4 subprocess
│   ├── sequence_lookup.py     # OEIS API + 15 builtin sequences
│   ├── paper_search.py        # arXiv API
│   ├── web_search.py          # Exa API
│   └── skill_tool.py          # Loads SKILL.md instructions
├── skills/
│   ├── loader.py              # YAML frontmatter parser
│   └── registry.py            # Multi-source discovery + cache
├── sessions/
│   ├── manager.py             # CRUD, persistence, token tracking
│   └── context_compactor.py   # LLM-based conversation summarization
├── autonomous/
│   ├── search.py              # Main search loop + checkpoint/resume
│   ├── strategies.py          # 18 strategies, priority queue, dead-end DB
│   ├── modal_runner.py        # Strategy execution via Pydantic AI
│   └── progress.py            # ProgressReport model
├── problems/
│   └── database.py            # 13 classic theorems with metadata
└── server/routes/
    ├── health.py              # GET /health
    ├── query.py               # POST /query (SSE)
    ├── compute.py             # POST /compute
    ├── prove.py               # POST /prove
    ├── search.py              # POST /search/sequence, /search/paper
    ├── export.py              # POST /export
    ├── sessions.py            # CRUD /sessions
    ├── problems.py            # GET /problems
    ├── autonomous.py          # Autonomous proof search control
    └── skills.py              # GET /skills
```

---

## API Endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok", "python": true, "lean": bool}` |

### Agent Query

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Main agent endpoint. Returns SSE stream of `AgentEvent` objects |

**Request:**
```json
{
  "query": "Prove that sqrt(2) is irrational",
  "model": "openrouter:deepseek/deepseek-r1-0528:free",
  "model_provider": null,
  "session_id": "abc-123",
  "chat_history": [{"role": "user", "content": "..."}]
}
```

**Response:** Server-Sent Events stream. Each event is one of the types defined in [SSE Event Types](#sse-event-types).

### Symbolic Computation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/compute` | Direct SymPy computation (bypasses agent) |

**Request:**
```json
{
  "operation": "factor_polynomial",
  "expression": "x^4 - 1",
  "variable": "x",
  "from_val": null,
  "to_val": null,
  "point": null,
  "matrix": null,
  "matrix2": null,
  "order": 6
}
```

**Response:**
```json
{
  "success": true,
  "result": "(x - 1)*(x + 1)*(x**2 + 1)",
  "latex": "(x - 1)(x + 1)(x^{2} + 1)",
  "numeric": null,
  "duration": 12
}
```

### Lean Verification

| Method | Path | Description |
|--------|------|-------------|
| POST | `/prove` | Run Lean 4 code for verification |

**Request:**
```json
{
  "code": "theorem foo : 1 + 1 = 2 := by norm_num",
  "mode": "check",
  "timeout": 30
}
```

**Response:**
```json
{
  "success": true,
  "goals": [],
  "proofComplete": true,
  "errors": null,
  "rawOutput": "...",
  "codeUsed": "...",
  "duration": 1500
}
```

### Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search/sequence` | OEIS sequence lookup |
| POST | `/search/paper` | arXiv paper search |

**Sequence request modes:** `"identify"` (match terms, requires 3+), `"search"` (text query), `"get"` (by ID like A000045)

**Paper request modes:** `"search"` (keyword search), `"get"` (by arXiv ID)

### Export

| Method | Path | Description |
|--------|------|-------------|
| POST | `/export` | Export proof as LaTeX, Lean, or Markdown |

**Request:**
```json
{
  "title": "Irrationality of sqrt(2)",
  "statement": "...",
  "proof": "...",
  "lean_code": "...",
  "sympy_computations": [],
  "format": "latex"
}
```

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/sessions` | List all sessions (most recent first) |
| POST | `/sessions` | Create new session (body: `{"model": "..."}`) |
| GET | `/sessions/{id}` | Load session by ID |
| DELETE | `/sessions/{id}` | Delete session |
| GET | `/sessions/{id}/messages` | Get messages for context restoration |

Sessions are stored as JSON files in `.bourbaki/sessions/{id}.json`. Token count is estimated (`len(text) // 4`) and auto-compaction triggers at 100k tokens, summarizing older messages while keeping the 6 most recent.

Max 50 sessions are retained (oldest auto-deleted).

### Problems

| Method | Path | Description |
|--------|------|-------------|
| GET | `/problems` | List problems. Query params: `domain`, `technique`, `min_difficulty`, `max_difficulty`, `famous` |
| GET | `/problems/random` | Random problem. Query params: `domain`, `difficulty`, `technique` |
| GET | `/problems/{id}` | Get specific problem by ID |

**13 built-in problems** across Number Theory (6) and Combinatorics (4+):

| ID | Title | Difficulty | Technique |
|----|-------|-----------|-----------|
| sum-of-integers | Sum of first n integers | 1 | induction |
| sum-of-squares | Sum of squares formula | 2 | induction |
| euclid-primes | Infinitely many primes | 2 | contradiction |
| sqrt2-irrational | Irrationality of sqrt(2) | 2 | contradiction |
| bezout-identity | Bezout's identity | 3 | strong-induction |
| fermat-little | Fermat's little theorem | 3 | induction |
| pigeonhole-basic | Basic pigeonhole | 1 | pigeonhole |
| handshaking-lemma | Handshaking lemma | 2 | counting |
| ramsey-r33 | R(3,3) = 6 | 3 | pigeonhole + cases |
| erdos-ko-rado | Erdos-Ko-Rado theorem | 4 | counting + contradiction |

### Autonomous Proof Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/autonomous/start` | Start long-running proof search |
| POST | `/autonomous/pause` | Pause current search |
| POST | `/autonomous/resume` | Resume from checkpoint |
| GET | `/autonomous/progress` | Get progress report |
| GET | `/autonomous/insights` | Get accumulated insights |

**Start request:**
```json
{
  "problem": {"id": "sqrt2-irrational", "statement": "..."},
  "max_iterations": 100,
  "max_hours": 4.0,
  "strategies": ["direct-proof", "contradiction"],
  "checkpoint_interval": 10
}
```

The search engine cycles through 18 strategies with dynamic prioritization, tracks dead ends (max 3 attempts per strategy), and checkpoints progress to `.bourbaki/progress/`.

### Skills

| Method | Path | Description |
|--------|------|-------------|
| GET | `/skills` | List all available proof technique skills |

Returns: `[{"name": "proof-by-induction", "description": "...", "source": "builtin"}, ...]`

---

## SSE Event Types

Events are streamed from `POST /query` as Server-Sent Events. Each line is `data: <json>`.

| Type | Fields | Description |
|------|--------|-------------|
| `thinking` | `message` | Agent reasoning step |
| `tool_start` | `tool`, `args` | Tool execution beginning |
| `tool_end` | `tool`, `args`, `result`, `duration_ms` | Tool execution completed |
| `tool_error` | `tool`, `error` | Tool execution failed |
| `tool_limit` | `tool`, `warning`, `blockReason`, `blocked` | Tool rate limit hit |
| `answer_start` | — | Agent beginning final answer |
| `done` | `answer`, `toolCalls[]`, `iterations` | Query complete |
| `checkpoint` | `checkpointId`, `iteration`, `reason`, `filepath`, `message` | Autonomous search checkpoint |
| `resume` | `checkpointId`, `iteration` | Resumed from checkpoint |

---

## Tools

The agent has 6 tools available during query processing. Each tool is guarded by the scratchpad (default limit: 3 calls per tool per query, Jaccard similarity dedup at 0.7 threshold).

### symbolic_compute

Native SymPy computation with 30+ operations.

**Operations by category:**

| Category | Operations |
|----------|------------|
| Number Theory | `factor_integer`, `prime_factors`, `is_prime`, `divisors`, `euler_phi`, `gcd`, `lcm`, `mod`, `mod_inverse` |
| Algebra | `factor_polynomial`, `simplify`, `expand`, `solve`, `evaluate` |
| Series | `sum_series`, `product_series`, `limit` |
| Calculus | `derivative`, `integral`, `taylor_series` |
| Matrix | `matrix_mult`, `determinant`, `eigenvalues`, `matrix_inverse`, `row_reduce`, `characteristic_polynomial`, `minimal_polynomial` |
| Analysis | `fourier_series`, `laplace_transform` |

### lean_prover

Runs Lean 4 code via async subprocess. Writes temp files to `.bourbaki/lean-temp/`, runs `lean` (or `lake env lean` as fallback), parses errors and goals from output. Default timeout: 30 seconds.

### sequence_lookup

OEIS sequence identification and lookup. Has 15 built-in sequences (Fibonacci, primes, powers of 2, squares, triangular numbers, factorials, Catalan numbers, etc.) as fast fallback. Falls through to OEIS API at `https://oeis.org/search`.

### paper_search

arXiv paper search. Supports keyword search and fetch-by-ID. Covers 14 math categories (math.NT, math.CO, math.AG, math.CA, math.LO, math.PR, math.GR, math.AT, math.RT, math.FA, math.DG, math.AP, math.OA, math.QA). Truncates abstracts to 300 characters.

### web_search

Exa API for academic and general web search. Categories: `"research paper"`, `"tweet"`, `"company"`, `"news"`, `"github"`, `"pdf"`. Requires `EXASEARCH_API_KEY`.

### skill_invoke

Loads proof technique instructions from SKILL.md files. Returns the full instruction text for the agent to follow. Skills are discovered from three directories with increasing precedence: builtin (`src/skills/`) < user (`~/.bourbaki/skills/`) < project (`.bourbaki/skills/`).

---

## Scratchpad

The scratchpad tracks tool usage per query to prevent waste:

- **Call limits:** Each tool can be called at most `tool_call_limit` times per query (default: 3)
- **Deduplication:** Jaccard similarity > 0.7 between queries to the same tool triggers a warning or block
- **Last-call warning:** When a tool has 1 call remaining, the agent is warned
- **Skill tracking:** Records which skills have been invoked to prevent re-loading

---

## Skills (21 Proof Techniques)

SKILL.md files contain structured instructions for proof techniques. Each has YAML frontmatter (name, description) and markdown body with steps, tool usage guidance, output format, and common patterns.

| Category | Skills |
|----------|--------|
| Basic | proof-by-induction, strong-induction, direct-proof, proof-by-contradiction, pigeonhole-argument, counting-argument |
| Analysis | epsilon-delta-proof, convergence-test, sequence-limit, inequality-chain |
| Geometry | coordinate-proof, synthetic-construction, transformation-proof |
| Algebra | group-homomorphism, ring-ideal-proof, polynomial-proof |
| Advanced | extremal-argument, probabilistic-method, conjecture-exploration, formalize-informal-proof, explain-proof |

---

## Autonomous Proof Search

Long-running proof search with 18 strategies:

| Strategy | Priority | Type |
|----------|----------|------|
| direct-computation | 100 | Compute |
| direct-proof | 90 | Logic |
| simple-induction | 85 | Induction |
| strong-induction | 80 | Induction |
| pigeonhole | 85 | Combinatorics |
| double-counting | 80 | Combinatorics |
| structural-induction | 75 | Induction |
| generalized-pigeonhole | 75 | Combinatorics |
| contradiction | 70 | Logic |
| infinite-descent | 65 | Number Theory |
| case-analysis | 60 | Logic |
| algebraic-manipulation | 55 | Algebra |
| extremal-principle | 50 | Optimization |
| similar-problems | 45 | Meta |
| probabilistic-method | 40 | Combinatorics |
| generalize | 35 | Meta |
| specialize | 30 | Meta |
| counterexample-search | 25 | Meta |

**Features:**
- Dynamic prioritization based on problem domain and tags
- Dead-end tracking (max 3 attempts per strategy before marking exhausted)
- Checkpoint/resume to `.bourbaki/progress/`
- Parallel strategy execution (up to 5 concurrent via `asyncio.gather`)
- Insight accumulation across attempts

---

## Frontend (TUI)

### Tech Stack

- **Runtime:** Bun
- **Framework:** React 19 + Ink 6 (terminal UI)
- **Language:** TypeScript
- **Deps:** dotenv, ink-spinner, ink-text-input

### Directory Layout

```
src/
├── index.tsx                  # Entry point: render(<CLI />)
├── cli.tsx                    # Main CLI component, command parser
├── constants.ts               # DEFAULT_PROVIDER, DEFAULT_MODEL
├── theme.ts                   # Color palette (purple/green/cyan)
├── components/
│   ├── index.ts               # Re-exports
│   ├── Intro.tsx              # Welcome banner with ASCII art
│   ├── Input.tsx              # Text input with cursor, slash commands, tab complete
│   ├── ModelSelector.tsx      # Provider + model selection UI, PROVIDERS registry
│   ├── ApiKeyPrompt.tsx       # API key confirmation + input
│   ├── ProofDisplay.tsx       # Markdown-rendered proof output
│   ├── AgentEventView.tsx     # Tool call / thinking event rendering
│   ├── HistoryItemView.tsx    # Single query-response turn
│   ├── ReasoningIndicator.tsx # Animated status indicator
│   ├── DebugPanel.tsx         # Debug log viewer
│   └── CursorText.tsx         # Text with block cursor
├── hooks/
│   ├── useAgentRunner.ts      # SSE connection to /query, history state
│   ├── useModelSelection.ts   # Provider → model → API key flow
│   ├── useInputHistory.ts     # Up/down arrow command history
│   ├── useTextBuffer.ts       # Low-level text editing + cursor
│   └── useDebugLogs.ts        # Debug log subscription
├── agent/
│   ├── types.ts               # AgentEvent union, AgentConfig, Message
│   └── sessionManager.ts      # TUI-side session tracking
├── utils/
│   ├── config.ts              # Read/write .bourbaki/settings.json
│   ├── env.ts                 # API key checking + saving to .env
│   ├── logger.ts              # Debug logger with subscription
│   ├── math-format.ts         # LaTeX → Unicode (170+ patterns)
│   ├── markdown-table.ts      # Markdown tables → box-drawing
│   ├── input-utils.ts         # Cursor navigation helpers
│   ├── thinking-verbs.ts      # 73 random verbs for status display
│   ├── ollama.ts              # Fetch local Ollama model list
│   └── long-term-chat-history.ts  # Persistent chat history stack
└── skills/                    # 21 SKILL.md files (proof techniques)
```

### Model Selection

Models are defined in `src/components/ModelSelector.tsx` in the `PROVIDERS` array. Each provider has a `providerId`, `displayName`, and list of `models`.

The `/model` TUI command triggers a multi-step flow:

```
idle → provider_select → model_select → [api_key_confirm → api_key_input] → idle
```

When a model is selected, it's stored as `provider:model` (e.g., `openrouter:deepseek/deepseek-r1-0528:free`) and persisted to `.bourbaki/settings.json`.

### Available Providers and Models

| Provider | Provider ID | Models | API Key Env Var |
|----------|------------|--------|-----------------|
| OpenRouter | `openrouter` | openrouter/pony-alpha, openrouter/aurora-alpha, arcee-ai/trinity-large-preview:free, stepfun/step-3.5-flash:free, qwen/qwen3-next-80b-a3b-instruct:free, z-ai/glm-4.5-air:free, qwen/qwen3-coder:free, deepseek/deepseek-r1-0528:free | `OPENROUTER_API_KEY` |
| Ollama Cloud | `ollama-cloud` | qwen3-coder:480b-cloud, kimi-k2-thinking:cloud, kimi-k2:1t-cloud, deepseek-v3.1:671b-cloud, minimax-m2:cloud, glm-4.6:cloud, qwen3-vl:235b-instruct-cloud, qwen3-vl:235b-cloud, gpt-oss:120b-cloud, gpt-oss:20b-cloud | `OLLAMA_CLOUD_API_KEY` |
| OpenAI | `openai` | gpt-5.2, gpt-4.1 | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | claude-sonnet-4-5, claude-opus-4-5 | `ANTHROPIC_API_KEY` |
| Google | `google` | gemini-3-flash-preview, gemini-3-pro-preview | `GOOGLE_API_KEY` |
| xAI | `xai` | grok-4-0709, grok-4-1-fast-reasoning | `XAI_API_KEY` |
| Ollama (Local) | `ollama` | Dynamic (fetched from local Ollama API) | None |

### TUI Commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/model` | Change model/provider |
| `/sessions` | List saved sessions |
| `/sessions <id>` | Restore a session |
| `/restore <id>` | Restore a session |
| `/new` | Start fresh session |
| `/debug` | Toggle debug panel |
| `/problems` | List available problems |
| `/prove <id>` | Start autonomous proof search |
| `/pause` | Pause proof search |
| `/progress` | Show proof search progress |
| `/skills` | List available proof techniques |
| `/export [format]` | Export last proof (latex/lean/markdown) |
| `exit` / `quit` | Exit CLI |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Submit query |
| `Shift+Enter` | Newline (multi-line input) |
| `Up/Down` | Navigate command history |
| `Ctrl+A` | Move to line start |
| `Ctrl+E` | Move to line end |
| `Option+Left/Right` | Move by word |
| `Option+Backspace` | Delete word backward |
| `Tab` | Autocomplete slash command |
| `Escape` | Cancel selection or running query |
| `Ctrl+C` | Cancel or exit |

### Data Flow

```
User types query
  ↓
handleSubmit() in cli.tsx
  ↓ (not a /command)
runQuery(text) via useAgentRunner
  ↓
buildModelStr() → "openrouter:openrouter/pony-alpha"
  ↓
ensureSession() → POST /sessions (if no session yet)
  ↓
fetch(POST /query, {query, model, session_id})
  ↓
SSE stream begins
  ↓
handleEvent() dispatches:
  thinking  → update ReasoningIndicator
  tool_start → add to EventListView
  tool_end   → complete tool event
  answer_start → transition to answering
  done       → render final ProofDisplay
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (already gitignored):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | At least one LLM key | OpenRouter API key (access many models with one key) |
| `OPENAI_API_KEY` | At least one LLM key | OpenAI API key |
| `ANTHROPIC_API_KEY` | At least one LLM key | Anthropic API key |
| `GOOGLE_API_KEY` | At least one LLM key | Google AI API key |
| `XAI_API_KEY` | At least one LLM key | xAI API key |
| `OLLAMA_CLOUD_API_KEY` | Optional | Ollama Cloud API key |
| `EXASEARCH_API_KEY` | Optional | Exa search API key (for web/paper search) |
| `LEAN_PATH` | Optional | Path to Lean 4 project with Mathlib |
| `BOURBAKI_BACKEND_URL` | Optional | Backend URL for TUI (default: `http://localhost:8000`) |

The backend loads `.env` from both `backend/.env` and the project root `.env` (via Pydantic Settings `env_file` config). API keys are auto-exported to environment variables on startup so Pydantic AI SDKs can discover them.

### Runtime Data

All runtime data lives in `.bourbaki/` (gitignored):

```
.bourbaki/
├── settings.json              # TUI settings (provider, model)
├── sessions/                  # Session JSON files
├── progress/                  # Autonomous search checkpoints
├── lean-temp/                 # Temporary Lean files
└── messages/                  # Chat history for TUI
```

---

## Running

```bash
# Start the Python backend
cd backend && uvicorn bourbaki.main:app --reload --port 8000

# Start the TUI (in another terminal)
bun start

# Or point TUI at a different backend
BOURBAKI_BACKEND_URL=http://localhost:8000 bun start
```

---

## Integrations

| Integration | Protocol | Purpose | Auth |
|-------------|----------|---------|------|
| OpenRouter | HTTPS (via Pydantic AI) | LLM inference for multiple model providers | `OPENROUTER_API_KEY` |
| OpenAI | HTTPS (via Pydantic AI) | GPT models | `OPENAI_API_KEY` |
| Anthropic | HTTPS (via Pydantic AI) | Claude models | `ANTHROPIC_API_KEY` |
| Google AI | HTTPS (via Pydantic AI) | Gemini models | `GOOGLE_API_KEY` |
| xAI | HTTPS (via Pydantic AI) | Grok models | `XAI_API_KEY` |
| Ollama | HTTP (localhost:11434) | Local model inference | None |
| OEIS | HTTPS (`oeis.org/search`) | Integer sequence lookup | None |
| arXiv | HTTPS (`export.arxiv.org/api`) | Paper search | None |
| Exa | HTTPS (`api.exa.ai/search`) | Academic/general web search | `EXASEARCH_API_KEY` |
| Lean 4 | Local subprocess | Formal theorem verification | Lean 4 + Mathlib installed |
| SymPy | Python native | Symbolic computation | None (bundled) |

---

## Solver Architecture

The solver is the proving engine that takes a Lean 4 theorem statement and produces a verified proof. It operates through a pipeline of phases, each progressively more expensive.

### Overview

```
Theorem Statement (Lean 4)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: Recursive Decomposition                            │
│   LLM sketch → have/sorry skeleton → solve subgoals        │
│   Files: decomposer.py, sketch.py, formalizer.py            │
└──────────────────────┬──────────────────────────────────────┘
                       │ if unsolved
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Best-First Search Tree (core engine, 89% of solves)│
│   Tactic candidates → REPL expansion → score → repeat      │
│   Files: search_tree.py, tactics.py, scoring.py, lean_repl  │
└──────────────────────┬──────────────────────────────────────┘
                       │ if unsolved
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2.5: Multi-Agent Coordinator                          │
│   Strategist → Searcher → Prover → Verifier (retry loop)   │
│   Files: coordinator.py, roles.py, messages.py              │
└──────────────────────┬──────────────────────────────────────┘
                       │ if unsolved
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Strategy Rotation (fallback)                       │
│   18 strategies × dead-end tracking × LLM agent             │
│   Files: search.py, strategies.py, modal_runner.py          │
└─────────────────────────────────────────────────────────────┘
```

### Phase 0: Recursive Decomposition

An LLM generates 1-3 proof sketches for the theorem. Each sketch is formalized into a Lean skeleton with `have` subgoals and `sorry` placeholders. Subgoals are solved independently via the search tree; failed subgoals are recursively decomposed up to a configurable depth.

```
backend/bourbaki/autonomous/
├── decomposer.py        # DecompositionConfig, decompose_and_prove()
├── sketch.py            # LLMSketchGenerator — proof strategy generation
└── formalizer.py         # formalize_sketch(), stitch_proofs()
```

Inspired by HILBERT (Apple, NeurIPS 2025) and DeepSeek-Prover V2.

### Phase 1: Best-First Search Tree

The core proving engine. Works like LeanDojo's ReProver:

1. **Initialize** the theorem in the Lean REPL (one-time Mathlib import ~20s, then instant per tactic)
2. **Maintain a priority queue** of proof states, each containing remaining goals
3. **Pop** the most promising state (lowest score = most promising)
4. **Generate candidate tactics** from four sources:
   - Goal-aware pattern matching (36 patterns: `∀` → `intro`, `=` → `ring/simp`, `<` → `linarith`, etc.)
   - Mathlib lemma search (Loogle, LeanSearch, LeanExplore)
   - Standard automation (norm_num, omega, simp, nlinarith, positivity, etc.)
   - Structural tactics (induction, cases, by_contra)
5. **Expand** by applying each candidate via `lean_tactic()` in the REPL
6. **Score** children: `goal_count * 10 + complexity + depth * 0.5 - novelty_bonus`
7. If goals reach 0 → **proof found**. Otherwise loop until budget exhausted.

```
backend/bourbaki/autonomous/
├── search_tree.py       # ProofNode, ProofSearchTree, best_first_search()
├── tactics.py           # generate_candidates(), generate_mathlib_queries()
├── scoring.py           # NoveltyTracker, score_proof_state()
└── novelty.py           # Goal state deduplication
```

```
backend/bourbaki/tools/
├── lean_repl.py         # LeanREPLSession, REPLSessionPool, lean_tactic()
└── mathlib_search.py    # Loogle + LeanSearch + LeanExplore (semantic)
```

**Key insight:** Using the Lean REPL instead of `lean_prover` subprocess gives ~200x speedup (0.03s/tactic vs 90s/tactic) by amortizing Mathlib import across all tactic attempts.

### Phase 2.5: Multi-Agent Coordinator

Four specialized LLM agents collaborate in a retry loop:

| Role | Responsibility | Tools |
|------|---------------|-------|
| **Strategist** | Generate proof sketch + subgoals | symbolic_compute, paper_search |
| **Searcher** | Find relevant Mathlib lemmas | mathlib_search (3 modes) |
| **Prover** | Construct Lean proof guided by strategy + lemmas | lean_tactic |
| **Verifier** | Validate complete proof | lean_prover |

On verification failure, the coordinator loops back to the Strategist with accumulated error feedback (up to `max_retries`). An `ensemble_prove()` mode launches parallel provers with different strategies, returning the first verified proof.

```
backend/bourbaki/agent/
├── coordinator.py       # ProofCoordinator, CoordinatorResult
├── roles.py             # STRATEGIST, SEARCHER, PROVER, VERIFIER
└── messages.py          # AgentMessage, MessageBus (async Queue routing)
```

The coordinator resolves custom model providers (e.g., `glm:glm-5` → `AnthropicModel` via Z.AI endpoint) using the same `_resolve_model_object()` as the main agent.

### Phase 3: Strategy Rotation (Fallback)

When all previous phases fail, the system rotates through 18 proof strategies using an LLM agent. Tracks dead ends to avoid repeating failed approaches. Supports checkpoint/resume for long-running searches.

### Lean Interaction

Two complementary tools for Lean 4 interaction:

| Tool | Speed | Use Case |
|------|-------|----------|
| `lean_tactic` (REPL) | ~30ms/tactic | Interactive proof construction, search tree expansion |
| `lean_prover` (subprocess) | ~90s/call | Final whole-file verification |

The REPL maintains a persistent subprocess (`lake env lean-repl`) with Mathlib loaded. Commands are JSON over stdin/stdout with environment chaining (`env` IDs). A `REPLSessionPool` supports up to N parallel sessions.

### Mathlib Search

Three complementary APIs for finding relevant lemmas:

| Mode | API | Strength |
|------|-----|----------|
| `type` / `name` | Loogle (`loogle.lean-lang.org`) | Exact type signature matching |
| `natural` | LeanSearch (`leansearch.net`) | Natural language queries |
| `semantic` | LeanExplore (`leanexplore.com/api/v2`) | Hybrid semantic + BM25 + PageRank |

The search tree queries all three modes with deduplication by lemma name. Semantic mode tries LeanExplore first (requires API key) and falls back to LeanSearch.

### Autoformalize

Converts natural language to Lean 4 code in two modes:

| Mode | Process |
|------|---------|
| `statement` | LLM generates Lean code → verify with `lean_prover` → retry once on failure |
| `proof_step` | LLM generates a single tactic for the current proof state |

```
backend/bourbaki/tools/autoformalize.py
```

---

## Benchmark

### miniF2F

The standard benchmark for automated theorem proving. 488 problems (244 validation, 244 test) from AMC, AIME, IMO, and textbooks, formalized in Lean 4.

```
backend/bourbaki/benchmarks/
├── loader.py            # Parse Lean 4 files, extract theorem statements
├── minif2f.py           # Baseline benchmark runner (automation + search tree)
└── run_enhanced.py      # Enhanced runner (baseline + multi-agent fallback)
```

**Benchmark runner flow:**

1. Load problems from `.bourbaki/miniF2F-lean4/` Lean files
2. Initialize REPL with full Mathlib import (~20s one-time)
3. For each problem:
   a. Try automation tactics via REPL (norm_num, omega, ring, simp, etc.)
   b. If unsolved: run best-first search tree with budget
4. (Enhanced mode) For unsolved problems: run multi-agent coordinator with LLM

**Current results (miniF2F valid split, 244 problems):**

| Phase | Solved | Rate |
|-------|--------|------|
| Automation alone | 56/244 | 23% |
| + Search tree v1 | 144/244 | 59% |
| + Search tree v2 | 217/244 | 89% |
| + Semantic search | 218/244 | 89.3% |
| + Multi-agent (GLM-5) | 224/244 | 91.8% |

**Per-category breakdown (latest):**

| Category | Solved | Rate |
|----------|--------|------|
| imo | 20/20 | 100% |
| induction | 8/8 | 100% |
| mathd | 130/130 | 100% |
| numbertheory | 8/8 | 100% |
| aime | 8/12 | 67% |
| algebra | 11/18 | 61% |
| unknown | 33/48 | 69% |

### Comparison with Reference Systems

| System | Organization | miniF2F Valid | Key Technique |
|--------|-------------|---------------|---------------|
| HILBERT | Apple | 99.2% | Recursive decomposition + Goedel-V2-32B + MPNet retrieval |
| Goedel-Prover V2 | — | 90.4% | 2-round self-correction (8B matches 671B) |
| Aristotle | Harmonic | 90% | MCGS + 200B transformer + test-time training |
| **Bourbaki** | — | **91.8%** | Best-first search + multi-agent (GLM-5) |
| DeepSeek-Prover V2 | DeepSeek | 88.9% | Recursive decomposition + GRPO RL |

### Capability Matrix

| Capability | HILBERT | Aristotle | DeepSeek-V2 | Numina | Bourbaki |
|------------|---------|-----------|-------------|--------|----------|
| Best-first search | Yes | MCGS | Sampling | Yes | Yes |
| Recursive decomposition | Core | No | Core | Blueprint | Basic |
| Semantic retrieval | MPNet+FAISS | — | — | LeanDex | LeanExplore API |
| Self-correction | — | — | — | — | Yes (basic) |
| Multi-agent | — | — | — | Claude+MCP | Yes |
| Lean LSP | — | — | — | 17 tools | No (REPL only) |
| Trained value model | Goedel-V2-32B | 200B | GRPO RL | — | Heuristic |

---

## Adding Models

To add new models to a provider, edit the `PROVIDERS` array in `src/components/ModelSelector.tsx`. Models are strings matching the format expected by the provider's API (e.g., OpenRouter uses `org/model-name` format).

To change the default model, edit `src/constants.ts`:

```typescript
export const DEFAULT_PROVIDER = 'openrouter';
export const DEFAULT_MODEL = 'openrouter:openrouter/pony-alpha';
```

The backend's default model (used when no model is specified) is set in `backend/bourbaki/config.py`:

```python
default_model: str = "openai:gpt-4o"
```

The TUI default takes precedence since it always sends a model with each request.

# Bourbaki - Project Context for Claude

This file helps Claude understand the project when starting a new session.

## What is Bourbaki?

Bourbaki is an AI agent for mathematical reasoning and theorem proving. Named after Nicolas Bourbaki, the collective pseudonym of mathematicians who wrote foundational treatises.

## Architecture

```
TUI (React + Ink, Bun)  ──HTTP+SSE──→  Python Backend (FastAPI + Pydantic AI)
                                              │
                                         Pydantic AI Agent
                                              │
                                         ┌────┼────┬──────────┬──────────┬──────┐
                                         │    │    │          │          │      │
                                      SymPy  Lean OEIS     arXiv      Exa   Skills
                                     native  sub  httpx    httpx     httpx  SKILL.md
```

The backend is Python (FastAPI + Pydantic AI). The TUI is TypeScript (React + Ink) and communicates with the backend via HTTP + Server-Sent Events.

For long-running theorem proving the backend runs a proposer-builder-reviewer
loop in `backend/bourbaki/prover/` (commit `2113629`, May 2026). The legacy
HILBERT pipeline (`sketch` / `formalizer` / `decomposer` / `search_tree` /
`scoring` / `strategies` / `search` / `modal_runner` / `progress`) was deleted
in Phase 3; `backend/bourbaki/autonomous/` retains only `tactics.py` for its
blocklist.

## Core Tools (Python Backend)

| Tool | File | Purpose |
|------|------|---------|
| `symbolic_compute` | `backend/bourbaki/tools/symbolic_compute.py` | Native SymPy (30 ops + aliases) |
| `lean_prover` | `backend/bourbaki/tools/lean_prover.py` | Lean 4 via asyncio subprocess |
| `lean_tactic` | `backend/bourbaki/tools/lean_repl.py` | Lean 4 REPL for tactic-by-tactic interaction |
| `mathlib_search` | `backend/bourbaki/tools/mathlib_search.py` | Loogle + LeanSearch for Mathlib lemma lookup |
| `sequence_lookup` | `backend/bourbaki/tools/sequence_lookup.py` | OEIS via httpx |
| `paper_search` | `backend/bourbaki/tools/paper_search.py` | arXiv via httpx |
| `web_search` | `backend/bourbaki/tools/web_search.py` | Exa API for academic/general web search |
| `skill_invoke` | `backend/bourbaki/tools/skill_tool.py` | SKILL.md instruction loader |

## Skills (21 Proof Techniques)

SKILL.md files live in `src/skills/` (builtin), `~/.bourbaki/skills/` (user), `.bourbaki/skills/` (project). The Python backend discovers and loads them.

**Basic:** proof-by-induction, strong-induction, direct-proof, proof-by-contradiction, pigeonhole-argument, counting-argument

**Analysis:** epsilon-delta-proof, convergence-test, sequence-limit, inequality-chain

**Geometry:** coordinate-proof, synthetic-construction, transformation-proof

**Algebra:** group-homomorphism, ring-ideal-proof, polynomial-proof

**Advanced:** extremal-argument, probabilistic-method, conjecture-exploration, formalize-informal-proof, explain-proof

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `backend/bourbaki/` | Python backend (FastAPI + Pydantic AI) |
| `backend/bourbaki/agent/` | Agent core, prompts, scratchpad, event mapper |
| `backend/bourbaki/tools/` | Native Python tools (SymPy, Lean, OEIS, arXiv) |
| `backend/bourbaki/skills/` | Skill loader and registry |
| `backend/bourbaki/sessions/` | Session persistence and context compaction |
| `backend/bourbaki/prover/` | Proposer-builder-reviewer-memory loop (current proving engine) |
| `backend/bourbaki/autonomous/` | Holds only `tactics.py` (blocklist still used by builder); other modules deleted in Phase 3 (commit `2113629`) |
| `backend/bourbaki/benchmarks/` | miniF2F + PutnamBench runners (`loader.py`, `minif2f.py`, `putnam.py`) |
| `backend/bourbaki/problems/` | Problem database (13 classic problems) |
| `backend/bourbaki/server/routes/` | FastAPI route handlers |
| `src/` | TypeScript TUI (React + Ink) |
| `src/components/` | TUI React components |
| `src/hooks/` | React hooks (useAgentRunner, useModelSelection) |
| `src/agent/` | Event types, session manager (TUI-side) |
| `src/skills/` | SKILL.md files (21 proof techniques) |
| `.bourbaki/` | Runtime data (sessions, checkpoints, proofs) |

## Commands

```bash
# Start Python backend
cd backend && uvicorn bourbaki.main:app --reload --port 8000

# Start TUI (connects to backend at localhost:8000)
bun start

# Or set custom backend URL
BOURBAKI_BACKEND_URL=http://localhost:8000 bun start
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Server health check |
| POST | /query | Agent SSE stream |
| POST | /compute | Symbolic computation |
| POST | /prove | Lean 4 verification |
| POST | /search/sequence | OEIS lookup |
| POST | /export | Export proof as LaTeX/Lean/Markdown |
| POST | /search/paper | arXiv search |
| GET | /sessions | List sessions |
| POST | /sessions | Create session |
| GET | /sessions/{id} | Load session |
| DELETE | /sessions/{id} | Delete session |
| GET | /sessions/{id}/messages | Get messages |
| GET | /problems | List problems |
| GET | /problems/random | Random problem |
| GET | /problems/{id} | Get problem |
| POST | /autonomous/start | **Deprecated, returns 410 Gone** (use `/query` with `use_loop=True`; legacy pipeline deleted in commit `2113629`) |
| POST | /autonomous/pause | **Deprecated, returns 410 Gone** |
| POST | /autonomous/resume | **Deprecated, returns 410 Gone** |
| GET | /autonomous/progress | **Deprecated, returns 410 Gone** |
| GET | /autonomous/insights | **Deprecated, returns 410 Gone** |
| GET | /skills | List proof technique skills |

## TUI Commands

| Command | Action |
|---------|--------|
| `/help` | Show all commands |
| `/sessions` | List saved sessions |
| `/sessions <id>` | Restore a session |
| `/restore <id>` | Restore a session |
| `/new` | Start fresh session |
| `/model` | Change model/provider |
| `/debug` | Toggle debug panel |
| `/problems` | List available problems |
| `/prove <id>` | Start proof attempt (legacy TUI handler still POSTs to `/autonomous/start`, which now returns 410; the loop is reachable via `/query` with `use_loop=True` or the `attempt_proof_loop` driver) |
| `/pause` | Pause proof search (legacy, 410) |
| `/progress` | Show proof search progress (legacy, 410) |
| `/skills` | List available proof techniques |
| `/export [format]` | Export last proof (latex/lean/markdown) |

## Key Files

| File | Purpose |
|------|---------|
| `backend/bourbaki/agent/core.py` | Agent runner using Pydantic AI agent.iter() with scratchpad enforcement |
| `backend/bourbaki/agent/scratchpad.py` | Tool call tracking, limits, and deduplication |
| `backend/bourbaki/events.py` | AgentEvent models (SSE wire format) |
| `backend/bourbaki/config.py` | Pydantic Settings (.env config) |
| `backend/bourbaki/server/routes/query.py` | SSE streaming endpoint |
| `backend/bourbaki/prover/prover.py` | `ProverLoop` driver (proposer / builder / reviewer / memory routing) |
| `backend/bourbaki/prover/state.py` | `ProverState`, `ProposalMessage`, `FeedbackMessage`, `ProverResult`, `ReviewDecision` |
| `backend/bourbaki/tools/proof_code_builder.py` | `assemble_standalone_proof(preamble, code)` shared by builder and reviewer |
| `src/agent/types.ts` | Event type definitions (the contract) |
| `src/hooks/useAgentRunner.ts` | TUI ↔ backend bridge (fetch+SSE) |

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic AI, SymPy, httpx
- **TUI:** Bun, React + Ink, TypeScript
- **Verification:** Lean 4 + Mathlib
- **Sequences:** OEIS API
- **Papers:** arXiv API

## Development Roadmap

Plans for the proposer-builder-reviewer refactor live in `.bourbaki/plans/`
(gitignored, checked in only as context for new sessions):

- `.bourbaki/plans/REFACTOR_PLAN.md` — consolidated phasing
- `.bourbaki/plans/proposer-builder-loop.md` — full design, prompts, signatures
- `.bourbaki/plans/refactor-audit.md` — keep / reuse / drop classification
- `.bourbaki/plans/decomposer-blockers.md` — historical failure modes (pre-refactor)
- `.bourbaki/plans/lean-interact-eval.md` — alternative REPL library evaluation

Phase status (commits in chronological order): Phase 1 scaffold (`49211ce`),
Phase 2 loop body (`9d8adbb`, `6beda20`, `fdcb76b`, `c06e006`), Phase 2 fixes
(`aa99595`, `8bc16ec`, `ef364ee`), Phase 3 legacy delete (`2113629`),
Phase 4 mathlib_search proposer tool (`4ef9398`) and Pass@N (`3222a07`).
Open follow-up issues: #11 (status tracker), #14 (full 244-problem run),
#15 (v0.3.0 release + tag), #17 (mathlib_search A/B), #18 (Pass@N A/B).

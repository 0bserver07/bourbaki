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
| `backend/bourbaki/autonomous/` | Long-running proof search with strategies |
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
| POST | /autonomous/start | Start proof search |
| POST | /autonomous/pause | Pause search |
| POST | /autonomous/resume | Resume search |
| GET | /autonomous/progress | Get progress |
| GET | /autonomous/insights | Get insights |
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
| `/prove <id>` | Start autonomous proof search |
| `/pause` | Pause proof search |
| `/progress` | Show proof search progress |
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
| `src/agent/types.ts` | Event type definitions (the contract) |
| `src/hooks/useAgentRunner.ts` | TUI ↔ backend bridge (fetch+SSE) |

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic AI, SymPy, httpx
- **TUI:** Bun, React + Ink, TypeScript
- **Verification:** Lean 4 + Mathlib
- **Sequences:** OEIS API
- **Papers:** arXiv API

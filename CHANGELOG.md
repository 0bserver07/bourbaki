# Changelog

## v0.2.2 — 2026-03-08

### REPL session corruption fix + honest benchmark numbers

The verified miniF2F baseline is **63/244 (25.8%)** — not the 91.8% previously
reported from REPL-only detection. Every solve now compiles as a standalone
Lean file via `lean_prover`.

#### What happened

The REPL-reported numbers (91.8%–94.3%) included false positives: the REPL
reported `goals=[]` for tactics that don't compile standalone. The actual
verified number dropped to 63/244 after adding `lean_prover` verification.

Additionally, a critical REPL session corruption bug was causing cascading
failures during benchmark runs. When a tactic timed out, unconsumed response
data remained in the stdout pipe, corrupting all subsequent commands. This
meant problems later in the sequence failed not because they were hard, but
because the REPL was broken.

#### Fixes

- **REPL pipe recovery** — After a tactic timeout, drain remaining output to
  resync the pipe. If drain fails (tactic hung), kill session so it auto-restarts.
  Handles both internal timeouts and external cancellation (`asyncio.wait_for`).

- **Removed false-positive tactics** — `apply Set.mem_of_mem_filter` (158 false
  positives), bogus tactic candidates, `decide` on membership goals (caused
  timeout-induced corruption).

- **Init error handling** — `lean_tactic()` now correctly reports failure when
  `send_cmd` returns a timeout/parse error during initialization (was silently
  returning `proofComplete=True`).

#### Verified Results

| Category | Verified | Rate |
|----------|----------|------|
| mathd | 54/130 | 42% |
| algebra | 5/18 | 28% |
| AMC/misc | 4/48 | 8% |
| aime | 0/12 | 0% |
| imo | 0/20 | 0% |
| induction | 0/8 | 0% |
| numbertheory | 0/8 | 0% |

Top verified tactics: norm_num (23), omega (13), ring (8), linarith (7),
decide (7), nlinarith (3), simp_all (2).

#### Lessons Learned

1. **Never trust REPL-reported proof completion without standalone verification.**
   The REPL reports `goals=[]` for tactics that produce terms the type checker
   accepts locally but that don't compile in a standalone file (missing imports,
   universe issues, etc.).

2. **Subprocess pipe protocols are fragile.** When `asyncio.wait_for` cancels a
   read mid-stream, the remaining bytes stay in the pipe. The next read gets a
   mix of old and new data. This is a fundamental issue with any line-delimited
   protocol over stdin/stdout — you must drain or restart after any interruption.

3. **Benchmark corruption cascades silently.** The first timeout corrupted the
   session, but subsequent failures looked like normal tactic failures. Without
   logging the JSON parse errors, the corruption was invisible.

4. **The gap between 25.8% and SOTA (99.2%) is multi-step proofs.** The 63 verified
   solves are all single-tactic proofs. The search tree finds multi-step proofs
   via REPL but can't reliably translate them to standalone code.

---

## v0.2.1 — 2026-02-18

### 94.3% on miniF2F test split — no fine-tuning, no custom training

Bourbaki now solves 230 out of 244 problems on the miniF2F test split using
a coding-agent architecture with an off-the-shelf LLM (GLM-5) and no custom
training. The baseline search tree alone (92.6%, no LLM) beats Aristotle (90%)
and Goedel-Prover V2 (90.4%).

#### Benchmark Results

| Split | Baseline | + Multi-agent | Total |
|-------|----------|--------------|-------|
| **Valid** | 218/244 (89.3%) | +6 | **224/244 (91.8%)** |
| **Test** | 226/244 (92.6%) | +4 | **230/244 (94.3%)** |
| **Combined** | 444/488 (91.0%) | +10 | **454/488 (93.0%)** |

#### New in v0.2.1

- **Deeper recursive decomposition** — Max depth 3 (was 2), parallel independent
  subgoal solving, budget/timeout decay per depth level, Lean verification of
  stitched proofs, sketch fallthrough on verification failure.

- **Local FAISS embedding index** — Offline semantic search over ~225K Mathlib
  declarations using sentence-transformers (all-MiniLM-L6-v2). <50ms query time,
  no API dependency. Semantic fallback chain: local FAISS → LeanExplore → LeanSearch.

- **Lean LSP integration** — Direct JSON-RPC communication with `lean --server`
  for completions, diagnostics, hover/type info, and proof goal state. Three
  complementary Lean interaction layers: REPL (30ms/tactic), Prover (whole-file),
  LSP (intelligent assistance).

- **Error-conditioned repair** — Goedel-V2-style tactic correction: when a tactic
  fails, the error message guides targeted repair candidates (unknown identifier →
  cast/open, type mismatch → norm_cast, unsolved goals → chain with simp_all, etc.).

- **UCB exploration fix** — ProofNode.visits now correctly propagates up ancestors,
  making the UCB exploration bonus functional.

- **Tool call limits raised** — lean_tactic 3→25, lean_prover 3→10,
  mathlib_search 3→10. Hard problems can now iterate properly.

- **REPL buffer fix** — 4MB stream buffer (was 64KB) prevents crashes on large
  Lean responses.

#### Test Coverage

247 tests across 21 test files (+108 from v0.2.0).

---

## v0.2.0 — 2026-02-17

### 91.8% on miniF2F (valid split) — competitive with Aristotle and DeepSeek-Prover

Bourbaki now solves 224 out of 244 problems on the miniF2F benchmark (valid split),
the standard benchmark used by HILBERT (Apple), Aristotle (Harmonic), DeepSeek-Prover,
and other state-of-the-art theorem provers.

#### Benchmark Results

| Category | Solved | Rate |
|----------|--------|------|
| IMO | 20/20 | 100% |
| Induction | 8/8 | 100% |
| MathD | 130/130 | 100% |
| Number Theory | 8/8 | 100% |
| AIME | 8/12 | 67% |
| Algebra | 11/18 | 61% |
| **Total** | **224/244** | **91.8%** |

#### How We Compare

| System | miniF2F Valid | Organization |
|--------|-------------|--------------|
| HILBERT | 99.2% | Apple |
| **Bourbaki** | **91.8%** | — |
| Goedel-Prover V2 | 90.4% | — |
| Aristotle | 90% | Harmonic |
| DeepSeek-Prover V2 | 88.9% | DeepSeek |

#### New Features

- **Best-first proof search tree** — Tactic-by-tactic exploration using Lean REPL
  with goal-aware candidate generation, novelty tracking, and UCB-adjusted scoring.
  Took baseline from 23% (automation alone) to 89%.

- **Multi-agent proof coordinator** — Four specialized LLM agents (Strategist,
  Searcher, Prover, Verifier) collaborate in a retry loop with error feedback.
  Solves 6 additional problems that pure automation can't crack.

- **Semantic Mathlib retrieval** — LeanExplore API integration for hybrid
  semantic + BM25 + PageRank lemma search, alongside existing Loogle and
  LeanSearch backends. Goal-state-aware queries with cross-mode deduplication.

- **Recursive subgoal decomposition** — LLM-generated proof sketches formalized
  into `have`/`sorry` skeletons, with subgoals solved independently and
  recursively decomposed on failure (HILBERT-inspired).

- **Autoformalize tool** — Natural language to Lean 4 conversion with statement
  verification and proof step generation modes.

- **Novelty tracking** — Goal state deduplication and intrinsic motivation bonus
  for exploring novel proof configurations (DeepSeek-V1.5-inspired).

- **REPL session pool** — Parallel tactic screening across multiple persistent
  Lean REPL sessions.

- **Self-correction improvements** — 2-round correction cap, sketch-first workflow,
  structured error classification with recovery hints.

- **Enhanced benchmark runner** — Two-phase benchmark: baseline automation + search
  tree, then multi-agent fallback on unsolved problems. CLI interface with
  configurable model, timeout, and problem filtering.

#### Architecture

The solver pipeline runs four phases in order of increasing cost:

```
Phase 0: Recursive Decomposition (sketch → subgoals → solve independently)
Phase 1: Best-First Search Tree (tactic candidates → REPL → score → expand)
Phase 2.5: Multi-Agent Coordinator (Strategist → Searcher → Prover → Verifier)
Phase 3: Strategy Rotation (18 strategies × dead-end tracking × LLM agent)
```

See `ARCHITECTURE.md` for full documentation.

#### Test Coverage

139 tests across 14 test files covering search tree, decomposition, coordinator,
roles, messages, tactics, mathlib search, autoformalize, and integration scenarios.

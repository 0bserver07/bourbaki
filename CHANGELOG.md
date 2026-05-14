# Changelog

## v0.3.0-pending — 2026-04-25 → 2026-05-14

### Proposer-builder-reviewer loop replaces HILBERT pipeline

A complete refactor of the autonomous prover. The `sketch → formalize →
decompose → stitch` HILBERT-style pipeline (where stitching brittleness
was eating ~50% of the budget) is replaced with an ax-prover-style
loop: GLM-5.1 proposes a complete proof, a warm `LeanREPLSession` runs
it, a reviewer node gates on `check_1 (statement preserved) AND check_2
(no sorry/admit)`, and `lean_prover` runs once at approval as the
final ground-truth gate.

Phase 3 (commit `2113629`) deleted the legacy `autonomous/` pipeline —
`sketch.py`, `formalizer.py`, `decomposer.py`, `search_tree.py`,
`scoring.py`, `strategies.py`, `search.py`, `modal_runner.py`,
`progress.py` — net `-6,576 / +118` lines. The `/autonomous/*` route
handlers now return HTTP 410 Gone with a deprecation message pointing
clients at `/query` with `use_loop=True`. The blocklist in
`autonomous/tactics.py` is the only legacy module that survived.

**10-problem A/B (same subset as the 50% decomposer baseline):**

| Approach | Verified | False positives | Wall time |
|----------|----------|-----------------|-----------|
| Decomposer (Apr 1) | 5/10 (50%) | 0 | ~50 min |
| Loop, run 1 (Apr 25) | 5/10 (50%) | 2 | 21 min |
| Loop, run 2 — after 5 follow-up fixes | **9/10 (90%)** | **0** | 22 min |

The single remaining failure (`mathd_algebra_31`, an NNReal
`Filter.Tendsto` fixed-point problem) is genuinely hard — the
decomposer also failed it.

**35-problem stratified follow-up (2026-05-09):** **22/35 (62.9%) verified, 0 false positives.**
More than doubles the 28.6% heuristic baseline on the same sample.
Per-source breakdown: aime 0/3, imo 0/3 (LLM unfamiliar), mathd 13/15
(87%), induction 2/2 and numbertheory 2/2 (100%), algebra 3/5, amc 2/3,
unknown 2/5. Result file:
`.bourbaki/benchmarks/results/2026-05-09_2241_minif2f_valid.json`.

**Phase 4 (already wired, off by default):** `mathlib_search` as a
proposer tool (commit `4ef9398`, A/B tracked in
[#17](https://github.com/0bserver07/bourbaki/issues/17)) and Pass@N
sampling via `attempt_proof_pass_at_n` (commit `3222a07`, A/B tracked
in [#18](https://github.com/0bserver07/bourbaki/issues/18)).

**Status:** v0.3.0 tag withheld pending the full 244-problem run with
the new loop, tracked in
[#14](https://github.com/0bserver07/bourbaki/issues/14). The retraction
of v0.2.0 / v0.2.1 GitHub releases is tracked in
[#15](https://github.com/0bserver07/bourbaki/issues/15) (release titles
were updated to "RETRACTED" on 2026-05-14).

**Architecture:** `backend/bourbaki/prover/` — see
[`.bourbaki/plans/proposer-builder-loop.md`](.bourbaki/plans/proposer-builder-loop.md)
for the design and
[`.bourbaki/plans/refactor-audit.md`](.bourbaki/plans/refactor-audit.md)
for keep/reuse/drop classification.

**Key fixes that turned 50% into 90%:**

- Reviewer used `preamble + proposal.code` for its `lean_prover` gate;
  `state.final_proof_code` was bare `proposal.code`. Outer benchmark
  verifier ran without `set_option maxHeartbeats 0` from the preamble
  → default heartbeat caps flipped honest passes into phantom false
  positives. Shared `assemble_standalone_proof()` helper now produces
  byte-identical source for both sites.
- Added 90s `asyncio.wait_for` cap on every LLM call (proposer, reviewer,
  ExperienceMemory) so a single hang can't burn the per-problem budget.
- `missing_target_theorem` is no longer terminal — a single-character
  typo in a 50-character theorem name no longer kills the loop.
- `_route_builder` and `_route_reviewer` now respect `is_terminal`;
  previously only `_route_proposer` did, so a builder-issued terminal
  feedback was misrouted to "retry".
- Builder now strips imports from `state.preamble` too, not just from
  the proposal code (REPL has Mathlib pre-loaded; re-importing errors).

**Late-cycle fix that may bump the 35-problem number (commit `7b07c07`):**

- Reviewer's final `lean_prover` gate was using the function's 30s default
  timeout. Standalone `lake env lean` with `import Mathlib` needs 60-180s
  on a cold cache (the REPL stays warm; standalone compiles don't). So
  on a busy or cold system every reviewer call silently timed out and
  the loop reported FAILED even when the proposer had generated a
  correct proof. Bumped to **240s** to match the outer benchmark verifier.
- Implication: **the 22/35 (62.9%) May 9 number is a lower bound.**
  Some failures may have been correct proofs that just timed out at the
  reviewer's gate. Re-running with the fix is tracked separately
  (see [#19](https://github.com/0bserver07/bourbaki/issues/19)).

**z.ai routing reverted to Anthropic-compat (commit `66cba4c`):**

- The earlier "switch to OpenAI-compat to dodge pydantic_ai's
  `args_as_dict` crash" was wrong-direction — z.ai's billing splits the
  two endpoints into separate resource pools and the user's funds are on
  the Anthropic-compat side. Restored `glm:` → Anthropic-compat
  (`https://api.z.ai/api/anthropic`) and patched the upstream
  `args_as_dict` bug at the source with a defensive shim in
  `bourbaki.prover._pydantic_ai_compat`. New `glm-oai:` prefix retained
  as an opt-in for the OpenAI-compat pool. Verified live: 6.1s for a
  structured `ProverResult` response.

**Wave-1 parallel agent integration (commits `4eb3430`..`dcdfc57`):**

- Deleted `agent/coordinator.py`, `agent/roles.py`, `agent/messages.py`,
  `benchmarks/run_enhanced.py` (~1,296 LoC) — vestigial multi-agent
  infra not referenced by anything live.
- Added `backend/scripts/` standalone runners + `justfile` + `backend/
  bourbaki/benchmarks/results_db.py` for benchmark result inspection.
- Pyright across `backend/bourbaki/`: **48 errors → 0**.
- Drafted `backend/bourbaki/prover/prompts_v2.py` (-150 tokens/iter,
  tactic shortlist, ≤2-sentence reasoning cap). Not swapped in yet —
  needs an A/B run first ([#17](https://github.com/0bserver07/bourbaki/issues/17)).
- TUI's `/prove`, `/pause`, `/progress` deprecated (legacy routes now
  return HTTP 410); commands print deprecation messages instead.

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

## v0.2.1 — 2026-02-18 — **RETRACTED**

> **RETRACTED 2026-05-14.** The 91.8% / 94.3% claims below were inflated
> ~15× by REPL false positives — the REPL reported `goals=[]` for tactics
> whose standalone Lean compile rejected them. The corresponding GitHub
> release title now reads "v0.2.1 — RETRACTED (inflated numbers)". The
> honest verified rate on the same code is 6.2% on the valid split
> (re-verified 2026-02-22; see the v0.2.2 entry and
> [`docs/REALITY_CHECK.md`](docs/REALITY_CHECK.md) for the audit).
> Retained here only for historical context; do not cite these numbers.

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

## v0.2.0 — 2026-02-17 — **RETRACTED**

> **RETRACTED 2026-05-14.** The 91.8% / 224-of-244 number below was
> inflated ~15× by REPL false positives — the same REPL bug that
> invalidated v0.2.1. The corresponding GitHub release title now reads
> "v0.2.0 — RETRACTED (inflated numbers)". The honest verified rate on
> the same code is 6.2% on the valid split (re-verified 2026-02-22;
> see the v0.2.2 entry below and
> [`docs/REALITY_CHECK.md`](docs/REALITY_CHECK.md)). Retained here only
> for historical context; do not cite these numbers.

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

# Bourbaki Assessment Report

> **SUPERSEDED 2026-05-13.** This report was written on 2026-03-19 and
> describes the decomposer-era architecture (sketch → formalize →
> decompose → search → stitch). That pipeline was deleted in Phase 3
> (commit `2113629`, May 2026, net `-6,576 / +118` lines) and replaced by
> the proposer-builder-reviewer loop in `backend/bourbaki/prover/`.
>
> The numbers in this report (28.6% heuristic, 40% with decomposer) are
> historical. Current verified rates (lean_prover-gated, 0 false positives):
> **9/10 (90%)** on a 10-problem subset (2026-04-25) and **22/35 (62.9%)**
> on a 35-problem stratified sample (2026-05-09). For the current state
> read `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, and
> `docs/REALITY_CHECK.md`.
>
> Most module references below (`sketch.py`, `formalizer.py`, `decomposer.py`,
> `search_tree.py`, `scoring.py`, `strategies.py`, `modal_runner.py`,
> `progress.py`) no longer exist on disk; they survive here only as the
> pre-refactor history of the project. The blocklist in
> `autonomous/tactics.py` is the only legacy module that still ships.
>
> *Last edited 2026-03-19 (original).*
> *Original status: Active development, post-correction.*
> *Original author: Project maintainers.*

---

## Summary

Bourbaki is an AI agent for mathematical theorem proving built in 8 days
(Feb 9-18, 2026). It reported 91.8% on miniF2F — a number that turned out
to be 15x inflated due to REPL false positives. The actual verified rate is
~28%. A correction sprint (Mar 14-19) fixed the verification pipeline, wired
the LLM decomposer into the benchmark, and produced the first honest numbers
with the full system running.

| Metric | Claimed (Feb 18) | Verified (Mar 19) | Current (May 9) |
|--------|:---:|:---:|:---:|
| miniF2F valid | 91.8% — **RETRACTED** | 28.6% (heuristics) / 40% (decomposer, 4/10) | **22/35 (62.9%)** stratified sample |
| PutnamBench | 95.4% — **RETRACTED** | 0% | Re-run pending after #13 closed in `66cba4c` |
| SOTA comparison | "Competitive with Aristotle" — **RETRACTED** | 70pp behind HILBERT | Full 244 run is v0.3.0 blocker, [#14](https://github.com/0bserver07/bourbaki/issues/14) |
| False positives in results | Unknown (~15× inflation) | 0 (inline verification) | 0 |

---

## What happened

### The build (Feb 9-18)

35 commits in 8 days. From `let there be proof` to a complete system:
search tree, REPL, FAISS retrieval, multi-agent coordinator, decomposer,
sketch generator, LSP integration, benchmark runners, TUI.

### The inflation (Feb 18)

91.8% miniF2F claimed. The REPL said "proof complete" and we believed it.
No standalone verification existed. The number looked plausible (below
HILBERT at 99.2%, above Aristotle at 90%).

### The crash (Feb 19-23)

PutnamBench: 0% verified. miniF2F: dropped to 6.2%, then 25.8% after
removing bogus tactics. Root cause: the REPL's sorry-initialized proof
states have different elaboration behavior than standalone compilation.
`exact <_, _>` alone produced 100+ false positives.

### The correction (Mar 14-19)

This session. Four fixes merged, benchmark re-run, decomposer wired in.

---

## Correction sprint results (Mar 14-19)

### Fixes applied

| Issue | Fix | Impact |
|-------|-----|--------|
| #2 proof_code | `_build_proof_code()` — full standalone Lean files | Multi-step proofs now verify |
| #3 blocklist | Centralized tactic blocklist at 4 pipeline entry points | `exact <_, _>` and friends blocked |
| #6 inline verify | `lean_prover` gate on every `goals=[]` during search | 0 false positives reach results |
| #9 guardrails | Benchmark runners default `verify=True`, honest reporting | Can't accidentally report unverified numbers |
| expand() fix | Don't short-circuit on complete proofs | Recovered 1 solve, search explores properly |
| decomposer wiring | `decompose_and_prove()` as Phase 2 in benchmark | First LLM-assisted solves |
| parallel fix | `parallel_subgoals=False` | REPL concurrency crash fixed |

### Verified benchmark results

**35-problem stratified sample (heuristics only):**

| Category | Verified | Rate |
|----------|---------|------|
| mathd | 8/15 | 53% |
| algebra | 1/5 | 20% |
| unknown (AMC) | 1/5 | 20% |
| aime | 0/3 | 0% |
| imo | 0/3 | 0% |
| induction | 0/2 | 0% |
| numbertheory | 0/2 | 0% |
| **Total** | **10/35** | **28.6%** |

**10-problem subset with decomposer enabled (GLM-5):**

| Result | Count |
|--------|-------|
| Verified | 4/10 (40%) |
| REPL-reported | 4/10 |
| False positives | 0 |

The decomposer added 1 solve over the heuristic baseline on the same problem
set (4/10 vs 3/10 without). Small signal but it's the first verified evidence
that the LLM decomposition adds value.

### What the solves look like

Single-tactic (majority): `norm_num`, `omega`, `ring`, `linarith`

Multi-step (new, from search tree):
- `mathd_algebra_192`: `aesop -> field_simp -> ring -> norm_num` (4 tactics)
- `mathd_algebra_234`: `simp [pow_succ] -> nlinarith` (2 tactics)
- `mathd_algebra_547`: `aesop -> ring` (2 tactics, recovered by expand() fix)

---

## Architecture vs SOTA

### What we tested and learned (Mar 19)

**LLM tactic suggestions (wrong approach):** Added `generate_llm_tactics()`
that asks GLM-5 to suggest tactics for each proof state. Result: 0 new tactics
added over the heuristic generator on every single call across 35 problems.
The LLM suggests the same `norm_num`, `omega`, `ring` that the rule-based
system already has. Individual tactic suggestion is not where LLMs add value.

**LLM decomposition (right approach):** Wired `decompose_and_prove()` into the
benchmark. GLM-5 generates real mathematical decompositions: "completing_square"
with subgoal `0 <= (a - 2 - c) ^ 2`, "calc" chains, "linear_combination"
strategies. This is how HILBERT and Seed-Prover use LLMs — for proof planning,
not tactic suggestion.

### Honest comparison to SOTA

| Component | HILBERT (99.2%) | Seed-Prover (100%) | Bourbaki (28-40%) |
|-----------|:---:|:---:|:---:|
| Tactic generation | Fine-tuned 7B/32B | RL-trained LLM | Hardcoded heuristics |
| Proof decomposition | Recursive (Gemini 2.5 Pro) | Lemma-based (RL-trained) | GLM-5 sketches (wired in Mar 19) |
| Verification | Lean server (real-time) | Lean compiler | `lean_prover` inline (fixed Mar 14) |
| Search strategy | 4 candidates + error correction | Iterative refinement + conjectures | UCB best-first |
| Pass@N | Multiple candidates per step | 5000+ conjectures | 1 attempt |
| Self-correction | Lean error feedback (6 passes) | RL-trained | Rule-based corrections |
| Training | Goedel-V2-32B fine-tuned | Multi-stage RL on VAPO | None |

**miniF2F is saturated by SOTA.** Seed-Prover hit 100% valid. HILBERT fails
on only 2 problems. Bourbaki at 28-40% is not competitive on this benchmark.

**The real frontier is PutnamBench and IMO problems.** AxiomProver solved all
12 Putnam 2025 problems (raised $200M). Seed-Prover 1.5 solved 11/12.
HILBERT solved 462/660 PutnamBench. Bourbaki solved 0.

### What the gap actually is

The gap is not "we need a model" — we have GLM-5. The gap is:

1. **The decomposer works but subgoal solving is too weak.** GLM-5 generates
   good decompositions. The search tree can't solve the resulting subgoals
   because they require domain-specific tactics the heuristic generator
   doesn't produce.

2. **REPL session management blocks parallel solving.** The decomposer wants
   to solve subgoals in parallel but the REPL is single-process. Fixed with
   `parallel_subgoals=False` but this makes decomposition slow.

3. **`induction...simp_all` wastes search budget.** This false positive
   pattern poisons every induction-shaped problem. Each verification costs
   ~6s. Needs blocklisting (#10).

4. **No Pass@N.** Running 32 attempts per problem would likely boost numbers
   significantly with no code changes.

5. **No trained tactic policy.** The heuristic generator produces the same
   candidates regardless of proof state. A trained model would produce
   problem-specific tactics.

---

## Lessons learned (updated)

### 1. Verify before you celebrate

The original lesson. Still applies. Now structurally enforced by inline
verification (#6) and benchmark guardrails (#9).

### 2. Don't claim, check

This session repeated the pattern twice:
- "It's a model problem" — wrong, we have GLM-5 and weren't using it
- "LLM tactic suggestions will help" — wrong, added 0 new tactics

The fix: read the code before making claims. Check the data before
proposing solutions.

### 3. The pieces were there, just not connected

The decomposer, sketch generator, and multi-agent coordinator were all
built in the original sprint but never wired into the benchmark runner.
The benchmark was running a fraction of the system. Wiring the decomposer
in (one function call) produced the first LLM-assisted verified solves.

### 4. miniF2F is the wrong target

The benchmark is saturated by SOTA systems. Even a significant improvement
(28% -> 50%) would still be 50pp behind. The architecture may be better
tested on problems where decomposition matters more — multi-step proofs
that automation tactics can't solve in one shot.

### 5. Speed without wiring is waste

35 commits built everything. But the benchmark runner only used the search
tree with hardcoded tactics — maybe 20% of the system. The remaining 80%
(decomposer, sketch generator, multi-agent coordinator, LLM integration)
sat unused for a month.

---

## Current status (as of 2026-03-19)

### What works (verified)

- Heuristic search: 28.6% on 35-problem sample
- With decomposer: 40% on 10-problem subset (first signal)
- Inline verification: 0 false positives in results
- Multi-step proofs: 3 verified (2-4 tactics each)
- GLM-5 decompositions: mathematically sensible strategies

### Known blockers (March 2026 — most resolved by Phase 3 refactor)

- ~~REPL concurrency for parallel subgoal solving~~ — moot: decomposer deleted
- ~~`induction...simp_all` false positive factory (#10)~~ — closed
- ~~Decomposer timeouts on complex problems (subgoal search too slow)~~ — moot: decomposer deleted

### Issue status (rechecked 2026-05-13 against `gh issue list`)

| # | Issue | Status |
|---|-------|--------|
| #1 | Re-run miniF2F full split after REPL pipe-corruption fix | **CLOSED** |
| #2 | Fix proof_code context (imports + open + preamble) | **CLOSED** |
| #3 | Tactic blocklist for REPL false positives | **CLOSED** |
| #4 | Pass@N sampling | **CLOSED** (implemented in commit `3222a07`; A/B tracked in [#18](https://github.com/0bserver07/bourbaki/issues/18)) |
| #5 | Add polyrith tactic | **CLOSED** |
| #6 | Inline `lean_prover` verification during search | **CLOSED** |
| #7 | Trained value model | **CLOSED** (parked — current loop uses typed feedback, no learned value) |
| #8 | RL-trained tactic policy | **CLOSED** (parked — same reason) |
| #9 | Verification guardrails | **CLOSED** |
| #10 | Blocklist `induction…simp_all` | **CLOSED** |
| #11 | Status tracker — inflated results correction | **Open** (note: in this Mar-19 doc the same number was misused for REPL concurrency; that issue was retired with the decomposer) |
| #12 | Decomposer subgoals timeout | **CLOSED** |
| #13 | pydantic_ai Anthropic adapter crashes on z.ai responses | **CLOSED** (commit `66cba4c`) |
| #14 | Full 244-problem miniF2F run with the new loop | **Open** — v0.3.0 release blocker |
| #15 | Tag v0.3.0 + retract v0.2.1 release notes | **Open** (GitHub releases already retitled "RETRACTED") |
| #16 | Wire ProverLoop into `run_putnam` | **CLOSED** (commit `87114ef`) |
| #17 | `mathlib_search` proposer tool A/B | **Open** |
| #18 | Pass@N live A/B (N=4 on the 35-problem subset) | **Open** |

---

## Commit timeline (extended 2026-05-13)

Pre-Phase-2 history (decomposer era):

```
Feb 09  e641f2c  let there be proof
Feb 18  29205c1  v0.2.1: 94.3% claimed              <-- inflated, RETRACTED
Feb 19  43d3f21  PutnamBench audit: 0% verified      <-- crash
Feb 23  a8e8c3e  miniF2F verified: 63/244 (25.8%)    <-- honest baseline
Mar 08  1e6f199  REPL pipe corruption fix
Mar 08  e4d23d4  Revert blog post
Mar 14  21e67da  Fix proof_code (#2)
Mar 14  51ea8dd  Tactic blocklist (#3)
Mar 14  173365b  Inline verification (#6)
Mar 14  52e4531  Verification guardrails (#9)
Mar 18  6bbd34c  expand() fix + verified run (10/35 = 28.6%)
Mar 19  c385ad9  Decomposer wired in (4/10 = 40%)
```

Proposer-builder-reviewer refactor (April-May):

```
Apr 25  49211ce  Scaffold prover module (Phase 1)
Apr 25  9d8adbb  Proposer node (Phase 2)
Apr 25  6beda20  Builder node (Phase 2)
Apr 25  fdcb76b  Reviewer + memory strategies (Phase 2)
Apr 25  c06e006  ProverLoop driver + benchmark wiring (Phase 2)
Apr 25  aa99595  Post-Phase-2 review fixes
Apr 25  8bc16ec  Builder strips imports from preamble too
Apr 25  ef364ee  Five A/B fixes turning 50% → 90% on 10-problem
May 09  aa4e0af  Replace v0.2.1's inflated claims with honest history
May 09  4ef9398  Wire mathlib_search as proposer tool (Phase 4)
May 09  3222a07  Pass@N sampling
May 09  2113629  Delete legacy autonomous/ pipeline (Phase 3, -6,576 / +118)
May 11  677cd3e  Refresh ARCHITECTURE / README / REALITY_CHECK for new loop
May 11  04dea47  Light-theme SVG redesign
May 13  ddd6edc  Switch z.ai routing to OpenAI-compat endpoint (#13)
May 13  87114ef  Wire ProverLoop into run_putnam (#16)
May 13  fd3066f  pyright cleanup on putnam.py after #16
May 13  66cba4c  Shim pydantic_ai args_as_dict for z.ai responses (#13)
```

---

*This report documents what happened, what we tried, what worked, and what
didn't. The May 13 superseded note at the top of this file points to the
docs that describe the post-refactor state.*

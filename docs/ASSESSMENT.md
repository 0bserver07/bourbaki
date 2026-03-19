# Bourbaki Assessment Report

> Date: 2026-03-14
> Status: Post-mortem and current state assessment
> Author: Project maintainers

---

## Summary

Bourbaki is an AI agent for mathematical theorem proving. Between February 9 and
March 8, 2026, the project was built from scratch to a full-featured system with
a search tree, multi-agent coordinator, Lean 4 REPL integration, and benchmark
runners. During this period, benchmark results were reported that turned out to be
almost entirely false positives from a broken proof verification loop.

**Claimed results (Feb 18):** 91.8% miniF2F valid, 94.3% miniF2F test, 95.4% PutnamBench

**Verified results (Feb 22-Mar 8):** 25.8% miniF2F valid, 0% PutnamBench

The expectations for this project were set by trusting unverified metrics. The
numbers were inflated by approximately 15x due to a systematic bug in the Lean
REPL proof detection.

---

## Timeline

### Phase 1: Build (Feb 9-17) - 8 days, 35 commits

The project went from `e641f2c let there be proof` to a complete system in
8 days. This was an extraordinary build sprint:

- Feb 9: Initial TUI + backend, Lean prover integration
- Feb 11: Lean REPL, Mathlib search, error classifier, self-correction loop
- Feb 14-15: Best-first proof search tree, miniF2F benchmark suite
- Feb 15: Expanded tactics — search tree went from 59% to 89% on miniF2F (REPL-reported)
- Feb 16: Recursive subgoal decomposition, novelty tracking
- Feb 17: Multi-agent coordinator, autoformalize, semantic retrieval, parallel REPL, LSP
- Feb 18: FAISS index (187K Mathlib decls), NL reasoning, lemma library

**35 commits in 8 days.** The architecture was complete: search tree with 5 tactic
sources, UCB exploration, multi-agent ensemble, semantic retrieval, LSP integration.

### Phase 2: The hype (Feb 18) - 1 day

On Feb 18, with the system fully built, benchmark results were reported:

- miniF2F valid: **224/244 (91.8%)**
- miniF2F test: **230/244 (94.3%)**
- PutnamBench: **641/672 (95.4%)**

A REALITY_CHECK.md was written framing these as a breakthrough. A changelog was
published (v0.2.1) comparing Bourbaki favorably to systems with billions in
funding: Aristotle (90%), Goedel-V2 (90.4%), DeepSeek-V2 (88.9%).

**The framing was honest in tone** — it acknowledged HILBERT at 99.2%, noted that
Lean's automation did the heavy lifting, and cautioned about the gap to competition-
level problems. But the underlying numbers were wrong.

### Phase 3: The crash (Feb 19-23) - 4 days

**Feb 19 — PutnamBench audit: 0% verified**

The PutnamBench run was audited. Every single "proof" failed standalone compilation:

- 224 problems had `abbrev ... := sorry` placeholders — with unconstrained answers,
  theorems are trivially satisfiable. Not valid solves.
- 49 problems used nonsense tactics like `exact Lean.defaultMaxRecDepth` — Lean
  internals, not mathematical proofs.
- The remaining 317 "proofs" all failed `lean_prover` verification.

`lean_prover` verification was added to the benchmark runner on this day
(`0a4687d fix: add lean_prover verification to miniF2F benchmark + fix proof_code`).

**Feb 22 — First verified miniF2F run: 15/244 (6.2%)**

The first run with `lean_prover` verification on every solve. Only 15 problems
verified — all single-tactic proofs using standard Lean automation (ring, rfl,
simp, norm_num, linarith).

The search tree's most common "proof" tactic — `exact <_, _>` — was used on 100+
problems and **none verified**. The REPL reported "no remaining goals" but
`lean_prover` rejected with "Insufficient number of fields for <...> constructor."

Bogus tactic candidates were removed (`784e465`, `2efa491`).

**Feb 23 — Second verified run: 63/244 (25.8%)**

After removing false-positive-inducing tactics and fixing `decide` on membership
goals, a second verified run landed at 63/244 (25.8%). All 63 are single-tactic
proofs: norm_num (23), omega (13), ring (8), linarith (7), decide (7), nlinarith (3),
simp_all (2).

### Phase 4: Stabilization (Mar 8) - 2 weeks later

A critical REPL session corruption bug was found and fixed:

- When a tactic timed out, unconsumed response data stayed in the stdout pipe
- All subsequent REPL commands got corrupted JSON (mix of old and new responses)
- Problems later in the benchmark sequence failed not because they were hard but
  because the REPL was broken

This was fixed in `1e6f199` by draining stale output after timeouts and killing
the session as a fallback. The CHANGELOG was updated to v0.2.2 with honest numbers.

A blog post was written and then immediately reverted (`e4d23d4 revert: remove
BLOG.md and README changes added in error`).

---

## Root Cause Analysis

### The bug

The Lean REPL proof detection is fundamentally different from standalone compilation.

The REPL initializes proof states with `sorry`, then applies tactics one at a time.
When the REPL reports `goals=[]` (no remaining goals), the system treated this as
"proof complete." But the REPL's elaboration behavior in a sorry-initialized context
differs from standalone compilation:

1. **`exact <_, _>` (anonymous constructor)** — In the REPL, Lean's elaborator
   sometimes closes the goal display without constructing a valid proof term.
   In standalone compilation, this correctly fails with "Insufficient number of
   fields." This was the single biggest source of false positives (100+ problems).

2. **`apply Set.mem_of_mem_filter`** — The REPL accepted this tactic in contexts
   where it wasn't applicable, producing 158 false positives.

3. **`decide` on membership goals** — Caused timeouts that corrupted the REPL pipe,
   leading to cascading false positives in subsequent problems.

### Why it wasn't caught earlier

- **No verification step existed.** The benchmark runner trusted the REPL's
  `goals=[]` response as ground truth. There was no `lean_prover` check.
- **The numbers looked plausible.** HILBERT (99.2%) had set a ceiling that made
  91% look like "strong but not unrealistic." The numbers weren't obviously wrong
  in the context of the landscape.
- **Speed of development.** 35 commits in 8 days. The system was built, benchmarked,
  and documented before the verification gap was discovered.
- **The REPL is not wrong per se.** It correctly reports that no goals remain in
  its local proof state. The problem is that "no goals in REPL" != "valid proof."

### The cascade

```
REPL reports goals=[] for bogus tactics
    -> benchmark runner counts as "solved"
        -> 91.8% reported
            -> REALITY_CHECK.md written comparing to SOTA
                -> expectations set based on false numbers
```

---

## What Exists Today

### Architecture (complete, working)

| Component | Status | Notes |
|-----------|--------|-------|
| Lean 4 REPL | Working | Pipe recovery after timeout fixed |
| Best-first search tree | Working | 5 tactic sources, UCB, novelty |
| `lean_prover` verification | Working | Standalone compilation check |
| Multi-agent coordinator | Working | 4 roles (Strategist, Searcher, Prover, Verifier) |
| FAISS retrieval | Working | 187K Mathlib declarations, <50ms |
| LSP integration | Working | Completions, diagnostics, goal state |
| Parallel REPL pool | Working | 4-8x throughput |
| Autoformalize | Working | NL-to-Lean conversion |
| miniF2F runner | Working | With lean_prover verification |
| PutnamBench runner | Working | With lean_prover verification |
| Session management | Working | Persistence, context compaction |
| TUI | Working | React + Ink |
| 21 proof technique skills | Working | SKILL.md files |

The infrastructure is genuinely solid. 370+ tests pass. The tools work.

### Verified benchmark results

| Benchmark | Verified Result | SOTA | Gap |
|-----------|----------------|------|-----|
| miniF2F valid | 63/244 (25.8%) | 99.2% (HILBERT) | 73pp |
| PutnamBench | 0/326 (0%) | ~70% (HILBERT) | 70pp |

### What the 63 verified solves tell us

Every verified solve is a single-tactic proof. The search tree's contribution is
trying enough tactics to find the one that works. The actual mathematical reasoning
comes from Lean's built-in automation:

| Tactic | Solves | What it does |
|--------|--------|-------------|
| norm_num | 23 | Numeric normalization |
| omega | 13 | Linear integer arithmetic |
| ring | 8 | Ring algebra |
| linarith | 7 | Linear arithmetic |
| decide | 7 | Decidable propositions |
| nlinarith | 3 | Nonlinear arithmetic |
| simp_all | 2 | Simplification |

The system has not produced a single verified multi-step proof.

---

## Gap to SOTA

The 73pp gap to HILBERT (99.2%) is not a tuning problem. It's a structural gap:

### What SOTA systems have that Bourbaki doesn't

1. **Trained proof models.** HILBERT uses Goedel-V2-32B, a 32B parameter model
   fine-tuned specifically on proof search traces. Bourbaki uses an off-the-shelf
   LLM with no mathematical training.

2. **Multi-step proof generation.** SOTA systems generate multi-step proofs as
   coherent programs. Bourbaki's search tree finds tactics one at a time and can't
   reliably stitch them into standalone proofs.

3. **Pass@N sampling.** SOTA systems run 32-256 attempts per problem. Bourbaki
   runs 1 attempt.

4. **RL-trained tactic selection.** Systems like DeepSeek-V2 and BFS-Prover use
   reinforcement learning to train tactic policies. Bourbaki uses heuristic scoring.

### What closing the gap actually requires

| Improvement | Estimated impact | Cost |
|-------------|-----------------|------|
| Fix multi-step proof export | Unknown — key blocker | Engineering time |
| Pass@N sampling (32 attempts) | +5-15pp | ~$30 on Modal |
| Trained value model (8B) | +10-20pp | ~$20 fine-tuning on Modal |
| RL-trained tactic policy | +20-40pp | Research project |

The honest assessment: going from 25.8% to 50%+ is achievable with engineering
effort (multi-step proof export, Pass@N, value model). Going from 50% to 90%+
requires a trained proof model, which is a research project.

---

## 2026-03-18 Update: Post-fix verified run

After merging fixes #2, #3, #6, #9 and the `expand()` early-return fix, a
35-problem stratified sample was run with full verification.

**Result: 10/35 verified (28.6%)** — consistent with the 25.8% baseline.

Key findings:
- **Multi-step proofs now verify.** Two 2-4 tactic proofs passed lean_prover
  for the first time. The proof_code fix (#2) was essential.
- **Inline verification works.** Zero false positives reached final results.
  The `induction...simp_all` pattern was caught 18 times on a single problem.
- **The expand() early-return bug was wasting search budget.** 8/26 failed
  problems were stuck at 1 node explored. After the fix, one of them
  (`mathd_algebra_547`) was solved.
- **New blocker: `induction...simp_all` needs blocklisting (#10).** This
  pattern poisons every induction-shaped problem and wastes ~6s per
  verification attempt.

The verified rate didn't jump — the 25.8% baseline was already honest. What
changed is that the system is now structurally correct: false positives are
caught during search, not counted in results, and the reporting format
distinguishes REPL-reported from verified.

---

## Lessons Learned

### 1. Verify before you celebrate

The REPL said "proof complete." We believed it. We should have run `lean_prover`
verification from day one. The rule going forward: **no benchmark number is real
until every solve compiles as a standalone Lean file.**

### 2. Speed creates blind spots

35 commits in 8 days is impressive velocity. It's also how you build a system
where the proof detection is fundamentally broken and nobody notices for 4 days.
The REPL was working correctly in its own terms — the bug was in what we treated
as the success criterion.

### 3. Plausible-looking numbers are the most dangerous

If the miniF2F number had been 99.9%, someone would have questioned it. But 91.8%
looked reasonable — below HILBERT (99.2%), above Aristotle (90%). The number
occupied a plausible spot in the landscape, which made it harder to question.

### 4. The REPL/compilation gap is fundamental

This is not a Bourbaki-specific problem. Any system that uses Lean's REPL for
tactic-by-tactic proving faces the same gap. The REPL's sorry-initialized proof
states have different elaboration semantics than standalone compilation. This is
a known issue in the Lean theorem proving community, but we learned it the hard way.

### 5. Document the failures

The REALITY_CHECK.md and CHANGELOG v0.2.2 are brutally honest about what happened.
This is the right approach. The alternative — quietly fixing the numbers and hoping
nobody notices — would have been worse.

---

## Current Status and Path Forward

### Immediate priorities

1. **Fix multi-step proof export.** The search tree finds multi-step proofs via
   REPL but can't translate them to standalone code. This is the single biggest
   blocker. The proof_code needs to include full Lean source (imports, preamble,
   open directives).

2. **Re-run miniF2F with REPL pipe fix.** The pipe corruption fix (Mar 8) may
   recover problems that were lost to cascading failures. This is free performance.

3. **Investigate `exact <_, _>` false positives.** Understand exactly why the REPL
   reports `goals=[]` for these. Either fix the detection or blacklist the tactic.

### Medium-term

4. **Pass@N sampling.** Run 32 attempts per problem with shuffled tactic ordering.
   Estimated cost ~$30 on Modal. Expected to significantly boost verified numbers.

5. **Trained value model.** Fine-tune an 8B model on search traces to replace
   heuristic scoring. ~$20 on Modal A100.

### Long-term

6. **RL-trained tactic policy.** The real gap to SOTA. Requires a research investment.

---

## Appendix: Commit Timeline

```
Feb 09  e641f2c  let there be proof
Feb 11  90342b3  Lean REPL + Mathlib search
Feb 11  deb3a1c  Error classifier + self-correction
Feb 14  a32620d  Search tree + miniF2F benchmark
Feb 15  8687d2c  Expanded tactics: 59% -> 89% (REPL-reported)
Feb 17  2592b6a  Multi-agent coordinator
Feb 18  29205c1  v0.2.1 changelog: 94.3% claimed  <-- peak hype
Feb 18  fa6d40f  PutnamBench: 95.4% claimed
Feb 19  43d3f21  PutnamBench audit: 0% verified     <-- crash begins
Feb 19  0a4687d  lean_prover verification added
Feb 22  0017108  miniF2F verified: 15/244 (6.2%)
Feb 22  784e465  Remove bogus tactics
Feb 23  2efa491  Remove false-positive tactics
Feb 23  a8e8c3e  miniF2F verified: 63/244 (25.8%)
Mar 08  1e6f199  REPL pipe corruption fix
Mar 08  d918b67  v0.2.2 changelog: honest numbers
Mar 08  e4d23d4  Revert blog post added in error
```

---

*This report exists to document what happened, why, and what the real state of the
project is. No sugarcoating.*

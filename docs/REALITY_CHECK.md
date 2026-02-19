# Reality Check

An honest assessment of where Bourbaki stands, what the results mean,
and what they don't. Updated as we go.

---

## 2026-02-18 — v0.2.1 (94.3% miniF2F test)

### What's genuinely notable

- **94.3% on miniF2F test split** using an off-the-shelf general-purpose LLM
  (GLM-5) with no fine-tuning, no custom training data, no reinforcement learning.

- **The baseline (92.6% without the LLM) is pure tactic search** — no neural
  model at all, just a heuristic search tree over Lean tactics. That baseline
  alone beats Aristotle (90%) and ties with Goedel-Prover V2 (90.4%), systems
  that use dedicated 200B+ transformers or fine-tuned 32B models.

- **The architecture is a coding agent with tools**, not a specialized theorem
  proving model. Best-first search + Lean REPL + off-the-shelf LLM coordination.
  That's a fundamentally different (and more general) approach than training a
  dedicated prover.

- **Open source, reproducible.** No proprietary training data, no private model
  weights. Anyone with Lean 4 + Mathlib + an LLM API key can reproduce the results.

### What to be honest about

- **HILBERT is at 99.2%.** The remaining 5pp gap is real and hard. It likely
  requires a trained value/policy model (Goedel-V2-32B) and significantly more
  compute per problem.

- **Our search tree benefits from Lean/Mathlib being very good.** Many of the 226
  baseline solves are Lean's automation (norm_num, omega, simp, ring, linarith)
  doing the heavy lifting, not Bourbaki. The search tree's contribution is
  systematic exploration of tactic combinations, not mathematical insight.

- **The multi-agent coordinator only adds 4-6 problems per split.** The LLM
  contribution is relatively small compared to the search tree. Most of the
  result comes from tactic automation, not from language model reasoning.

- **miniF2F is a well-studied benchmark.** Being near the top on miniF2F doesn't
  mean we can handle IMO competition problems the way Aristotle or AlphaProof
  can. Those systems solve open problems; we solve formalized textbook/competition
  problems where the answer is known.

- **We haven't published a paper or had peer review.** The reference systems
  (HILBERT, Aristotle, DeepSeek-Prover) have published at NeurIPS, ICML, etc.
  Our results are self-reported benchmark runs.

- **The test split (94.3%) is stronger than the valid split (91.8%).** This is
  unusual — typically you'd expect similar or slightly worse performance on the
  test split. It could indicate variance or that the test split problems happen
  to align better with our tactic generators.

### The honest framing

> "94.3% on miniF2F test split, competitive with Aristotle and Goedel-V2,
> using search + off-the-shelf LLM, no fine-tuning."

That's the claim. Nothing more, nothing less.

### What this validates

A small open-source project with a coding-agent architecture and no custom
training can be competitive with well-funded labs on the standard formal
theorem proving benchmark. Good search + good tools + off-the-shelf LLM
coordination gets you surprisingly far.

### Where the gap is

| Gap | Impact | What would close it |
|-----|--------|-------------------|
| Trained value model | +3-5pp | Fine-tune 8B+ model on proof search traces (needs GPU) |
| Pass@N sampling | +1-2pp | Run 32-256 attempts per problem, take first success (needs compute) |
| Deeper decomposition | +1-2pp | More LLM sketch iterations, better subgoal splitting |
| Competition-level reasoning | qualitative | Test-time training (Aristotle), RL (DeepSeek) |

### Comparison table

| System | miniF2F Valid | miniF2F Test | Model | Training | Open Source |
|--------|-------------|-------------|-------|----------|-------------|
| HILBERT | 99.2% | — | Goedel-V2-32B | Fine-tuned | Yes (Apache 2.0) |
| **Bourbaki** | **91.8%** | **94.3%** | **GLM-5 (off-the-shelf)** | **None** | **Yes (MIT)** |
| Goedel-Prover V2 | 90.4% | — | Custom 8B/32B | Self-correction RL | Partial |
| Aristotle | 90% | — | Custom 200B | MCGS + test-time training | No |
| DeepSeek-Prover V2 | 88.9% | — | DeepSeek-V2 | GRPO RL | No |

---

## 2026-02-19 — PutnamBench Audit (0% verified)

### What happened

We ran PutnamBench (672 Putnam competition problems). The initial run reported
95.4% (641/672). Audit revealed this was entirely inflated:

- **224 answer-sorry problems** had `abbrev ... := sorry` placeholders. With an
  unconstrained answer, theorems are trivially satisfiable. Not valid solves.
- **49 suspicious tactics** like `exact Lean.defaultMaxRecDepth` — Lean internals,
  not mathematical proofs.
- **All 317 remaining "proofs" failed lean_prover verification.**

### The REPL false positive problem

Our search tree uses the Lean REPL to detect proof completion. The REPL reported
"no remaining goals" for tactics like `simp`, `exact mem_of`, `norm_num` on
PutnamBench problems. But when the same code was compiled as a standalone Lean
file via `lean_prover`, every single proof was rejected.

This is a **systematic false positive** specific to PutnamBench's more complex
formulations. The REPL's sorry-initialized proof states have different elaboration
context than standalone compilation. Tactics that appear to close goals in the
REPL don't actually produce valid proof terms.

### Verified results

| Benchmark | REPL-reported | lean_prover verified | Status |
|-----------|--------------|---------------------|--------|
| miniF2F valid | 224/244 (91.8%) | **Pending verification** | Spot checks pass |
| miniF2F test | 230/244 (94.3%) | **Pending verification** | Spot checks pass |
| PutnamBench (theorem-only) | 317/326 | **0/326 (0%)** | All false positives |
| PutnamBench (answer-sorry) | 224/346 | Excluded | Need answer generation |

### Why miniF2F is probably fine but PutnamBench isn't

miniF2F problems are simpler — standard competition math with straightforward
type signatures. The REPL detection works correctly for these. Spot-checked
`norm_num` proofs verified successfully with lean_prover.

PutnamBench problems use deeper Mathlib types (MeasureTheory, Topology,
EuclideanSpace), complex dependent types, and longer theorem statements. The
gap between REPL elaboration and standalone compilation is wider for these.

### Lessons learned

1. **Always verify with lean_prover.** REPL-based detection is an approximation.
2. **Set verification timeout to 150s+.** Mathlib import alone takes 75-100s.
3. **Answer-sorry problems need answer generation.** Can't just skip them.
4. **Filter non-proof tactics.** Lean internals aren't mathematical proofs.
5. **PutnamBench is genuinely harder.** HILBERT's 70% with a trained 32B model
   puts our 0% in context — these problems are beyond tactic automation.

### What this means for claimed results

The miniF2F numbers (91.8% valid, 94.3% test) should be re-verified with
lean_prover to produce bulletproof numbers. Until then, they should be treated
as "REPL-reported" with the caveat that REPL detection has known false positive
issues on complex problems.

---

*This document exists to keep us honest. Update it whenever results change.*

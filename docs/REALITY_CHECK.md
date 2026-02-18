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

*This document exists to keep us honest. Update it whenever results change.*

---
name: formalize-informal-proof
description: Convert an informal mathematical proof into verified Lean 4 code. Triggers on "formalize", "convert to Lean", "make rigorous", "verify this proof", or when user pastes informal proof text
---

# Formalize Informal Proof

Convert natural language or textbook proofs into verified Lean 4 code.

## Step 1: Parse the Informal Proof

- [ ] Identify the theorem statement
- [ ] List each logical step
- [ ] Identify implicit assumptions
- [ ] Flag gaps or "clearly/obviously" claims

Create a step map:
| Step | Informal | Implicit assumptions |
|------|----------|---------------------|
| 1 | "Let x be..." | x exists |
| 2 | "Then clearly..." | [needs justification] |
| ... | ... | ... |

## Step 2: Identify Required Imports

Based on concepts used:
- Number theory → `Mathlib.NumberTheory.*`
- Algebra → `Mathlib.Algebra.*`
- Analysis → `Mathlib.Analysis.*`
- Combinatorics → `Mathlib.Combinatorics.*`

## Step 3: Write Lean Structure

```lean
import Mathlib

-- State the theorem
theorem name (hypotheses) : conclusion := by
  -- Step 1: ...
  sorry
  -- Step 2: ...
  sorry
```

### Tools to use:
- `lean_prover`: Verify structure compiles

## Step 4: Fill Each Step

For each informal step:
1. Identify the Lean tactic equivalent
2. Check if Lean accepts it
3. If error: add intermediate lemmas

| Informal | Lean tactic |
|----------|-------------|
| "Let x = ..." | `let x := ...` or `obtain ⟨x, hx⟩ := ...` |
| "Assume P" | `intro h` (if goal is P → Q) |
| "By definition" | `rfl` or `unfold ...` |
| "By arithmetic" | `ring` or `omega` or `linarith` |
| "By induction" | `induction n with k ih` |
| "Clearly" | `simp`, `trivial`, or needs actual proof |

## Step 5: Handle Gaps

When the informal proof says "clearly" or "obviously":
- [ ] Try automated tactics: `simp`, `ring`, `omega`, `decide`
- [ ] If fails: This is a real gap
- [ ] Either find the proof or report to user

## Step 6: Validate Complete Proof

- [ ] No `sorry` remains
- [ ] Proof compiles clean
- [ ] Add comments linking to informal steps

### Output Format

```
**Formalization Report**

**Original theorem:** [as stated informally]

**Formalized statement:**
```lean
theorem ...
```

**Step-by-step translation:**

| Informal step | Lean tactic | Notes |
|---------------|-------------|-------|
| ... | ... | ... |

**Complete Lean proof:**
```lean
[full annotated code]
```

**Gaps found:** [if any]
- [description of gap and how it was resolved]
```

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| "WLOG" (without loss of generality) | `wlog` tactic or explicit case split |
| "By symmetry" | `symm` or prove both directions |
| Implicit type coercions | Add explicit casts |
| "Sufficiently large N" | Use `Filter.Eventually` |

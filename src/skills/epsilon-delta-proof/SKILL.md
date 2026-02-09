---
name: epsilon-delta-proof
description: Prove limit statements using epsilon-delta definitions. Triggers on "epsilon-delta", "limit", "prove continuous", "show limit equals", "formal limit proof"
---

# Epsilon-Delta Proof

Prove limit statements using the rigorous epsilon-delta definition.

## The Definition

For a limit: lim_{x→a} f(x) = L

**Formal statement:** ∀ε>0, ∃δ>0 such that 0 < |x-a| < δ ⟹ |f(x)-L| < ε

## Step 1: Set Up the Framework

- [ ] Identify f(x), a, and L
- [ ] State what we need to prove
- [ ] Write: "Let ε > 0 be given. We need to find δ > 0 such that..."

### Template
```
Let ε > 0 be given.
We need to find δ > 0 such that:
  whenever 0 < |x - a| < δ, we have |f(x) - L| < ε.
```

## Step 2: Work Backwards

- [ ] Start with |f(x) - L|
- [ ] Simplify and bound in terms of |x - a|
- [ ] Find the relationship between ε and δ

### Common Pattern
If |f(x) - L| ≤ C · |x - a| for some constant C, then choose δ = ε/C.

## Step 3: Choose δ

Based on your analysis:
- [ ] δ must be positive
- [ ] δ may depend on ε
- [ ] Sometimes need δ = min(1, ε/C) to handle multiple constraints

## Step 4: Verify Forward

- [ ] Assume 0 < |x - a| < δ
- [ ] Show step-by-step that |f(x) - L| < ε
- [ ] Each inequality must be justified

## Step 5: Lean Formalization

```lean
-- Limit definition in Lean/Mathlib
-- Uses Filter.Tendsto and nhds (neighborhood)
import Mathlib.Topology.Basic

example (f : ℝ → ℝ) (a L : ℝ) :
    Filter.Tendsto f (nhds a) (nhds L) ↔
    ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - L| < ε := by
  sorry -- This is the relationship
```

Key tactics:
- `norm_num` - numerical bounds
- `linarith` - linear inequalities
- `nlinarith` - nonlinear (limited)
- `abs_sub_abs_le_abs_sub` - triangle inequality

## Common Examples

**Example 1:** lim_{x→2} 3x = 6
- |3x - 6| = 3|x - 2|
- Choose δ = ε/3

**Example 2:** lim_{x→0} x² = 0
- Need |x² - 0| < ε when |x| < δ
- If |x| < 1 and |x| < √ε, then |x²| < ε
- Choose δ = min(1, √ε)

## Output Format

```
**Limit:** lim_{x→a} f(x) = L

**Proof:**

Let ε > 0 be given.
Choose δ = [formula].

Verification:
Suppose 0 < |x - a| < δ.
Then:
  |f(x) - L| = ...
            ≤ ...
            < ε ✓

Therefore, lim_{x→a} f(x) = L. ∎
```

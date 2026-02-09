---
name: proof-by-induction
description: Prove statements about natural numbers using mathematical induction. Triggers on "prove for all n", "prove for every integer", "show that for n ≥", statements with recursive/iterative structure
---

# Proof by Induction

Prove a statement P(n) holds for all natural numbers n ≥ base.

## Step 1: Formalize the Statement

- [ ] Identify P(n) precisely
- [ ] Identify base case value (usually 0 or 1)
- [ ] Write Lean theorem signature

Example:
```lean
theorem sum_odds (n : ℕ) : (∑ k in Finset.range n, 2*k + 1) = n^2 := by
  sorry
```

### Tools to use:
- `lean_prover`: Verify theorem statement is well-formed

## Step 2: Base Case

- [ ] State what we need to prove: P(base)
- [ ] Compute the value concretely
- [ ] Verify with symbolic_compute
- [ ] Prove in Lean

Base case tactics to try:
- `simp` - simplification
- `ring` - polynomial arithmetic
- `decide` - decidable propositions
- `rfl` - reflexivity

### Tools to use:
- `symbolic_compute`: Verify base case numerically
- `lean_prover`: Prove base case formally

## Step 3: Inductive Step

- [ ] Assume P(k) holds (inductive hypothesis `ih`)
- [ ] Goal: prove P(k+1)
- [ ] Strategy: Express P(k+1) in terms of P(k)

In Lean:
```lean
induction n with k ih
-- Base case: prove P(0)
-- Inductive case: ih : P(k), goal: P(k+1)
```

Key tactics:
- `induction n with k ih` - start induction
- `simp [ih]` - use inductive hypothesis
- `ring` - solve polynomial equations
- `omega` - linear integer arithmetic
- `rw [ih]` - rewrite using hypothesis

### Tools to use:
- `lean_prover`: Execute induction and prove steps

## Step 4: Verify Complete Proof

- [ ] Ensure no `sorry` remains
- [ ] Full proof compiles without errors
- [ ] Explain each step

### Output Format

```
**Theorem:** [Statement in natural language]

**Proof by induction on n:**

**Base case (n = 0):**
[Explanation]
P(0) = ... = ... ✓

**Inductive step:**
Assume P(k): [hypothesis]
Prove P(k+1): [goal]

[Explanation of key insight]

P(k+1) = ...
       = ... [using P(k)]
       = ... ✓

**Lean proof:**
[Full Lean code with comments]
```

## Common Patterns

| Pattern | Tactic |
|---------|--------|
| Sum formula | `simp [Finset.sum_range_succ]; ring` |
| Product formula | `simp [Finset.prod_range_succ]; ring` |
| Divisibility | `use ...; ring` |
| Inequality | `linarith` or `nlinarith` |

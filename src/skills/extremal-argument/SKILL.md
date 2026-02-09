---
name: extremal-argument
description: Prove results by considering minimal or maximal elements - the extreme case often has special properties. Triggers on "smallest", "largest", "minimal", "maximal", "least", "greatest", "extreme", "consider the minimum"
---

# Extremal Argument

Derive results from properties of minimal/maximal elements.

## The Key Insight

The minimal/maximal element can't be made "more extreme," which constrains what's possible.

## Step 1: Define the Extreme Element

- [ ] What set are we considering?
- [ ] What property defines the set?
- [ ] Are we taking min or max?
- [ ] Why does this extreme element exist? (set is finite, or well-ordered)

```
Let S = { x : x has property P }
Assume S is non-empty.
Let x₀ = min(S) or max(S).
```

## Step 2: Use Extremality

The key: x₀ cannot be improved.

- [ ] If x₀ is minimal: nothing smaller has property P
- [ ] If x₀ is maximal: nothing larger has property P

This means:
- Any x < x₀ must fail property P
- This failure gives us information

## Step 3: Derive Consequences

Common patterns:
- Minimal counterexample → derive contradiction → no counterexamples exist
- Minimal element with property → characterize it precisely
- Maximal structure → can't add more without breaking property

## Step 4: Lean Formalization

Key concepts:
- `Nat.find` - find smallest n with property (if exists)
- `Finset.min' / max'` - min/max of finite set
- Well-founded induction

```lean
-- Minimal counterexample pattern
theorem no_counterexample (P : ℕ → Prop) :
    (∀ n, (∀ k < n, P k) → P n) → ∀ n, P n := by
  intro h
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih => exact h n ih
```

## Classic Examples

**Infinite primes (extremal version):**
- Suppose only finitely many primes
- Let P = largest prime
- Consider N = (product of all primes) + 1
- N has a prime factor p
- p must be in our list (only primes)
- But p divides N and divides product, so p divides 1
- Contradiction

**Minimal criminal:**
- Suppose theorem false for some n
- Let n₀ = smallest counterexample
- Prove n₀ can't be minimal (derive contradiction)
- Therefore no counterexample exists

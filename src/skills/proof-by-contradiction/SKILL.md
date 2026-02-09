---
name: proof-by-contradiction
description: Prove statements by assuming the negation and deriving a contradiction. Triggers on "cannot exist", "impossible", "no such", "prove X is irrational", "infinitely many", "there is no"
---

# Proof by Contradiction

Prove P by assuming ¬P and deriving a contradiction.

## When to Use

- Proving irrationality (√2 is irrational)
- Proving infinitude (infinitely many primes)
- Proving impossibility (no largest prime)
- Proving uniqueness (at most one X exists)

## Step 1: Negate the Statement

- [ ] Original claim: P
- [ ] Assume for contradiction: ¬P
- [ ] Make the negation concrete

| Original | Negation |
|----------|----------|
| √2 is irrational | √2 = p/q for integers p,q |
| Infinitely many primes | Finitely many primes: {p₁,...,pₙ} |
| No X exists | Some X exists |

### Tools to use:
- `symbolic_compute`: Help formalize the negation

## Step 2: Derive Consequences

From ¬P, derive properties:
- [ ] What does ¬P tell us?
- [ ] What can we construct or compute?
- [ ] Look for two conflicting facts

## Step 3: Find the Contradiction

Common contradiction patterns:
- X is both even and odd
- n > n (or n < n)
- Set is both empty and non-empty
- Number is both in set and not in set
- Two different values are equal

## Step 4: Lean Formalization

```lean
theorem statement : P := by
  by_contra h  -- h : ¬P
  -- derive contradiction
  -- ...
  exact absurd fact1 fact2
  -- or: contradiction
```

Key tactics:
- `by_contra h` - assume negation
- `push_neg at h` - simplify negated statement
- `absurd` - derive False from P and ¬P
- `contradiction` - automatic contradiction finder
- `exfalso` - switch to proving False

## Step 5: Verify and Explain

- [ ] Proof compiles
- [ ] Explain the key insight
- [ ] Show where contradiction arises

### Output Format

```
**Theorem:** [Statement]

**Proof by contradiction:**

Assume for contradiction that [¬P].

Then [consequence 1].
And [consequence 2].

But this means [contradiction], which is impossible.

Therefore, [P] must be true. ∎

**Lean proof:**
[Full code with comments]
```

## Classic Examples

**Irrationality of √2:**
- Assume √2 = p/q in lowest terms
- Then 2q² = p², so p is even: p = 2k
- Then 2q² = 4k², so q² = 2k², so q is even
- But p,q both even contradicts "lowest terms"

**Infinitude of primes:**
- Assume only primes are p₁, ..., pₙ
- Let N = p₁ × ... × pₙ + 1
- N is not divisible by any pᵢ
- So N is either prime or has a prime factor not in list
- Contradiction

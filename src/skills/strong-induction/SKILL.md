---
name: strong-induction
description: Prove statements using strong (complete) induction where the inductive hypothesis applies to all smaller cases. Triggers on "strong induction", "complete induction", "inductive hypothesis for all k < n", "well-ordering"
---

# Strong Induction

Also called complete induction. Instead of assuming P(k) to prove P(k+1), we assume P(j) for ALL j < k+1.

## When to Use

- The recurrence depends on multiple previous values (like Fibonacci)
- You need to refer back to cases other than just the immediate predecessor
- The proof naturally requires "all smaller cases"
- Well-ordering arguments

## The Principle

**Standard induction:** P(0) ∧ [∀k, P(k) → P(k+1)] → ∀n, P(n)

**Strong induction:** [∀n, (∀k<n, P(k)) → P(n)] → ∀n, P(n)

Note: Strong induction doesn't require a separate base case—it's built in!
(When n=0, the hypothesis "∀k<0, P(k)" is vacuously true.)

## Step 1: Set Up the Statement

- [ ] Identify P(n)
- [ ] Identify the domain (n ≥ 0, n ≥ 1, etc.)
- [ ] Write the inductive hypothesis: "Assume P(k) holds for all k < n"

## Step 2: The Inductive Step

- [ ] Let n be arbitrary (or n ≥ base case)
- [ ] Assume: ∀k < n, P(k) holds (strong IH)
- [ ] Goal: Prove P(n)

You can now use P(k) for ANY k < n, not just k = n-1.

## Step 3: Case Analysis (Often Needed)

Strong induction often requires cases:
- Base case(s): n = 0, 1, 2, ... (handle directly)
- Inductive case: n ≥ some threshold (use IH)

## Step 4: Lean Formalization

```lean
-- Strong induction in Lean
theorem strong_induction (P : ℕ → Prop)
    (h : ∀ n, (∀ k < n, P k) → P n) :
    ∀ n, P n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    exact h n ih
```

Or use the built-in:
```lean
theorem example (n : ℕ) : P n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    -- ih : ∀ k < n, P k
    -- Goal: P n
    sorry
```

## Output Format

```
**Theorem:** For all n ≥ 0, P(n).

**Proof by strong induction:**

Let n be arbitrary. Assume P(k) holds for all k < n.

**Case n = 0 (or small cases):**
[Direct verification]

**Case n ≥ [threshold]:**
By the strong inductive hypothesis, P(k) holds for all k < n.
In particular, P([specific values we need]).

[Use these to prove P(n)]

Therefore P(n). ∎
```

## Example: Every n ≥ 2 is a product of primes

```
Proof by strong induction on n.

Let n ≥ 2. Assume every integer k with 2 ≤ k < n is a product of primes.

Case 1: n is prime.
Then n is trivially a product of primes (itself).

Case 2: n is composite.
Then n = ab where 2 ≤ a, b < n.
By IH, a is a product of primes: a = p₁...pᵣ
By IH, b is a product of primes: b = q₁...qₛ
Therefore n = p₁...pᵣ · q₁...qₛ is a product of primes. ∎
```

## Comparison with Standard Induction

| Aspect | Standard | Strong |
|--------|----------|--------|
| IH | P(k) | ∀j<n, P(j) |
| Base case | Required separately | Built in |
| Power | Uses only P(n-1) | Uses all P(k) for k<n |
| When to use | Linear recurrences | Non-linear recurrences |

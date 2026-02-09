---
name: sequence-limit
description: Prove sequence limits using epsilon-N definition, monotone convergence, and squeeze theorem. Triggers on "sequence converges", "limit of sequence", "prove a_n tends to", "epsilon-N", "monotone convergence", "squeeze theorem", "bounded monotone"
tags: [proof, analysis]
---

# Sequence Limit

Prove that a sequence converges to a limit using rigorous techniques.

## When to Use

- Establishing that a sequence has a specific limit
- Proving convergence when the limit is unknown
- Applying the squeeze theorem for bounded sequences
- Working with recursively defined sequences

## Step 1: Identify the Approach

- [ ] Write the sequence explicitly: (a_n)_{n=1}^infty
- [ ] Conjecture the limit L (if not given)
- [ ] Choose the appropriate technique

### Technique Selection

| Situation | Technique |
|-----------|-----------|
| Limit L is known/guessed | Epsilon-N definition |
| Sequence is monotone and bounded | Monotone Convergence Theorem |
| Sequence is squeezed between two | Squeeze Theorem |
| Recursive sequence | Show monotone + bounded, find L from recurrence |
| Ratio of sequences | L'Hopital-style or dominant term analysis |

## Step 2: The Epsilon-N Definition

For lim_{n->infty} a_n = L:

**Formal statement:** For all epsilon > 0, exists N in Naturals such that for all n > N, |a_n - L| < epsilon

### Proof Template

```
Let epsilon > 0 be given.
We need to find N such that n > N implies |a_n - L| < epsilon.

[Solve |a_n - L| < epsilon for n]

Choose N = [formula depending on epsilon].

Verification:
For n > N:
  |a_n - L| = ...
            <= ...
            < epsilon

Therefore lim a_n = L.
```

### Finding N

1. Start with |a_n - L| < epsilon
2. Simplify the left side
3. Solve for n
4. N = ceiling of the solution (ensure N is a natural number)

## Step 3: Monotone Convergence Theorem

**Theorem:** Every bounded monotone sequence converges.

- Increasing and bounded above => converges to sup
- Decreasing and bounded below => converges to inf

### Proof Strategy

1. **Show monotone:**
   - Increasing: a_{n+1} >= a_n for all n, OR
   - Decreasing: a_{n+1} <= a_n for all n

2. **Show bounded:**
   - Find M such that |a_n| <= M for all n, OR
   - Find upper bound (if increasing) or lower bound (if decreasing)

3. **Find the limit:**
   - If a_n = f(a_{n-1}), let L = lim a_n
   - Then L = f(L) (taking limits of both sides)
   - Solve for L

## Step 4: Squeeze Theorem

**Theorem:** If a_n <= b_n <= c_n for all n >= N_0, and lim a_n = lim c_n = L, then lim b_n = L.

### Proof Strategy

1. Find lower bound sequence a_n with known limit L
2. Find upper bound sequence c_n with same limit L
3. Verify a_n <= b_n <= c_n
4. Conclude lim b_n = L

## Step 5: Lean Formalization

```lean
import Mathlib.Topology.Sequences
import Mathlib.Topology.Order.MonotoneConvergence

-- Epsilon-N definition
theorem seq_limit_def (a : Nat -> Real) (L : Real) :
    Filter.Tendsto a Filter.atTop (nhds L) <->
    forall epsilon > 0, exists N, forall n >= N, |a n - L| < epsilon := by
  sorry -- Use Metric.tendsto_atTop

-- Monotone convergence (increasing, bounded above)
theorem mono_cvg (a : Nat -> Real) (M : Real)
    (h_mono : Monotone a)
    (h_bdd : forall n, a n <= M) :
    exists L, Filter.Tendsto a Filter.atTop (nhds L) := by
  sorry -- Use tendsto_atTop_ciSup

-- Squeeze theorem
theorem squeeze (a b c : Nat -> Real) (L : Real)
    (hab : forall n, a n <= b n)
    (hbc : forall n, b n <= c n)
    (ha : Filter.Tendsto a Filter.atTop (nhds L))
    (hc : Filter.Tendsto c Filter.atTop (nhds L)) :
    Filter.Tendsto b Filter.atTop (nhds L) := by
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le ha hc hab hbc
```

Key Mathlib lemmas:
- `Metric.tendsto_atTop` - epsilon-N characterization
- `tendsto_atTop_ciSup` - monotone convergence
- `tendsto_of_tendsto_of_tendsto_of_le_of_le` - squeeze theorem

## Common Sequence Limits

| Sequence | Limit | Technique |
|----------|-------|-----------|
| 1/n | 0 | Epsilon-N (N = ceiling(1/epsilon)) |
| 1/n^k (k > 0) | 0 | Epsilon-N or comparison to 1/n |
| r^n (\|r\| < 1) | 0 | Epsilon-N with logarithms |
| n^k / a^n (a > 1) | 0 | Ratio test / exponential dominates |
| (1 + 1/n)^n | e | Monotone convergence (classic!) |
| (1 + x/n)^n | e^x | Generalization of above |
| n^{1/n} | 1 | Squeeze: 1 <= n^{1/n} <= 2 for large n |
| (n!)^{1/n} / n | 1/e | Stirling approximation |
| sqrt(n+1) - sqrt(n) | 0 | Rationalize: 1/(sqrt(n+1) + sqrt(n)) |

## Worked Examples

### Example 1: Prove lim_{n->infty} 1/n = 0

**Method:** Epsilon-N definition

Let epsilon > 0 be given.
We need |1/n - 0| = 1/n < epsilon.
This holds when n > 1/epsilon.

Choose N = ceiling(1/epsilon).

For n > N:
  1/n < 1/N <= epsilon

Therefore lim 1/n = 0.

### Example 2: Prove lim_{n->infty} (1 + 1/n)^n = e

**Method:** Monotone convergence

Let a_n = (1 + 1/n)^n.

**Monotonicity:** a_n is increasing (proven via AM-GM or logarithmic analysis).

**Boundedness:** a_n < 3 for all n.
Proof: By binomial theorem,
a_n = sum_{k=0}^n C(n,k) (1/n)^k
    = sum_{k=0}^n (1/k!) * [n(n-1)...(n-k+1)/n^k]
    < sum_{k=0}^n 1/k!
    < 1 + 1 + 1/2 + 1/4 + 1/8 + ... = 3

By monotone convergence, a_n converges. Define e = lim a_n.

### Example 3: Prove lim_{n->infty} n^{1/n} = 1

**Method:** Squeeze theorem

For n >= 1, clearly n^{1/n} >= 1.

For upper bound, let n^{1/n} = 1 + h_n where h_n >= 0.
Then n = (1 + h_n)^n >= 1 + n*h_n + C(n,2)*h_n^2 >= n(n-1)h_n^2/2

So h_n^2 <= 2/(n-1), giving h_n <= sqrt(2/(n-1)).

Thus: 1 <= n^{1/n} <= 1 + sqrt(2/(n-1))

As n -> infty, the upper bound -> 1.
By squeeze theorem, lim n^{1/n} = 1.

### Example 4: Recursive sequence a_1 = 1, a_{n+1} = sqrt(2 + a_n)

**Method:** Monotone convergence + solve for limit

**Claim:** a_n is increasing and bounded above by 2.

*Boundedness:* By induction. a_1 = 1 < 2.
If a_n < 2, then a_{n+1} = sqrt(2 + a_n) < sqrt(4) = 2.

*Monotonicity:* a_2 = sqrt(3) > 1 = a_1.
If a_{n+1} > a_n, then a_{n+2} = sqrt(2 + a_{n+1}) > sqrt(2 + a_n) = a_{n+1}.

By monotone convergence, let L = lim a_n.

Taking limits: L = sqrt(2 + L)
L^2 = 2 + L
L^2 - L - 2 = 0
(L - 2)(L + 1) = 0

Since a_n > 0, we have L = 2.

## Output Format

```
**Sequence:** a_n = [formula]

**Claim:** lim_{n->infty} a_n = L

**Method:** [Epsilon-N / Monotone Convergence / Squeeze Theorem]

**Proof:**

[For Epsilon-N:]
Let epsilon > 0. Choose N = [formula].
For n > N: |a_n - L| = ... < epsilon.

[For Monotone Convergence:]
Monotonicity: [show a_{n+1} >= a_n or <=]
Boundedness: [show |a_n| <= M]
Limit equation: [if recursive, solve L = f(L)]

[For Squeeze:]
Lower bound: a_n >= [sequence] -> L
Upper bound: a_n <= [sequence] -> L

**Conclusion:** lim a_n = L. //
```

## Common Pitfalls

1. **N must be a natural number:** Always take ceiling when N = f(epsilon) is not an integer.

2. **Verifying monotonicity:** a_{n+1} - a_n >= 0 OR a_{n+1}/a_n >= 1 (for positive terms).

3. **Multiple solutions for L:** When solving L = f(L), check which solution is consistent with the sequence's sign/bounds.

4. **Squeeze requires same limit:** Both bounding sequences must converge to the SAME limit.

5. **Forgetting base case:** In recursive sequences, verify the first few terms satisfy claimed bounds.

6. **Index confusion:** Be careful with n > N vs n >= N. The definition uses "for all n sufficiently large."

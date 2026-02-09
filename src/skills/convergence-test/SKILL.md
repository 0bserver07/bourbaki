---
name: convergence-test
description: Determine convergence or divergence of series using ratio test, root test, comparison test, and integral test. Triggers on "series converges", "test for convergence", "ratio test", "root test", "comparison test", "integral test", "sum of 1/n"
tags: [proof, analysis]
---

# Convergence Test

Determine whether an infinite series converges or diverges using classical tests.

## When to Use

- You have an infinite series Sigma a_n and need to determine convergence
- The series involves factorials, exponentials, or powers
- You need to compare to a known convergent/divergent series
- The terms can be compared to an integrable function

## Step 1: Identify the Series

- [ ] Write the series: Sigma_{n=1}^infty a_n
- [ ] Identify the general term a_n
- [ ] Check basic necessary condition: a_n -> 0
- [ ] Decide which test to try based on the form

### Decision Guide

| Series Form | Recommended Test |
|-------------|------------------|
| Factorials (n!, (2n)!) | Ratio Test |
| nth powers (a^n, n^n) | Root Test |
| Powers of n (1/n^p) | p-Series / Comparison |
| Rational functions | Comparison / Limit Comparison |
| Decreasing positive terms | Integral Test |
| Alternating signs | Alternating Series Test |

## Step 2: The Tests

### Ratio Test

For a_n > 0, compute L = lim_{n->infty} |a_{n+1} / a_n|

| L | Conclusion |
|---|------------|
| L < 1 | Converges absolutely |
| L > 1 | Diverges |
| L = 1 | Inconclusive |

**Best for:** Factorials, exponentials, products

### Root Test

Compute L = lim_{n->infty} |a_n|^{1/n}

| L | Conclusion |
|---|------------|
| L < 1 | Converges absolutely |
| L > 1 | Diverges |
| L = 1 | Inconclusive |

**Best for:** Terms with nth powers, especially n^n

### Comparison Test (Direct)

If 0 <= a_n <= b_n for all n:
- Sigma b_n converges => Sigma a_n converges
- Sigma a_n diverges => Sigma b_n diverges

### Limit Comparison Test

If lim_{n->infty} a_n/b_n = L where 0 < L < infty:
- Sigma a_n and Sigma b_n have the same convergence behavior

### Integral Test

If f(x) is continuous, positive, and decreasing for x >= 1, and a_n = f(n):
- Sigma a_n converges iff integral_1^infty f(x) dx converges

## Step 3: Reference Series

| Series | Convergence | Condition |
|--------|-------------|-----------|
| Sigma 1/n^p (p-series) | p > 1: Converges | p <= 1: Diverges |
| Sigma r^n (geometric) | \|r\| < 1: Converges to 1/(1-r) | \|r\| >= 1: Diverges |
| Sigma 1/n! | Converges to e-1 | Always |
| Sigma 1/n (harmonic) | Diverges | Always |
| Sigma 1/(n ln n) | Diverges | n >= 2 |
| Sigma 1/(n (ln n)^p) | p > 1: Converges | p <= 1: Diverges |

## Step 4: Lean Formalization

```lean
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Basic

-- Ratio test
theorem ratio_test {a : Nat -> Real} (ha : forall n, a n > 0)
    (hL : Filter.Tendsto (fun n => a (n+1) / a n) Filter.atTop (nhds L))
    (hL1 : L < 1) :
    Summable a := by
  sorry -- Use Mathlib's ratio test

-- Geometric series
example (r : Real) (hr : |r| < 1) :
    HasSum (fun n => r^n) (1 / (1 - r)) := by
  exact hasSum_geometric_of_abs_lt_one hr

-- p-series convergence (p > 1)
example (p : Real) (hp : 1 < p) :
    Summable (fun n : Nat => 1 / (n : Real)^p) := by
  sorry -- Use Real.summable_nat_rpow
```

Key Mathlib lemmas:
- `Summable.of_ratio_test` - ratio test
- `hasSum_geometric_of_abs_lt_one` - geometric series
- `Real.summable_nat_rpow_iff` - p-series characterization

## Worked Examples

### Example 1: Sigma n!/n^n

**Test:** Ratio test (factorials and powers)

a_n = n!/n^n, a_{n+1} = (n+1)!/(n+1)^{n+1}

a_{n+1}/a_n = [(n+1)! / (n+1)^{n+1}] * [n^n / n!]
            = (n+1) * n^n / (n+1)^{n+1}
            = n^n / (n+1)^n
            = (n/(n+1))^n
            = (1 - 1/(n+1))^n -> 1/e

Since L = 1/e < 1, the series **converges**.

### Example 2: Sigma 1/n^2

**Test:** p-series with p = 2 > 1

This is a p-series with p = 2.
Since p = 2 > 1, the series **converges** (to pi^2/6).

Alternatively, use comparison: 1/n^2 < 1/(n(n-1)) = 1/(n-1) - 1/n for n >= 2, which telescopes.

### Example 3: Sigma 1/(n ln n)

**Test:** Integral test

Let f(x) = 1/(x ln x) for x >= 2.
f is continuous, positive, and decreasing.

integral_2^infty 1/(x ln x) dx = [ln(ln x)]_2^infty = infty

Since the integral diverges, the series **diverges**.

### Example 4: Sigma (2n)!/(n!)^2 * (1/4)^n (Central binomial)

**Test:** Ratio test

a_n = C(2n,n) / 4^n

a_{n+1}/a_n = [C(2n+2, n+1) / 4^{n+1}] / [C(2n,n) / 4^n]
            = C(2n+2, n+1) / (4 * C(2n,n))
            = (2n+2)!(n!)^2 / ((n+1)!)^2(2n)! * 1/4
            = (2n+2)(2n+1) / (4(n+1)^2)
            = (2n+1) / (2(n+1)) -> 1

Ratio test is inconclusive. Use Stirling or more refined analysis: series diverges.

## Output Format

```
**Series:** Sigma_{n=1}^infty a_n

**Analysis:**

General term: a_n = [formula]
Necessary condition: lim a_n = [value] [check/fail]

**Test Applied:** [Test name]

Computation:
[Show the limit calculation]

L = [value]

**Conclusion:** The series [converges/diverges] because [reason].

**Verification (if converges):**
Sum = [value if known] or bounded by [comparison]
```

## Common Pitfalls

1. **Forgetting the necessary condition:** If a_n does not tend to 0, the series diverges immediately.

2. **Misapplying the ratio test:** When L = 1, the test is inconclusive - you must use another method.

3. **Wrong comparison direction:** To show convergence, bound above by convergent series. To show divergence, bound below by divergent series.

4. **Ignoring the tail:** The first N terms don't affect convergence. Focus on the behavior as n -> infty.

5. **Confusing absolute vs conditional:** Ratio/root tests give absolute convergence. Alternating series may converge conditionally.

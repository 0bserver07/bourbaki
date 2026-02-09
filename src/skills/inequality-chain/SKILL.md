---
name: inequality-chain
description: Prove inequalities using AM-GM, Cauchy-Schwarz, triangle inequality, and chaining techniques. Triggers on "prove inequality", "AM-GM", "Cauchy-Schwarz", "triangle inequality", "show that a <= b", "bound", "estimate"
tags: [proof, analysis]
---

# Inequality Chain

Prove inequalities by combining classical inequalities and chaining bounds.

## When to Use

- Proving one expression is bounded by another
- Optimization problems (finding min/max)
- Establishing estimates in analysis proofs
- Problems involving sums, products, or norms

## Step 1: Classify the Inequality

- [ ] Identify the quantities being compared
- [ ] Determine the constraint set (positive reals, unit sum, etc.)
- [ ] Recognize patterns suggesting specific inequalities
- [ ] Plan the chain of inequalities needed

### Pattern Recognition

| Pattern | Likely Tool |
|---------|-------------|
| Sum vs product | AM-GM |
| Sum of squares, products | Cauchy-Schwarz |
| Absolute values | Triangle Inequality |
| Convex functions | Jensen's Inequality |
| Products <= 1, sums >= n | Weighted AM-GM |

## Step 2: The Classical Inequalities

### AM-GM Inequality (Arithmetic-Geometric Mean)

For non-negative reals a_1, ..., a_n:

**(a_1 + a_2 + ... + a_n) / n >= (a_1 * a_2 * ... * a_n)^{1/n}**

Equality holds iff a_1 = a_2 = ... = a_n.

**Weighted version:** For weights w_i > 0 with sum = 1:
w_1*a_1 + w_2*a_2 + ... + w_n*a_n >= a_1^{w_1} * a_2^{w_2} * ... * a_n^{w_n}

**Two-variable form:** (a + b)/2 >= sqrt(ab) for a, b >= 0

**Useful forms:**
- a + b >= 2*sqrt(ab)
- a + 1/a >= 2 for a > 0
- a^2 + b^2 >= 2ab
- (a + b + c)/3 >= (abc)^{1/3}

### Cauchy-Schwarz Inequality

For real sequences a_1, ..., a_n and b_1, ..., b_n:

**(Sigma a_i * b_i)^2 <= (Sigma a_i^2) * (Sigma b_i^2)**

Equality holds iff a_i = k*b_i for some constant k.

**Integral form:**
(integral f*g)^2 <= (integral f^2) * (integral g^2)

**Engel form (Titu's Lemma):**
a_1^2/b_1 + a_2^2/b_2 + ... + a_n^2/b_n >= (a_1 + a_2 + ... + a_n)^2 / (b_1 + b_2 + ... + b_n)

**Useful for:** Bounding inner products, proving Holder-type inequalities

### Triangle Inequality

For real numbers or vectors:

**|a + b| <= |a| + |b|**

**Reverse form:** ||a| - |b|| <= |a - b|

**General form:** |Sigma a_i| <= Sigma |a_i|

**In metric spaces:** d(x, z) <= d(x, y) + d(y, z)

### Jensen's Inequality

For a convex function f and weights w_i with sum = 1:

**f(Sigma w_i * x_i) <= Sigma w_i * f(x_i)**

Reversed for concave functions.

**Common applications:**
- f(x) = x^2: (mean)^2 <= mean of squares
- f(x) = e^x: exp(mean) <= mean of exp
- f(x) = ln(x): ln(mean) >= mean of ln (concave)

## Step 3: Chaining Techniques

### The Art of Chaining

Build a sequence: a <= b <= c <= d <= ... <= z

**Strategy:**
1. Start from one side
2. Apply one inequality at a time
3. Each step should simplify or approach the target
4. Keep track of equality conditions

### Common Substitutions

| Constraint | Substitution |
|------------|--------------|
| a + b = S (fixed sum) | a = S/2 + t, b = S/2 - t |
| ab = P (fixed product) | a = sqrt(P)*e^t, b = sqrt(P)*e^{-t} |
| a^2 + b^2 = 1 | a = cos(theta), b = sin(theta) |
| a, b, c > 0, abc = 1 | a = x/y, b = y/z, c = z/x |
| Homogeneous degree k | Normalize: set a + b + c = 1 or abc = 1 |

### Homogenization

If the inequality is homogeneous (same degree on both sides), you can normalize:
- Set the sum to 1 (or n)
- Set the product to 1
- Choose values to simplify

## Step 4: Lean Formalization

```lean
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Pow.Real

-- AM-GM for two terms
theorem am_gm_two (a b : Real) (ha : 0 <= a) (hb : 0 <= b) :
    Real.sqrt (a * b) <= (a + b) / 2 := by
  exact Real.sqrt_mul_le_add_of_sq_le_sq ha hb (by ring_nf; nlinarith [sq_nonneg (a - b)])

-- Cauchy-Schwarz (inner product form)
theorem cauchy_schwarz {n : Nat} (a b : Fin n -> Real) :
    (Finset.univ.sum (fun i => a i * b i))^2 <=
    (Finset.univ.sum (fun i => (a i)^2)) * (Finset.univ.sum (fun i => (b i)^2)) := by
  sorry -- Use inner_mul_le_norm_mul_norm

-- Triangle inequality
example (a b : Real) : |a + b| <= |a| + |b| := abs_add a b
```

Key Mathlib lemmas:
- `Real.add_pow_le_pow_mul_pow_of_sq_le_sq` - AM-GM
- `inner_mul_le_norm_mul_norm` - Cauchy-Schwarz
- `abs_add` - triangle inequality
- `sq_nonneg` - squares are non-negative

## Worked Examples

### Example 1: Prove a + b >= 2*sqrt(ab) for a, b >= 0

**Tool:** AM-GM (two variables)

For a, b >= 0:
(a + b) / 2 >= sqrt(ab)     [AM-GM]
a + b >= 2*sqrt(ab)          [multiply by 2]

Equality iff a = b.

### Example 2: For a, b, c > 0, prove (a + b + c)(1/a + 1/b + 1/c) >= 9

**Tool:** Cauchy-Schwarz (Engel form)

Method 1 - Cauchy-Schwarz:
(sqrt(a)*1/sqrt(a) + sqrt(b)*1/sqrt(b) + sqrt(c)*1/sqrt(c))^2
<= (a + b + c)(1/a + 1/b + 1/c)

LHS = (1 + 1 + 1)^2 = 9

So (a + b + c)(1/a + 1/b + 1/c) >= 9.

Method 2 - AM-GM:
(a + b + c)/3 >= (abc)^{1/3} and (1/a + 1/b + 1/c)/3 >= (1/abc)^{1/3}

Multiply: [(a+b+c)/3][(1/a+1/b+1/c)/3] >= 1
So (a + b + c)(1/a + 1/b + 1/c) >= 9.

Equality iff a = b = c.

### Example 3: Prove |sin(x) - sin(y)| <= |x - y|

**Tool:** Mean Value Theorem + Triangle Inequality

By MVT, sin(x) - sin(y) = cos(c)(x - y) for some c between x and y.

|sin(x) - sin(y)| = |cos(c)| * |x - y| <= 1 * |x - y| = |x - y|

Since |cos(c)| <= 1 for all c.

### Example 4: Minimize a^2 + b^2 subject to a + b = 10

**Tool:** Cauchy-Schwarz or direct AM-GM

By Cauchy-Schwarz:
(a + b)^2 <= 2(a^2 + b^2)     [with (1,1) and (a,b)]
100 <= 2(a^2 + b^2)
a^2 + b^2 >= 50

Minimum is 50 when a = b = 5.

Alternatively: a^2 + b^2 = (a + b)^2 - 2ab = 100 - 2ab.
By AM-GM: ab <= (a+b)^2/4 = 25.
So a^2 + b^2 >= 100 - 50 = 50.

## Output Format

```
**Inequality:** [statement to prove]

**Constraints:** [domain and conditions]

**Strategy:** [which inequalities to use]

**Proof:**

Starting from [LHS/RHS]:
[expression]
<= [apply inequality 1, cite it]
   [intermediate expression]
<= [apply inequality 2]
   ...
= [target expression]

**Equality condition:** [when equality holds]

Therefore [LHS] <= [RHS]. //
```

## Common Pitfalls

1. **Wrong direction:** AM-GM gives lower bound for sum, upper bound for product. Don't flip!

2. **Ignoring equality conditions:** Always state when equality holds - this often reveals the extremal case.

3. **Forgetting non-negativity:** AM-GM requires non-negative terms. Cauchy-Schwarz works for all reals.

4. **Loose bounds:** Chaining too many inequalities can give weak results. Aim for tight bounds.

5. **Not using constraints:** If ab = 1 is given, substitute b = 1/a to reduce variables.

6. **Homogeneity errors:** Check that both sides have the same degree before normalizing.

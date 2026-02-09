---
name: polynomial-proof
description: Prove properties of polynomials including irreducibility, factorization, and minimal polynomials. Use Eisenstein's criterion, reduction mod p, and the factor theorem. Triggers on "irreducible polynomial", "Eisenstein", "minimal polynomial", "factor theorem", "reducible", "polynomial factorization"
tags: [proof, algebra, polynomial, irreducibility]
---

# Polynomial Proof

Prove properties of polynomials including irreducibility, factorization, roots, and minimal polynomials.

## The Key Insight

Irreducibility over a ring depends on the ring. A polynomial irreducible over ℚ may factor over ℝ, ℂ, or 𝔽_p. Criteria like Eisenstein and mod p reduction help determine irreducibility.

## Step 1: Identify the Ring

- [ ] What ring are the coefficients in? (ℤ, ℚ, ℝ, ℂ, 𝔽_p)
- [ ] Is the polynomial monic?
- [ ] What is the degree?

### Key Definitions

**Irreducible over R:** A polynomial f ∈ R[x] is irreducible if:
1. f is not a unit
2. If f = gh, then g or h is a unit

**Over different rings:**
- Over ℤ: units are ±1
- Over ℚ, ℝ, ℂ: units are nonzero constants
- Over 𝔽_p: units are nonzero elements

## Step 2: Check for Easy Factorizations

### Rational Root Theorem
For f(x) = aₙxⁿ + ... + a₁x + a₀ ∈ ℤ[x]:

Possible rational roots: ±(divisors of a₀)/(divisors of aₙ)

If none work, f has no linear factors over ℚ.

### Factor Theorem
α is a root of f(x) ⟺ (x - α) divides f(x)

### Degree Considerations
- deg 1: Always irreducible (over fields)
- deg 2, 3: Irreducible ⟺ no roots
- deg ≥ 4: No roots doesn't imply irreducible

## Step 3: Apply Irreducibility Criteria

### Eisenstein's Criterion

**Theorem:** Let f(x) = aₙxⁿ + ... + a₁x + a₀ ∈ ℤ[x].
If there exists a prime p such that:
1. p ∤ aₙ (p does not divide leading coefficient)
2. p | aᵢ for all i < n (p divides all other coefficients)
3. p² ∤ a₀ (p² does not divide constant term)

Then f is irreducible over ℚ.

**Example:** x⁴ + 10x³ + 15x² + 5x + 5
With p = 5: 5 ∤ 1, 5 | 10, 15, 5, 5, and 25 ∤ 5. ✓

### Shift for Eisenstein

If Eisenstein doesn't apply directly, try substituting x → x + a:

**Example:** f(x) = xᵖ⁻¹ + xᵖ⁻² + ... + x + 1 = (xᵖ - 1)/(x - 1)

Substitute x → x + 1:
g(x) = f(x + 1) = ((x+1)ᵖ - 1)/x = xᵖ⁻¹ + (p choose 1)xᵖ⁻² + ... + p

Apply Eisenstein with prime p. Since f(x) irreducible ⟺ g(x) irreducible, done.

### Reduction Mod p

**Theorem:** If f ∈ ℤ[x] is monic and f̄ (reduction mod p) is irreducible in 𝔽_p[x] with deg(f̄) = deg(f), then f is irreducible over ℚ.

**Procedure:**
1. Choose a prime p
2. Reduce all coefficients mod p
3. Check if the reduced polynomial is irreducible over 𝔽_p

**Caution:** Converse is false. f can be irreducible over ℚ but reducible mod every p.

### Perron's Criterion

If f(x) = xⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₀ ∈ ℤ[x] with a₀ ≠ 0 and:
```
|aₙ₋₁| > 1 + |aₙ₋₂| + ... + |a₀|
```
then f is irreducible over ℚ.

## Step 4: Minimal Polynomials

### Definition
The minimal polynomial of α over F is the monic polynomial m(x) ∈ F[x] of least degree such that m(α) = 0.

### Properties
- m(x) is irreducible over F
- If f(α) = 0 for f ∈ F[x], then m | f
- deg(m) = [F(α) : F] (extension degree)

### Finding Minimal Polynomials

1. Express powers of α: 1, α, α², ... until linear dependence
2. Find the linear relation: c₀ + c₁α + ... + cₙαⁿ = 0
3. The minimal polynomial is xⁿ + (cₙ₋₁/cₙ)xⁿ⁻¹ + ... + (c₀/cₙ)

## Step 5: Lean Formalization

```lean
import Mathlib.RingTheory.Polynomial.Basic
import Mathlib.RingTheory.Polynomial.Eisenstein.Basic

-- Irreducible polynomial
example : Irreducible (X^2 + 1 : ℤ[X]) := by
  sorry -- Need to prove no factorization

-- Eisenstein's criterion in Mathlib
-- Polynomial.irreducible_of_eisenstein_criterion

-- Factor theorem
example (f : ℚ[X]) (a : ℚ) (h : f.eval a = 0) :
    (X - C a) ∣ f := by
  exact Polynomial.dvd_iff_isRoot.mpr h

-- Minimal polynomial
-- minpoly ℚ α gives the minimal polynomial of α over ℚ

variable {F : Type*} [Field F] {E : Type*} [Field E] [Algebra F E]
variable (α : E)

-- Minimal polynomial divides any polynomial with α as root
example (f : F[X]) (h : Polynomial.aeval α f = 0) :
    minpoly F α ∣ f := minpoly.dvd F α h
```

## Example: Prove x⁴ + 1 is Irreducible over ℚ

**Problem:** Show that x⁴ + 1 is irreducible over ℚ.

**Proof:**

**Attempt 1 - Eisenstein:** No prime works directly.

**Attempt 2 - Substitution:** Let g(x) = (x + 1)⁴ + 1.

g(x) = x⁴ + 4x³ + 6x² + 4x + 2

With p = 2:
- 2 ∤ 1 (leading coefficient) ✓
- 2 | 4, 6, 4, 2 ✓
- 4 ∤ 2 ✓

By Eisenstein, g(x) is irreducible over ℚ.

Since g(x) = f(x + 1), and the substitution x → x + 1 is an automorphism of ℚ[x], f(x) is irreducible over ℚ. ∎

## Example: Find Minimal Polynomial of √2 + √3 over ℚ

**Problem:** Find the minimal polynomial of α = √2 + √3.

**Solution:**

Let α = √2 + √3.
α - √2 = √3
(α - √2)² = 3
α² - 2√2α + 2 = 3
α² - 1 = 2√2α
(α² - 1)² = 8α²
α⁴ - 2α² + 1 = 8α²
α⁴ - 10α² + 1 = 0

So α is a root of f(x) = x⁴ - 10x² + 1.

**Verify irreducibility:**
- No rational roots (±1 don't work)
- Suppose f = (x² + ax + b)(x² + cx + d) over ℚ
- Expanding and comparing: bd = 1, a + c = 0, b + d + ac = -10
- This gives b + d - a² = -10 and bd = 1
- Solving: no rational solutions

Therefore x⁴ - 10x² + 1 is the minimal polynomial. ∎

## Output Format

```
**Polynomial:** f(x) = [expression] over [ring]

**Irreducibility Test:**

Method: [Eisenstein/Reduction mod p/Direct/etc.]

[Detailed application of criterion]

**Conclusion:** f(x) is [irreducible/reducible] over [ring].
[If reducible, give factorization]

**Lean Proof:**
[Formal verification]
```

## Common Pitfalls

1. **Wrong ring:** x² - 2 is irreducible over ℚ but not over ℝ
2. **Eisenstein requires monic:** Or at least leading coefficient not divisible by p
3. **Degree 4+:** No roots ≠ irreducible; could factor into quadratics
4. **Mod p reduction:** Degree must be preserved (leading coefficient nonzero mod p)
5. **Minimal polynomial:** Must be monic by definition

## Advanced Techniques

### Cyclotomic Polynomials
The nth cyclotomic polynomial Φₙ(x) is irreducible over ℚ.
- Φₚ(x) = xᵖ⁻¹ + ... + x + 1 for prime p
- Degree: φ(n) (Euler's totient)

### Resultants and Discriminants
- Discriminant Δ(f): Related to repeated roots
- Δ = 0 ⟺ f has repeated roots
- For quadratic: Δ = b² - 4ac

### Newton Polygon
For polynomials over p-adic fields, Newton polygon gives factorization information.

### Algebraic Extensions
If [F(α):F] = n, then the minimal polynomial has degree n.
This connects irreducibility to extension degrees.

---
name: group-homomorphism
description: Prove properties about group homomorphisms, kernels, images, and isomorphisms. Use the First Isomorphism Theorem and properties preserved under homomorphisms. Triggers on "homomorphism", "kernel", "image", "isomorphism theorem", "quotient group", "normal subgroup", "group map"
tags: [proof, algebra, group-theory, homomorphism]
---

# Group Homomorphism Proof

Prove properties of groups using homomorphisms, their kernels and images, and the isomorphism theorems.

## The Key Insight

A homomorphism φ: G → H preserves the group operation: φ(ab) = φ(a)φ(b). This preservation transfers structural properties between groups.

## Step 1: Verify Homomorphism Property

- [ ] State the map φ: G → H explicitly
- [ ] Verify φ(ab) = φ(a)φ(b) for all a, b ∈ G
- [ ] Note: This automatically gives φ(e_G) = e_H and φ(a⁻¹) = φ(a)⁻¹

### Checking Homomorphism

To prove φ is a homomorphism:
```
Let a, b ∈ G.
φ(ab) = ... [compute using definition of φ]
       = ...
       = φ(a)φ(b) ✓
```

### Automatic Consequences

For any homomorphism φ: G → H:
- φ(e_G) = e_H (identity maps to identity)
- φ(a⁻¹) = φ(a)⁻¹ (inverses map to inverses)
- φ(aⁿ) = φ(a)ⁿ for all n ∈ ℤ (powers preserved)

## Step 2: Analyze Kernel and Image

### Kernel
```
ker(φ) = { g ∈ G : φ(g) = e_H }
```

**Key Properties:**
- ker(φ) is always a subgroup of G
- ker(φ) is always a normal subgroup of G
- φ is injective ⟺ ker(φ) = {e_G}

### Image
```
im(φ) = { φ(g) : g ∈ G }
```

**Key Properties:**
- im(φ) is always a subgroup of H
- φ is surjective ⟺ im(φ) = H

### Checking Kernel

To show a ∈ ker(φ):
```
φ(a) = ... = e_H ✓
Therefore a ∈ ker(φ).
```

To show ker(φ) = N for some subgroup N:
```
(⊆) If g ∈ ker(φ), then φ(g) = e_H, so... [show g ∈ N]
(⊇) If g ∈ N, then φ(g) = ... = e_H, so g ∈ ker(φ).
```

## Step 3: Apply Isomorphism Theorems

### First Isomorphism Theorem

**Statement:** If φ: G → H is a homomorphism, then
```
G / ker(φ) ≅ im(φ)
```

**The isomorphism:** g·ker(φ) ↦ φ(g)

**Usage:** To prove G/N ≅ H:
1. Find a surjective homomorphism φ: G → H
2. Show ker(φ) = N
3. Apply First Isomorphism Theorem

### Second Isomorphism Theorem

**Statement:** If H ≤ G and N ⊴ G, then
```
H / (H ∩ N) ≅ HN / N
```

### Third Isomorphism Theorem

**Statement:** If N ⊴ K ⊴ G, then
```
(G/N) / (K/N) ≅ G/K
```

## Step 4: Prove Properties Preserved

### Properties Preserved by Homomorphisms

| Property | If G has it | Then im(φ) has it |
|----------|-------------|-------------------|
| Abelian | ∀a,b: ab=ba | ∀a,b: φ(a)φ(b)=φ(b)φ(a) |
| Cyclic | G = ⟨g⟩ | im(φ) = ⟨φ(g)⟩ |
| Finitely generated | G = ⟨S⟩ | im(φ) = ⟨φ(S)⟩ |
| Divisible | ∀g∃h: hⁿ=g | ∀g'∃h': h'ⁿ=g' |

### Properties Preserved by Isomorphisms

All structural properties are preserved:
- Order of elements
- Number of elements of each order
- Subgroup lattice structure
- Center, commutator subgroup

## Step 5: Lean Formalization

```lean
import Mathlib.GroupTheory.QuotientGroup.Basic
import Mathlib.GroupTheory.Subgroup.Basic

variable {G H : Type*} [Group G] [Group H]

-- Kernel definition
example (φ : G →* H) : φ.ker = { g | φ g = 1 } := rfl

-- Kernel is normal
example (φ : G →* H) : φ.ker.Normal := MonoidHom.normal_ker φ

-- First Isomorphism Theorem
-- QuotientGroup.quotientKerEquivRange gives: G ⧸ ker(φ) ≃* range(φ)

-- Example: If φ is injective, then G ≅ im(φ)
example (φ : G →* H) (h : Function.Injective φ) :
    G ≃* φ.range := by
  exact MulEquiv.ofInjective φ h

-- Proving a map is a homomorphism
def squareMap : ℤ →+ ℤ where
  toFun := fun n => 2 * n  -- This is a homomorphism
  map_zero' := by simp
  map_add' := by intro a b; ring
```

## Example: Prove ℤ/nℤ ≅ Cyclic Group of Order n

**Problem:** Show that ℤ/nℤ is isomorphic to the multiplicative group of nth roots of unity.

**Proof:**

Define φ: ℤ → ℂ* by φ(k) = e^(2πik/n).

**Homomorphism:**
φ(a + b) = e^(2πi(a+b)/n) = e^(2πia/n) · e^(2πib/n) = φ(a)φ(b) ✓

**Kernel:**
φ(k) = 1 ⟺ e^(2πik/n) = 1 ⟺ k/n ∈ ℤ ⟺ n | k ⟺ k ∈ nℤ

So ker(φ) = nℤ.

**Image:**
im(φ) = {e^(2πik/n) : k ∈ ℤ} = μ_n (nth roots of unity)

**By First Isomorphism Theorem:**
ℤ/nℤ = ℤ/ker(φ) ≅ im(φ) = μ_n ∎

## Output Format

```
**Claim:** [Statement about homomorphism/groups]

**The homomorphism:**
φ: G → H
φ(g) = [formula]

**Verification:**
φ(ab) = ... = φ(a)φ(b) ✓

**Kernel:** ker(φ) = [description]
[Proof that this is the kernel]

**Image:** im(φ) = [description]
[Proof of surjectivity if relevant]

**Conclusion:**
[Apply isomorphism theorem or derive result]

**Lean Proof:**
[Formal verification]
```

## Common Pitfalls

1. **Forgetting to verify homomorphism:** Always check φ(ab) = φ(a)φ(b)
2. **Kernel is in domain:** ker(φ) ⊆ G, not in H
3. **Quotient requires normal:** G/N only works when N ⊴ G; for homomorphisms, ker is automatically normal
4. **Isomorphism vs. homomorphism:** Isomorphism requires bijectivity
5. **Order considerations:** |G/ker(φ)| = |im(φ)|, and |G| = |ker(φ)| · |im(φ)| when G is finite

## Advanced Techniques

### Finding Homomorphisms

To find all homomorphisms G → H:
1. Generators of G must map to elements whose relations are satisfied in H
2. If G = ⟨g | gⁿ = e⟩, then φ(g) must satisfy φ(g)ⁿ = e in H

### Exact Sequences

A sequence G₁ →^φ G₂ →^ψ G₃ is exact at G₂ if im(φ) = ker(ψ).

Short exact sequence: 1 → N → G → G/N → 1

### Recognizing Quotients

If you have ker(φ) = N, then im(φ) tells you what G/N looks like without computing cosets.

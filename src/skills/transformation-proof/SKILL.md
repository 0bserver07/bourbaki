---
name: transformation-proof
description: Prove geometric theorems using transformations (translations, rotations, reflections, dilations). Identify invariants preserved under transformations. Triggers on "rotation", "reflection", "translation", "symmetry", "transform", "invariant", "isometry", "congruent by transformation"
tags: [proof, geometry, transformation, symmetry]
---

# Transformation Proof

Prove geometric statements by applying transformations and analyzing what properties are preserved.

## The Key Insight

Transformations map figures to congruent or similar figures while preserving key properties. If we can transform one configuration into another, they share all preserved properties.

## Step 1: Identify the Transformation

### Isometries (Preserve Distance)

| Transformation | Description | Fixed Points |
|---------------|-------------|--------------|
| **Translation** | Slide by vector v | None (unless v = 0) |
| **Rotation** | Rotate by angle θ around center O | Center O only |
| **Reflection** | Flip across line ℓ | All points on ℓ |
| **Glide Reflection** | Reflect then translate along ℓ | None |

### Similarities (Preserve Shape)

| Transformation | Description | Ratio |
|---------------|-------------|-------|
| **Dilation** | Scale by factor k from center O | k |
| **Spiral Similarity** | Rotate and dilate from center | k |

### What Each Preserves

| Property | Translation | Rotation | Reflection | Dilation |
|----------|-------------|----------|------------|----------|
| Distance | Yes | Yes | Yes | No (scaled) |
| Angle measure | Yes | Yes | Yes | Yes |
| Orientation | Yes | Yes | No | Yes |
| Parallelism | Yes | Yes | Yes | Yes |
| Area | Yes | Yes | Yes | No (k²) |

## Step 2: Set Up the Transformation

- [ ] Identify what to transform (which figure or points)
- [ ] Choose transformation parameters (center, angle, axis, vector)
- [ ] Determine where key points map

### Notation
- T_v: Translation by vector v
- R_O,θ: Rotation by angle θ about point O
- S_ℓ: Reflection across line ℓ
- D_O,k: Dilation with center O and ratio k

### Key Relationships

**Rotation:** If R_O,θ(A) = A', then:
- OA = OA' (radii)
- ∠AOA' = θ

**Reflection:** If S_ℓ(A) = A', then:
- ℓ is perpendicular bisector of AA'
- If A is on ℓ, then A' = A

**Dilation:** If D_O,k(A) = A', then:
- O, A, A' are collinear
- OA'/OA = k

## Step 3: Apply and Analyze

- [ ] Apply transformation to relevant points/figures
- [ ] Identify what maps to what
- [ ] Use preservation properties to draw conclusions

### Common Proof Strategies

**Show congruence:** Find an isometry mapping one figure to another.

**Show similarity:** Find a similarity transformation connecting figures.

**Use symmetry:** If figure has symmetry, use the symmetry transformation.

**Composition:** Combine transformations (e.g., two reflections = rotation).

## Step 4: Lean Formalization

```lean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

-- Rotation in ℝ²
def rotate2 (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ,
   p.1 * Real.sin θ + p.2 * Real.cos θ)

-- Reflection across x-axis
def reflectX (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Rotation preserves distance from origin
theorem rotate_preserves_norm (θ : ℝ) (p : ℝ × ℝ) :
    (rotate2 θ p).1^2 + (rotate2 θ p).2^2 = p.1^2 + p.2^2 := by
  simp [rotate2]
  ring_nf
  rw [Real.cos_sq_add_sin_sq]
  ring

-- Two reflections = rotation
theorem two_reflections_rotation (p : ℝ × ℝ) :
    reflectX (reflectX p) = p := by
  simp [reflectX]
```

## Example: Prove Angle Sum using Rotation

**Problem:** Prove that the angles of a triangle sum to 180°.

**Proof using transformations:**

Consider triangle ABC. We perform rotations:

1. Rotate 180° about the midpoint M_AB of side AB
2. Rotate 180° about the midpoint M_BC of side BC

After both rotations, vertex A maps to a point A'', and:
- The angles at A, B, C together form a straight line at the image
- Therefore ∠A + ∠B + ∠C = 180°

## Example: Prove Regular Hexagon Property using Rotation

**Problem:** In regular hexagon ABCDEF, prove AC = 2·AB.

**Proof:**

The regular hexagon has 6-fold rotational symmetry about its center O.

By rotational symmetry (R_O,60°):
- All sides are equal: AB = BC = CD = DE = EF = FA
- O is equidistant from all vertices

Connect O to all vertices, creating 6 equilateral triangles.
Each central angle is 360°/6 = 60°.
Since OA = OB = AB (equilateral), triangle OAB is equilateral.

Now for diagonal AC:
- Triangle OAC contains angle ∠AOC = 120° (two 60° central angles)
- OA = OC (radii)
- By the Law of Cosines or noting this creates two equilateral triangles:
  AC = AB√3...

Actually, using the 30-60-90 triangle formed: AC = 2 · AB · cos(30°) = AB√3.

## Output Format

```
**Theorem:** [Statement]

**Transformation Approach:**
[Identify which transformation to use and why]

**Setup:**
- Transform: [Description with parameters]
- Maps: [A → A', B → B', ...]

**Analysis:**
[Use preserved properties to derive conclusion]

**Conclusion:**
[Final statement]

**Lean Proof:**
[Formal verification]
```

## Common Pitfalls

1. **Wrong center/axis:** The transformation parameters must be chosen so points map correctly
2. **Orientation issues:** Reflections reverse orientation; compositions of two reflections preserve it
3. **Not checking well-defined:** Ensure the transformation is properly defined (e.g., rotation angle, dilation center not on the figure when k ≠ 1)
4. **Forgetting fixed points:** The fixed points of a transformation often play a key role

## Advanced Techniques

### Composition Rules

| Composition | Result |
|-------------|--------|
| Two reflections (parallel axes) | Translation |
| Two reflections (intersecting axes) | Rotation (angle = 2× angle between axes) |
| Three reflections | Glide reflection |
| Rotation + Dilation (same center) | Spiral similarity |

### Symmetry Arguments

If a figure is invariant under transformation T, then:
- Any property preserved by T holds for corresponding parts
- Fixed points of T lie on symmetry elements

### Finding the Transformation

Given two congruent figures:
- Same orientation → rotation or translation
- Opposite orientation → reflection or glide reflection
- Find corresponding points to determine parameters

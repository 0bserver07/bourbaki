---
name: synthetic-construction
description: Prove geometric theorems using classical compass-and-straightedge reasoning. Use auxiliary constructions, congruence (SSS, SAS, ASA, AAS), and similarity arguments. Triggers on "prove congruent", "prove similar", "construct", "auxiliary line", "classical geometry", "Euclidean proof"
tags: [proof, geometry, synthetic, construction]
---

# Synthetic Construction Proof

Prove geometric theorems using classical Euclidean methods: auxiliary constructions, congruence, similarity, and angle chasing.

## The Key Insight

Add helpful lines or points to reveal hidden structure. The right auxiliary construction transforms a difficult problem into an obvious one.

## Step 1: Analyze the Given Figure

- [ ] List all given information
- [ ] Identify what needs to be proved
- [ ] Look for potential congruent or similar triangles
- [ ] Identify any special points (midpoints, incenters, circumcenters)

### Questions to Ask
- Are there equal segments or angles given?
- Is there symmetry in the figure?
- Are there parallel or perpendicular lines?
- What triangles exist or could be formed?

## Step 2: Plan Auxiliary Constructions

Common auxiliary constructions:

| Technique | When to Use |
|-----------|-------------|
| **Extend a line** | To create exterior angles or meet another line |
| **Draw a parallel** | To transfer angles, create similar triangles |
| **Draw a perpendicular** | To create right angles, use Pythagorean theorem |
| **Connect points** | To form triangles from scattered points |
| **Draw a midpoint** | To use midpoint theorem, create isoceles triangles |
| **Circumscribe circle** | When equal angles or cyclic quadrilaterals involved |
| **Drop altitude** | To decompose into right triangles |
| **Draw angle bisector** | To create equal angles, use angle bisector theorem |

### Key Insight
The auxiliary construction should connect what you know to what you need.

## Step 3: Apply Congruence Criteria

### Triangle Congruence (≅)

| Criterion | Conditions |
|-----------|------------|
| **SSS** | Three pairs of equal sides |
| **SAS** | Two sides and included angle equal |
| **ASA** | Two angles and included side equal |
| **AAS** | Two angles and non-included side equal |
| **HL** | Hypotenuse and leg (right triangles only) |

### Triangle Similarity (~)

| Criterion | Conditions |
|-----------|------------|
| **AA** | Two pairs of equal angles |
| **SAS~** | Two sides proportional, included angle equal |
| **SSS~** | All three sides proportional |

### From Congruence/Similarity

- Congruent triangles → corresponding parts equal (CPCTC)
- Similar triangles → corresponding sides proportional

## Step 4: Execute the Proof

- [ ] State each construction clearly
- [ ] Justify each step with a theorem or given
- [ ] Clearly mark the congruence/similarity
- [ ] Extract the conclusion using CPCTC or proportions

### Common Theorems to Cite

**Angle Theorems:**
- Vertical angles are equal
- Alternate interior angles (parallel lines) are equal
- Corresponding angles (parallel lines) are equal
- Angles in a triangle sum to 180°
- Exterior angle equals sum of remote interior angles

**Line Theorems:**
- Midpoint theorem: Line joining midpoints is parallel and half the third side
- Angle bisector theorem: Divides opposite side in ratio of adjacent sides
- Triangle inequality: Sum of any two sides > third side

**Circle Theorems:**
- Inscribed angle = half central angle
- Angles in same arc are equal
- Tangent perpendicular to radius

## Step 5: Lean Formalization

```lean
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Euclidean.Angle.Oriented.Basic

-- Example: Isoceles triangle has equal base angles
-- Using Euclidean geometry in Lean
theorem isoceles_base_angles_equal
    {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (A B C : V)
    (h : dist A B = dist A C) :
    -- angle ABC = angle ACB
    InnerProductGeometry.angle (B - A) (B - C) =
    InnerProductGeometry.angle (C - A) (C - B) := by
  sorry
```

### Useful Mathlib Concepts
- `EuclideanGeometry` for geometric proofs
- `dist` for distances
- `angle` for angles
- Congruence expressed via equal distances and angles

## Example: Prove Base Angles of Isoceles Triangle are Equal

**Given:** Triangle ABC with AB = AC
**Prove:** ∠ABC = ∠ACB

**Proof:**

*Construction:* Draw the angle bisector from A to BC, meeting BC at D.

*In triangles ABD and ACD:*
1. AB = AC (given)
2. AD = AD (common side)
3. ∠BAD = ∠CAD (AD is angle bisector)

By SAS, △ABD ≅ △ACD.

Therefore, ∠ABD = ∠ACD (CPCTC), i.e., ∠ABC = ∠ACB. ∎

## Output Format

```
**Theorem:** [Statement]

**Given:** [List given information]

**To Prove:** [Goal]

**Construction:** [Auxiliary elements added]

**Proof:**
[Step-by-step reasoning]

1. [Statement] ... [Justification]
2. [Statement] ... [Justification]
...

∴ △XYZ ≅ △ABC by [criterion]

**Conclusion:** [Final statement with CPCTC or proportions]

**Lean Proof:**
[Formal verification]
```

## Common Pitfalls

1. **Assuming what you're proving:** Don't use the conclusion in your proof
2. **Wrong congruence criterion:** SSA (two sides and non-included angle) is NOT valid in general
3. **Unclear construction:** State exactly what you're constructing and why it exists
4. **Missing justification:** Every step needs a theorem or given as support
5. **Diagram dependency:** Your proof must work even without the picture

## Advanced Techniques

### Proof by Construction of Congruent Copy
To prove AB = CD, construct triangle on CD congruent to triangle containing AB.

### Angle Chasing
Chain together angle relationships:
∠A = ∠B (alternate interior) = ∠C (vertical) = ∠D (corresponding)

### Auxiliary Circles
- Circumcircle through three points
- Incircle tangent to all sides
- Excircle opposite a vertex

### Spiral Similarity
For similar triangles with common vertex, track the spiral similarity transformation.

---
name: coordinate-proof
description: Prove geometric theorems using coordinate geometry. Convert shapes to algebraic expressions, use distance formula, midpoint formula, and slope calculations. Triggers on "prove using coordinates", "analytic geometry", "distance formula", "show collinear", "prove perpendicular", "prove parallel"
tags: [proof, geometry, coordinates, analytic]
---

# Coordinate Proof

Prove geometric statements by placing figures in a coordinate system and using algebraic computations.

## The Key Insight

Geometric relationships become algebraic equations. Distance, angle, and position relationships can all be verified through computation.

## Step 1: Choose a Coordinate System

- [ ] Place the origin strategically (often at a vertex or center)
- [ ] Align axes with sides or symmetry lines
- [ ] Use general coordinates (a, b) not specific numbers for generality

### Strategic Placement Guidelines

| Figure | Recommended Placement |
|--------|----------------------|
| Triangle | One vertex at origin, one side along x-axis |
| Rectangle | Center at origin, sides parallel to axes |
| Circle | Center at origin |
| Parallelogram | One vertex at origin, one side along x-axis |

Example for triangle ABC:
```
A = (0, 0)
B = (c, 0)      -- on x-axis, distance c from origin
C = (a, b)      -- general point
```

## Step 2: Assign Coordinates

- [ ] Label all vertices with coordinates
- [ ] Use symmetry to simplify when possible
- [ ] Keep coordinates general (parameters) for general proofs
- [ ] For special triangles (isoceles, right), use appropriate constraints

### Common Setups

**Isoceles triangle** (symmetric about y-axis):
```
A = (-a, 0), B = (a, 0), C = (0, h)
```

**Right triangle** (right angle at origin):
```
A = (0, 0), B = (a, 0), C = (0, b)
```

**Parallelogram**:
```
A = (0, 0), B = (a, 0), C = (a+b, c), D = (b, c)
```

## Step 3: Compute Algebraically

Use these fundamental formulas:

### Distance Formula
```
d(P₁, P₂) = √[(x₂-x₁)² + (y₂-y₁)²]
```

### Midpoint Formula
```
M = ((x₁+x₂)/2, (y₁+y₂)/2)
```

### Slope Formula
```
m = (y₂-y₁)/(x₂-x₁)
```

### Key Relationships

| Property | Algebraic Condition |
|----------|-------------------|
| Parallel lines | m₁ = m₂ |
| Perpendicular lines | m₁ · m₂ = -1 |
| Collinear points | Equal slopes between pairs |
| Equal distances | d₁ = d₂ |
| On a circle | (x-h)² + (y-k)² = r² |

## Step 4: Derive the Result

- [ ] Substitute coordinates into formulas
- [ ] Simplify algebraic expressions
- [ ] Verify the required relationship holds
- [ ] Use `symbolic_compute` for complex calculations

### Tools to use:
- `symbolic_compute`: Simplify algebraic expressions
- `lean_prover`: Verify formal proof in Lean

## Step 5: Lean Formalization

```lean
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Geometry.Euclidean.Basic

-- Distance in ℝ²
def dist2 (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

-- Midpoint
def midpoint2 (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1)/2, (p.2 + q.2)/2)

-- Slope (when x₁ ≠ x₂)
def slope (p q : ℝ × ℝ) : ℝ :=
  (q.2 - p.2) / (q.1 - p.1)

-- Example: Midpoint of hypotenuse equidistant from all vertices
theorem midpoint_equidistant (a b : ℝ) (ha : a > 0) (hb : b > 0) :
    let A : ℝ × ℝ := (0, 0)
    let B : ℝ × ℝ := (a, 0)
    let C : ℝ × ℝ := (0, b)
    let M := midpoint2 B C
    dist2 M A = dist2 M B ∧ dist2 M B = dist2 M C := by
  sorry -- Proof by computation
```

## Example: Prove Diagonals of Rectangle Bisect Each Other

**Problem:** Show that the diagonals of a rectangle bisect each other.

**Setup:**
```
A = (0, 0)
B = (a, 0)
C = (a, b)
D = (0, b)
```

**Proof:**
Diagonal AC: from (0,0) to (a,b)
Midpoint of AC = ((0+a)/2, (0+b)/2) = (a/2, b/2)

Diagonal BD: from (a,0) to (0,b)
Midpoint of BD = ((a+0)/2, (0+b)/2) = (a/2, b/2)

Both diagonals have the same midpoint (a/2, b/2), so they bisect each other.

## Output Format

```
**Theorem:** [Geometric statement]

**Coordinate Setup:**
[Coordinate assignments with justification]

**Computation:**
[Step-by-step algebraic work]

**Conclusion:**
[How computation proves the theorem]

**Lean Proof:**
[Formal verification]
```

## Common Pitfalls

1. **Overly specific coordinates:** Using (3, 4) instead of (a, b) proves only one case
2. **Loss of generality:** Placing a general triangle with one vertex at (0, 0) is fine; forcing two vertices on axes over-constrains
3. **Division by zero:** Check for vertical lines (undefined slope) separately
4. **Forgetting constraints:** If proving for rectangles, coordinates must satisfy perpendicularity

## Advanced Techniques

### Vector Form
For some proofs, vectors are cleaner:
- Position vector: **r** = (x, y)
- Displacement: **AB** = **B** - **A**
- Dot product: **u** · **v** = 0 ⟺ perpendicular

### Parametric Lines
Line through P in direction **d**:
```
(x, y) = P + t·d = (p₁ + t·d₁, p₂ + t·d₂)
```

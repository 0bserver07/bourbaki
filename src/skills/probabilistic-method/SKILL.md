---
name: probabilistic-method
description: Prove existence using probability - if a random object has the property with positive probability, such an object must exist. Triggers on "probabilistic", "random", "expected value", "exists by probability", "Erdos", "non-constructive existence"
---

# Probabilistic Method

Prove existence by showing positive probability.

**Pioneered by:** Paul Erdős, who used it extensively.

## Core Principle

1. Define a random process that generates objects
2. Show P(object has desired property) > 0
3. Conclude: such an object exists

Note: This is NON-CONSTRUCTIVE. You prove existence without building the object.

## Technique A: Basic Probabilistic Method

### Step 1: Define the Random Space
- [ ] What's the sample space? (all possible objects)
- [ ] What's the probability distribution?

### Step 2: Compute the Probability
- [ ] What event E corresponds to "has desired property"?
- [ ] Show P(E) > 0

### Step 3: Conclude
- [ ] If P(E) > 0, some object in sample space satisfies E

## Technique B: First Moment Method (Expectation)

If E[X] < n, then X < n for some outcome.
If E[X] > 0, then X > 0 for some outcome.

### Step 1: Define the Random Variable
- [ ] X = count of "bad" things (or "good" things)

### Step 2: Compute E[X]
- [ ] Often use linearity: E[X] = Σ P(event_i)

### Step 3: Conclude
- [ ] If E[bad things] < 1, some outcome has 0 bad things
- [ ] If E[good things] > 0, some outcome has good things

## Technique C: Alteration (Delete Bad Parts)

1. Generate random object
2. Count expected bad parts
3. If few bad parts in expectation, delete them
4. What remains still has the property

## Classic Example: Ramsey Lower Bound

**Claim:** There exists a 2-coloring of edges of Kₙ with no monochromatic K_k, for n sufficiently smaller than 2^(k/2).

**Proof:**
- Color each edge red/blue uniformly at random
- For any k-clique, P(monochromatic) = 2 × (1/2)^(k choose 2)
- E[# monochromatic k-cliques] = (n choose k) × 2^(1-(k choose 2))
- If this is < 1, some coloring has no monochromatic clique

## Lean Formalization

Probabilistic proofs in Lean are tricky—often proved via counting arguments instead:

- Instead of "P(E) > 0", prove "∃ x, x satisfies E"
- Use pigeonhole: if bad objects < total objects, good object exists
- Convert probabilistic argument to counting argument

### Tools to use:
- `symbolic_compute`: Calculate probabilities, expected values
- `web_search`: Find related probabilistic proofs

## When to Use

- You need to prove existence (not construct)
- Counting all objects is hard
- The "good" objects seem plentiful but hard to specify
- The problem has Erdős's fingerprints on it

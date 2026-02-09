---
name: counting-argument
description: Prove identities or existence using combinatorial counting - double counting, bijections, inclusion-exclusion. Triggers on "count", "number of ways", "how many", "bijection", "one-to-one correspondence", "combinatorial proof"
---

# Counting Argument

Prove results by counting objects in clever ways.

## Technique A: Double Counting

Count the same set in two different ways → the counts must be equal.

### Step 1: Identify the Set to Count
- [ ] What objects are we counting?
- [ ] Can we count them in two natural ways?

### Step 2: Count Method 1
- [ ] First way of organizing/counting
- [ ] Get expression E₁

### Step 3: Count Method 2
- [ ] Second way of organizing/counting
- [ ] Get expression E₂

### Step 4: Conclude
- [ ] E₁ = E₂ (both count the same thing)

### Classic Example: Handshaking Lemma
Count pairs (person, hand they shook):
- Method 1: Sum of degrees = Σ deg(v)
- Method 2: 2 × (number of edges) = 2|E|
- Therefore: Σ deg(v) = 2|E|

## Technique B: Bijection

Show |A| = |B| by constructing a one-to-one correspondence.

### Step 1: Define the Map
- [ ] f : A → B (describe what f does to each element)

### Step 2: Prove f is Injective
- [ ] If f(a₁) = f(a₂), prove a₁ = a₂

### Step 3: Prove f is Surjective
- [ ] For every b ∈ B, find a ∈ A with f(a) = b

### Step 4: Conclude
- [ ] f is a bijection → |A| = |B|

## Technique C: Inclusion-Exclusion

Count |A ∪ B ∪ C| by adding and subtracting overlaps.

|A ∪ B| = |A| + |B| - |A ∩ B|

|A ∪ B ∪ C| = |A| + |B| + |C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|

### Tools to use:
- `symbolic_compute`: Calculate binomial coefficients, sums
- `sequence_lookup`: Check if counts match known sequences
- `lean_prover`: Finset cardinality lemmas

## Lean Tactics

- `Finset.card_union_eq` - for disjoint sets
- `Finset.card_sdiff` - for set difference
- Bijection: construct `Equiv` type

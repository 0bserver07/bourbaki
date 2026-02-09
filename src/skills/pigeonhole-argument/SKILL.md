---
name: pigeonhole-argument
description: Prove existence using the pigeonhole principle - when n+1 items go into n categories, some category has multiple items. Triggers on "must exist", "at least two", "some pair", "among any n", "prove existence of duplicate/collision"
---

# Pigeonhole Argument

Prove that something must exist because there are more items than categories.

## The Pigeonhole Principle

If you put n+1 pigeons into n boxes, at least one box contains 2+ pigeons.

**Sounds trivial, but it's surprisingly powerful:**
- Prove there exist two people in London with the same number of hairs
- Prove in any group of 6 people, either 3 are mutual friends or 3 are mutual strangers
- Prove that some Fibonacci number is divisible by 1000

## Step 1: Identify the Setup

- [ ] What are the "pigeons"? (items being placed)
- [ ] What are the "holes"? (categories/boxes)
- [ ] Why are there more pigeons than holes?

| Question | Answer |
|----------|--------|
| Pigeons | [what objects] |
| Holes | [what categories] |
| Count | [n+1 pigeons, n holes] |

## Step 2: Formalize the Argument

- [ ] Define the mapping: how does each pigeon go into a hole?
- [ ] Count pigeons: prove there are n+1 (or more)
- [ ] Count holes: prove there are at most n
- [ ] Apply pigeonhole: conclude some hole has 2+ pigeons

### Tools to use:
- `symbolic_compute`: Count sizes
- `lean_prover`: Formalize with `Finset.card` and pigeonhole lemmas

## Step 3: Extract the Conclusion

The pigeonhole gives you: "some hole has 2+ pigeons"

Translate this back:
- [ ] What does "same hole" mean for the original problem?
- [ ] What property do items sharing a hole have?
- [ ] State the existence result

## Step 4: Lean Formalization

Key lemmas in Mathlib:
- `Fintype.exists_ne_map_eq_of_card_lt` - basic pigeonhole
- `Finset.exists_lt_card_fiber_of_mul_lt_card` - generalized pigeonhole

```lean
-- If |A| > |B|, any f : A → B has a collision
example (h : Fintype.card A > Fintype.card B) (f : A → B) :
    ∃ a₁ a₂, a₁ ≠ a₂ ∧ f a₁ = f a₂ := by
  exact Fintype.exists_ne_map_eq_of_card_lt h f
```

## Common Patterns

| Pattern | Pigeons | Holes |
|---------|---------|-------|
| Residues | Numbers | Remainders mod n |
| Graph coloring | Edges | Colors |
| Birthdays | People | Days of year |
| Subsets | Subsets of {1,...,n} | Possible sums |

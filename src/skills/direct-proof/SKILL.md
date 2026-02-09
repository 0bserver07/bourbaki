---
name: direct-proof
description: Prove statements directly by assuming the hypothesis and deriving the conclusion through logical steps. Triggers on "prove directly", "show that", "demonstrate", "verify", "simple proof"
---

# Direct Proof

The most straightforward proof technique: assume the hypothesis, derive the conclusion.

## When to Use

- The implication P → Q has a clear logical path
- No need for contradiction or cases
- Algebraic manipulations lead directly to the result
- The statement has a constructive flavor

## Step 1: Identify Hypothesis and Conclusion

For "If P then Q" (P → Q):
- [ ] Hypothesis: What are we assuming? (P)
- [ ] Conclusion: What do we need to show? (Q)
- [ ] Hidden assumptions: Types, domains, constraints

### Example
"If n is even, then n² is even"
- Hypothesis: n is even (∃k, n = 2k)
- Conclusion: n² is even (∃m, n² = 2m)

## Step 2: Unpack Definitions

- [ ] What does the hypothesis mean precisely?
- [ ] What does the conclusion require?
- [ ] Introduce variables from existential statements

### Tools to use:
- `symbolic_compute`: Verify algebraic steps

## Step 3: Chain of Reasoning

Build a chain of implications:
```
Assume P.
Then [consequence 1].
Therefore [consequence 2].
...
Thus Q. ∎
```

Each step must be justified by:
- Definition
- Previously proven statement
- Algebraic manipulation
- Known theorem

## Step 4: Lean Formalization

```lean
theorem example (h : P) : Q := by
  -- Unfold definitions
  unfold ...
  -- Get witnesses from existentials
  obtain ⟨k, hk⟩ := h
  -- Provide witnesses for conclusion
  use ...
  -- Algebraic verification
  ring
```

Key tactics for direct proofs:
- `intro h` - assume the hypothesis
- `obtain ⟨x, hx⟩ := h` - unpack existential
- `use x` - provide witness
- `exact h` - conclude with exact match
- `ring` - algebraic simplification
- `simp` - simplification

## Output Format

```
**Theorem:** If P then Q.

**Proof:**

Assume P.
[Unpack what P gives us]

Then [step 1].
[justification]

Therefore [step 2].
[justification]

...

Thus Q. ∎

**Lean proof:**
```lean
[verified code]
```
```

## Common Patterns

| Statement Type | Approach |
|----------------|----------|
| ∃x, P(x) | Construct x explicitly |
| ∀x, P(x) | Let x be arbitrary |
| P ∧ Q | Prove P and Q separately |
| P ↔ Q | Prove P → Q and Q → P |

## Example: n even ⟹ n² even

```
Assume n is even.
By definition, ∃k ∈ ℤ such that n = 2k.

Then:
  n² = (2k)²
     = 4k²
     = 2(2k²)

Let m = 2k². Then n² = 2m, so n² is even. ∎
```

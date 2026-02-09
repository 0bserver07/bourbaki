---
name: explain-proof
description: Explain a proof or Lean code step-by-step for learning. Triggers on "explain", "how does this work", "teach me", "what does this mean", "walk me through", "I don't understand"
---

# Explain Proof

Break down a proof into understandable pieces, explaining each step.

## Approach

1. Start with the big picture
2. Break into chunks
3. Explain each chunk in plain language
4. Connect to the formal notation
5. Check understanding before continuing

## For Lean Code

When explaining Lean tactics:

### Structure
```
**What we're proving:** [goal in plain English]

**Current state:**
- We know: [hypotheses]
- We need: [goal]

**This step:** `tactic_name`
- What it does: [explanation]
- Why here: [reasoning]
- After this: [new goal/state]
```

### Common Tactics Explained

| Tactic | Plain English |
|--------|---------------|
| `intro h` | "Assume the hypothesis, call it h" |
| `apply f` | "Use theorem/function f" |
| `exact h` | "This is exactly what h says" |
| `rw [h]` | "Rewrite using equation h" |
| `simp` | "Simplify using known facts" |
| `ring` | "This follows by algebra" |
| `induction n` | "Prove for 0, then prove n→n+1" |
| `cases h` | "Consider all possibilities for h" |
| `constructor` | "Prove both parts separately" |
| `left` / `right` | "Prove one side of OR" |
| `obtain ⟨x, hx⟩ := h` | "Unpack: h gives us x with property hx" |

## For Informal Proofs

### Structure
```
**The claim:** [what we're proving]

**The approach:** [proof technique: direct, contradiction, induction, etc.]

**Key insight:** [the clever idea that makes it work]

**Step by step:**
1. [first step]
   - Why: [justification]
2. [second step]
   ...
```

## Interactive Elements

After each section:
- "Does this make sense so far?"
- "Want me to go deeper on any part?"
- "Ready to try a similar example?"

## Exercises

After explaining, offer practice:
1. Predict what the next tactic should be
2. Fill in a missing step
3. Prove a similar theorem
4. Explain a step back to me

## Adapting to Level

Ask early: "How familiar are you with [concept]?"
- **Beginner:** More analogies, less notation
- **Intermediate:** Balance of intuition and rigor
- **Advanced:** Focus on subtle points, skip basics

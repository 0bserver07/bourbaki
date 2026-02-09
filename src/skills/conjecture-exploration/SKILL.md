---
name: conjecture-exploration
description: Explore a mathematical conjecture - gather evidence, find counterexamples, or build toward a proof. Triggers on "explore", "investigate", "what if", "is it true that", "find pattern", "check if"
---

# Conjecture Exploration

Systematically investigate a mathematical conjecture before attempting a proof.

## Phase 1: Understand the Statement

- [ ] Parse the conjecture into precise mathematical form
- [ ] Identify the domain (integers, reals, sets, etc.)
- [ ] Identify quantifiers (∀ for all, ∃ there exists)
- [ ] Formalize as: "For all X in DOMAIN, PROPERTY holds"

### Tools to use:
- `symbolic_compute`: Parse and simplify the statement
- `sequence_lookup`: Check if related to known sequences
- `web_search`: Check if this is a known result

## Phase 2: Gather Evidence

- [ ] Test small cases systematically

For integer conjectures, compute:
| n | LHS | RHS | Equal? |
|---|-----|-----|--------|
| 0 | ... | ... | ✓/✗    |
| 1 | ... | ... | ✓/✗    |
| ... | ... | ... | ...   |

- [ ] Test edge cases (0, 1, -1, large numbers)
- [ ] Look for counterexamples
- [ ] Identify patterns in the data

### Tools to use:
- `symbolic_compute`: Calculate values for test cases

## Phase 3: Decide Direction

Based on evidence gathered:

**If all cases pass + pattern is clear:**
→ Suggest proof strategy (induction, direct, etc.)
→ Transition to appropriate proof skill

**If counterexample found:**
→ Report the counterexample with details
→ Ask: "Should we explore a weaker version?"

**If pattern unclear:**
→ Checkpoint with user
→ Share all findings
→ Ask for guidance

## Phase 4: Report Findings

Present structured summary:
1. **Conjecture statement** (formalized)
2. **Evidence table** (test cases)
3. **Patterns observed**
4. **Related sequences/results** (from OEIS/arXiv)
5. **Recommendation** (prove / refine / abandon)

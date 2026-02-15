# Recursive Subgoal Decomposition (HILBERT-Style)

> Date: 2026-02-15
> Scope: Layered — architect for competition-level, validate on miniF2F
> Approach: HILBERT-style (informal sketch -> have/sorry skeleton -> solve each subgoal -> recurse on failures)
> Reference: HILBERT (Apple, 99.2% miniF2F), DeepSeek-Prover-V2, Aristotle, Numina-Lean-Agent

## Context

Bourbaki currently achieves 89% on miniF2F valid (217/244) using best-first tactic search
with heuristic scoring. The remaining 27 unsolved problems require multi-step reasoning
that flat search cannot find within budget. HILBERT achieves 99.2% on miniF2F using
off-the-shelf LLMs (no fine-tuning) with recursive subgoal decomposition — proving this
approach closes the gap without model training.

## Pipeline Overview

```
Theorem statement
      |
      v
+-------------------+
| Sketch Generator  |  <- LLM generates informal proof plan
+---------+---------+
          |
          v
+-------------------+
|  Formalizer       |  <- Converts sketch to Lean have/sorry skeleton
+---------+---------+
          |
          v
+-------------------+
|  Subgoal Solver   |  <- For each sorry: prove_with_search()
+---------+---------+
          |
     +----+----+
     | Success? |
     +----+----+
     yes  |  no
     |    |
     |    v
     |  Recurse: decompose failed subgoal
     |  (max depth 2-3)
     |    |
     v    v
+-------------------+
|  Proof Stitcher   |  <- Compose subgoal proofs into final proof
+-------------------+
```

Integration into search.py._run_loop() as Phase 0:
- Phase 0: Recursive decomposition (NEW)
- Phase 1: Best-first search on the full theorem (EXISTING)
- Phase 2: Strategy rotation fallback (EXISTING)

If Phase 0 solves the theorem, Phases 1-2 are skipped. If decomposition partially
succeeds (some subgoals solved), those become context for Phase 1.

## Component 1: Sketch Generator

File: `backend/bourbaki/autonomous/sketch.py`

Pluggable interface with a default LLM-based implementation.

```python
class ProofSketch:
    strategy: str           # e.g., "induction", "contradiction"
    steps: list[SketchStep] # ordered informal steps
    key_lemmas: list[str]   # Mathlib lemmas the sketch expects to use

class SketchStep:
    statement: str          # NL description: "Show base case n=0"
    formal_type: str | None # Optional Lean type hint
    depends_on: list[int]   # Indices of prior steps (for future DAG extension)

class SketchGenerator(Protocol):
    async def generate(self, theorem: str, context: SketchContext) -> list[ProofSketch]:
        ...

class LLMSketchGenerator(SketchGenerator):
    """Default: uses the configured LLM with a proof-planning prompt."""
    ...
```

The prompt asks the LLM to:
1. Identify the proof strategy
2. Break it into 2-6 intermediate steps
3. For each step, give NL description AND candidate Lean type
4. List Mathlib lemmas needed

Multiple sketches generated for diversity (e.g., induction vs contradiction).
The SketchGenerator protocol is pluggable for future Numina-style generator-verifier.

## Component 2: Formalizer

File: `backend/bourbaki/autonomous/formalizer.py`

Converts a ProofSketch into a Lean skeleton with sorry placeholders.

```python
class FormalizedSkeleton:
    code: str                    # Full Lean code with sorry placeholders
    subgoals: list[Subgoal]      # Extracted subgoals to solve
    compilation_ok: bool         # Whether the skeleton type-checks

class Subgoal:
    index: int                   # Position in the skeleton
    label: str                   # e.g., "step1", "base_case"
    lean_type: str               # The type of the have statement
    lean_context: str            # Lean code preceding this subgoal
    proof_state_id: int | None   # REPL proof state ID (after initialization)
```

Formalizer loop:
1. Generate Lean code from sketch: `have step1 : T1 := by sorry; ...`
2. Send to REPL to type-check
3. If compilation fails, feed Lean error back to LLM (max 2 rounds per Goedel-V2)
4. Extract each sorry position as a Subgoal with its proof state ID

Skeleton doesn't need to be perfect — subgoals that type-check can still be solved
even if others fail, and the skeleton can be revised.

## Component 3: Subgoal Solver + Recursive Decomposition

File: `backend/bourbaki/autonomous/decomposer.py`

```python
class DecompositionConfig:
    max_recursion_depth: int = 2       # Diminishing returns beyond 2-3 (HILBERT)
    max_sketches: int = 3              # Try up to 3 different proof sketches
    subgoal_search_budget: int = 50    # Budget per subgoal
    subgoal_search_timeout: float = 60 # Seconds per subgoal
    formalization_retries: int = 2     # Error-correction rounds for skeleton

class DecompositionResult:
    success: bool
    proof_code: str | None
    subgoals_total: int
    subgoals_solved: int
    solved_subgoal_proofs: dict[str, list[str]]  # label -> tactic list
    failed_subgoals: list[str]
    sketches_tried: int
    recursion_depth_reached: int

async def decompose_and_prove(
    theorem: str,
    config: DecompositionConfig,
    sketch_generator: SketchGenerator,
    session: LeanREPLSession | None = None,
    depth: int = 0,
) -> DecompositionResult:
```

Algorithm:
1. Generate proof sketches via sketch_generator.generate(theorem)
2. For each sketch, formalize into skeleton via formalizer
3. For each subgoal in skeleton:
   - Try prove_with_search(subgoal, budget=50)
   - If fails AND depth < max_recursion_depth: recurse with decompose_and_prove(subgoal, depth+1)
4. If all subgoals solved, stitch proofs and verify
5. If not all solved, try next sketch
6. Return partial results even on failure (for Phase 1 context)

## Component 4: Proof Stitcher

Integrated into decomposer.py.

Replaces each sorry with the tactic sequence found by the subgoal solver:

```lean
-- Before:
theorem foo : P := by
  have step1 : T1 := by sorry
  have step2 : T2 := by sorry
  exact final_step step1 step2

-- After:
theorem foo : P := by
  have step1 : T1 := by
    simp [Nat.add_comm]
    ring
  have step2 : T2 := by
    linarith [step1]
  exact final_step step1 step2
```

Final whole-proof verification via REPL. If stitched proof fails due to context
differences between isolated and composed execution, attempt minor repairs by
re-running subgoal solver with full composed context.

## Integration into search.py

New config fields on AutonomousSearchConfig:
```python
use_decomposition: bool = True
decomposition_max_depth: int = 2
decomposition_max_sketches: int = 3
decomposition_subgoal_budget: int = 50
```

_run_loop() phases:
- Phase 0: decompose_and_prove() [NEW]
- Phase 1: prove_with_search() [EXISTING best-first]
- Phase 2: strategy rotation [EXISTING]

If Phase 0 partially succeeds, solved subgoals and remaining goals passed as
context to Phase 1.

## Bundled Quick-Win Fixes

### Fix modal_runner.py
Register lean_tactic and lean_prover as tools on the strategy agent. Verify
generated code against Lean before returning success.

### Wire up ProofNode.visits for UCB
In expand(), increment node.visits. In best_first_search(), add exploration bonus:
score = base_score - C * sqrt(ln(parent.visits) / (1 + node.visits)) where C=1.0.

### Novelty bonus in scoring
Track seen goal patterns globally. Add bonus (lower score) for nodes reaching
previously-unseen goal states (intrinsic reward from DeepSeek V1.5).

### Raise tool_call_limits
Change defaults to 20+ for lean_tactic, 10+ for lean_prover.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| backend/bourbaki/autonomous/sketch.py | Create | SketchGenerator protocol + LLMSketchGenerator |
| backend/bourbaki/autonomous/formalizer.py | Create | Sketch to Lean skeleton with sorry |
| backend/bourbaki/autonomous/decomposer.py | Create | Recursive decompose_and_prove orchestrator |
| backend/bourbaki/autonomous/search.py | Modify | Add Phase 0, pass partial results to Phase 1 |
| backend/bourbaki/autonomous/search_tree.py | Modify | UCB exploration bonus, novelty scoring |
| backend/bourbaki/autonomous/scoring.py | Modify | Add novelty bonus, seen-state tracking |
| backend/bourbaki/autonomous/modal_runner.py | Modify | Register tools, verify code against Lean |
| backend/bourbaki/config.py | Modify | Add decomposition config, raise tool_call_limits |

## Testing & Validation

- Unit tests: Sketch generation, formalization, stitching (mock REPL)
- Integration test: Run decomposer on 5 known-solvable miniF2F problems
- Benchmark: Re-run miniF2F valid split with decomposition. Target: solve 5-10 of 27 remaining (89% -> 93-95%)
- Regression: Ensure previously-solved problems still solve (decomposition skips to Phase 1 for easy problems)

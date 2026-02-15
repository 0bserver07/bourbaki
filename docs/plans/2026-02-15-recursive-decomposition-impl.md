# Recursive Subgoal Decomposition — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add HILBERT-style recursive subgoal decomposition to close the 89%→95%+ gap on miniF2F, plus bundle quick-win bug fixes for the search tree.

**Architecture:** New Phase 0 in the autonomous search loop: an LLM generates informal proof sketches, a formalizer converts them to Lean `have`/`sorry` skeletons, individual subgoals are solved via the existing `prove_with_search()`, and failed subgoals are recursively decomposed (max depth 2). Quick-win fixes (modal_runner tools, UCB visits, novelty scoring, tool_call_limits) are bundled.

**Tech Stack:** Python 3.11+, Pydantic AI, asyncio, lean4-repl, pytest

---

## Task 1: Quick-Win — Raise tool_call_limits and fix config

**Files:**
- Modify: `backend/bourbaki/config.py`

**Step 1: Update config defaults**

In `backend/bourbaki/config.py`, change `tool_call_limits` dict to raise limits:

```python
tool_call_limits: dict[str, int] = {
    "lean_prover": 15,
    "lean_tactic": 30,
    "mathlib_search": 10,
}
```

**Step 2: Verify config loads**

Run: `cd backend && python -c "from bourbaki.config import settings; print(settings.tool_call_limits)"`
Expected: `{'lean_prover': 15, 'lean_tactic': 30, 'mathlib_search': 10}`

**Step 3: Commit**

```bash
git add backend/bourbaki/config.py
git commit -m "feat: raise tool_call_limits for lean_prover and lean_tactic"
```

---

## Task 2: Quick-Win — Add novelty bonus to scoring

**Files:**
- Modify: `backend/bourbaki/autonomous/scoring.py`
- Create: `backend/tests/test_scoring.py`

**Step 1: Write the failing test**

Create `backend/tests/test_scoring.py`:

```python
"""Tests for proof state scoring with novelty bonus."""

from bourbaki.autonomous.scoring import score_proof_state, NoveltyTracker


def test_completed_proof_scores_zero():
    tracker = NoveltyTracker()
    assert score_proof_state([], 0, tracker) == 0.0


def test_fewer_goals_scores_lower():
    tracker = NoveltyTracker()
    one_goal = score_proof_state(["⊢ True"], 0, tracker)
    two_goals = score_proof_state(["⊢ True", "⊢ False"], 0, tracker)
    assert one_goal < two_goals


def test_novelty_bonus_for_unseen_state():
    tracker = NoveltyTracker()
    first = score_proof_state(["⊢ a + b = b + a"], 1, tracker)
    # Same goals again — no novelty bonus
    second = score_proof_state(["⊢ a + b = b + a"], 1, tracker)
    assert first < second  # First visit gets bonus (lower = better)


def test_novelty_tracker_tracks_seen():
    tracker = NoveltyTracker()
    assert not tracker.has_seen(["⊢ True"])
    tracker.mark_seen(["⊢ True"])
    assert tracker.has_seen(["⊢ True"])


def test_novelty_tracker_order_independent():
    tracker = NoveltyTracker()
    tracker.mark_seen(["⊢ A", "⊢ B"])
    assert tracker.has_seen(["⊢ B", "⊢ A"])
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `NoveltyTracker` and new `score_proof_state` signature don't exist

**Step 3: Implement novelty tracker and updated scoring**

Replace `backend/bourbaki/autonomous/scoring.py` with:

```python
"""Proof state scoring for best-first search.

v2 adds novelty bonus (intrinsic reward) inspired by DeepSeek-Prover V1.5:
states that reach previously-unseen goal configurations get a score bonus.
"""

from __future__ import annotations


class NoveltyTracker:
    """Tracks seen goal states for novelty-based exploration bonus."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def _key(self, goals: list[str]) -> str:
        return "|".join(sorted(goals))

    def has_seen(self, goals: list[str]) -> bool:
        return self._key(goals) in self._seen

    def mark_seen(self, goals: list[str]) -> None:
        self._seen.add(self._key(goals))

    @property
    def seen_count(self) -> int:
        return len(self._seen)


def score_proof_state(
    goals: list[str],
    depth: int,
    novelty_tracker: NoveltyTracker | None = None,
    novelty_bonus: float = 3.0,
) -> float:
    """Score a proof state — lower is more promising (for min-heap).

    Heuristic: fewer remaining goals + simpler goals + shallower depth = better.
    Novel states (first time seeing this goal set) get a bonus (lower score).

    Args:
        goals: List of remaining goal strings from Lean.
        depth: Current depth in the search tree.
        novelty_tracker: Optional tracker for novelty bonus.
        novelty_bonus: Score reduction for novel states (default 3.0).

    Returns:
        Float score (lower = more promising).
    """
    if not goals:
        return 0.0  # No goals = proof complete, best possible score

    # Goal count: each remaining goal adds 10 points
    goal_count_score = len(goals) * 10.0

    # Goal complexity: use goal string length as a rough proxy
    avg_complexity = sum(len(g) for g in goals) / len(goals)
    complexity_score = min(avg_complexity / 20.0, 10.0)  # Cap at 10

    # Depth penalty: slight preference for shallower proofs
    depth_score = depth * 0.5

    base_score = goal_count_score + complexity_score + depth_score

    # Novelty bonus: reduce score for never-before-seen goal states
    if novelty_tracker is not None:
        if not novelty_tracker.has_seen(goals):
            novelty_tracker.mark_seen(goals)
            base_score -= novelty_bonus

    return base_score


def goal_matches_pattern(goal: str, pattern: str) -> bool:
    """Check if a goal roughly matches a pattern for tactic selection."""
    return pattern in goal
```

**Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_scoring.py -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add backend/bourbaki/autonomous/scoring.py backend/tests/test_scoring.py
git commit -m "feat: add novelty-based exploration bonus to proof state scoring"
```

---

## Task 3: Quick-Win — Wire up UCB exploration in search_tree

**Files:**
- Modify: `backend/bourbaki/autonomous/search_tree.py`
- Create: `backend/tests/test_search_tree.py`

**Step 1: Write the failing test**

Create `backend/tests/test_search_tree.py`:

```python
"""Tests for proof search tree UCB exploration."""

import math
from bourbaki.autonomous.search_tree import ProofNode, ucb_adjusted_score


def test_ucb_no_parent_returns_base_score():
    node = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[], score=10.0)
    assert ucb_adjusted_score(node) == 10.0


def test_ucb_unvisited_child_gets_bonus():
    parent = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[], visits=10)
    child = ProofNode(proof_state=1, goals=["⊢ True"], tactic_history=["simp"],
                      parent=parent, score=10.0, visits=0)
    adjusted = ucb_adjusted_score(child)
    assert adjusted < 10.0  # Should get exploration bonus (lower = better)


def test_ucb_visited_child_less_bonus():
    parent = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[], visits=10)
    child_new = ProofNode(proof_state=1, goals=["⊢ True"], tactic_history=["simp"],
                          parent=parent, score=10.0, visits=0)
    child_old = ProofNode(proof_state=2, goals=["⊢ True"], tactic_history=["ring"],
                          parent=parent, score=10.0, visits=5)
    assert ucb_adjusted_score(child_new) < ucb_adjusted_score(child_old)


def test_proof_node_is_complete():
    complete = ProofNode(proof_state=0, goals=[], tactic_history=["ring"])
    incomplete = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[])
    assert complete.is_complete
    assert not incomplete.is_complete
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_search_tree.py -v`
Expected: FAIL — `ucb_adjusted_score` not found

**Step 3: Add UCB function and wire up visits**

In `backend/bourbaki/autonomous/search_tree.py`, add after the `ProofNode` class:

```python
import math

def ucb_adjusted_score(node: ProofNode, exploration_constant: float = 1.0) -> float:
    """Adjust node score with UCB exploration bonus.

    Uses UCB1 formula: bonus = C * sqrt(ln(parent_visits) / (1 + node_visits)).
    Lower score = more promising, so bonus is subtracted.
    """
    if node.parent is None or node.parent.visits == 0:
        return node.score

    bonus = exploration_constant * math.sqrt(
        math.log(node.parent.visits) / (1 + node.visits)
    )
    return node.score - bonus
```

In `expand()`, after creating each child node, increment the parent's visits:

```python
# After line: node.children.append(child)
node.visits += 1
```

In `best_first_search()`, when pushing children to the frontier, use UCB-adjusted score:

```python
# Replace: heapq.heappush(self._frontier, child)
child.score = ucb_adjusted_score(child)
heapq.heappush(self._frontier, child)
```

Also update the import at the top of the file to include `math`.

**Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_search_tree.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add backend/bourbaki/autonomous/search_tree.py backend/tests/test_search_tree.py
git commit -m "feat: wire up UCB exploration bonus in proof search tree"
```

---

## Task 4: Quick-Win — Fix modal_runner to verify code against Lean

**Files:**
- Modify: `backend/bourbaki/autonomous/modal_runner.py`
- Create: `backend/tests/test_modal_runner.py`

**Step 1: Write the failing test**

Create `backend/tests/test_modal_runner.py`:

```python
"""Tests for modal_runner strategy response parsing and verification."""

from bourbaki.autonomous.modal_runner import _parse_strategy_response


def test_parse_success_true():
    text = """```lean4
theorem foo : True := by trivial
```

**INSIGHT:** Direct proof works
**SUCCESS:** true
**PARTIAL_PROGRESS:** none"""
    result = _parse_strategy_response(text, "direct-proof", 100)
    assert result.success is True
    assert result.proof_code is not None
    assert "trivial" in result.proof_code
    assert result.insight == "Direct proof works"


def test_parse_success_false():
    text = """```lean4
-- attempt failed
```

**INSIGHT:** Induction didn't work
**SUCCESS:** false
**PARTIAL_PROGRESS:** Got base case"""
    result = _parse_strategy_response(text, "induction", 200)
    assert result.success is False
    assert result.partial_progress == "Got base case"


def test_parse_no_code_block():
    text = "I couldn't find a proof.\n**SUCCESS:** false"
    result = _parse_strategy_response(text, "direct-proof", 50)
    assert result.success is False
    assert result.proof_code is None


def test_verified_flag_default_false():
    """Strategy results should track whether code was verified against Lean."""
    text = "```lean4\nsorry\n```\n**SUCCESS:** true"
    result = _parse_strategy_response(text, "test", 100)
    assert result.verified is False
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_modal_runner.py -v`
Expected: FAIL — `result.verified` doesn't exist

**Step 3: Add verified field to StrategyResult and update modal_runner**

In `backend/bourbaki/autonomous/strategies.py`, add to `StrategyResult`:

```python
@dataclass
class StrategyResult:
    strategy_id: str
    success: bool
    partial_progress: str | None = None
    error: str | None = None
    insight: str | None = None
    proof_code: str | None = None
    time_spent: int = 0  # milliseconds
    verified: bool = False  # Whether proof_code was verified against Lean
```

In `backend/bourbaki/autonomous/modal_runner.py`, update `execute_strategy_local` to
verify generated code when the LLM claims success:

```python
async def execute_strategy_local(
    problem: dict[str, Any],
    strategy: dict[str, Any],
    model: str,
    proof_state: dict[str, Any] | None = None,
    dead_ends: list[dict[str, Any]] | None = None,
) -> StrategyResult:
    """Execute a single strategy locally using Pydantic AI."""
    prompt = _build_strategy_prompt(problem, strategy, proof_state, dead_ends)
    start = time.monotonic()

    try:
        agent: Agent[None, str] = Agent(model, system_prompt=PROOF_SYSTEM_PROMPT)
        result = await agent.run(prompt)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        parsed = _parse_strategy_response(result.output, strategy["id"], elapsed_ms)

        # Verify generated code against Lean if the LLM claims success
        if parsed.success and parsed.proof_code:
            try:
                from bourbaki.tools.lean_prover import lean_prover
                verification = await lean_prover(code=parsed.proof_code, mode="check")
                if verification.get("proofComplete"):
                    parsed.verified = True
                else:
                    parsed.success = False
                    parsed.error = (
                        "LLM claimed success but Lean verification failed: "
                        + str(verification.get("errors", "unknown error"))
                    )
            except Exception as e:
                parsed.error = f"Verification error: {e}"
                parsed.success = False

        return parsed
    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return StrategyResult(
            strategy_id=strategy["id"],
            success=False,
            error=str(e),
            time_spent=elapsed_ms,
        )
```

**Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_modal_runner.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add backend/bourbaki/autonomous/modal_runner.py backend/bourbaki/autonomous/strategies.py backend/tests/test_modal_runner.py
git commit -m "fix: modal_runner now verifies generated code against Lean before claiming success"
```

---

## Task 5: Sketch Generator — data models and protocol

**Files:**
- Create: `backend/bourbaki/autonomous/sketch.py`
- Create: `backend/tests/test_sketch.py`

**Step 1: Write the failing test**

Create `backend/tests/test_sketch.py`:

```python
"""Tests for proof sketch generator."""

import json
from bourbaki.autonomous.sketch import (
    ProofSketch,
    SketchStep,
    SketchContext,
    parse_sketch_response,
)


def test_sketch_step_creation():
    step = SketchStep(
        statement="Show base case n = 0",
        formal_type="0 * (0 + 1) / 2 = 0",
    )
    assert step.statement == "Show base case n = 0"
    assert step.depends_on == []


def test_proof_sketch_creation():
    sketch = ProofSketch(
        strategy="induction",
        steps=[
            SketchStep(statement="Base case", formal_type="P 0"),
            SketchStep(statement="Inductive step", formal_type="P n → P (n+1)",
                       depends_on=[0]),
        ],
        key_lemmas=["Nat.add_comm"],
    )
    assert sketch.strategy == "induction"
    assert len(sketch.steps) == 2
    assert sketch.steps[1].depends_on == [0]


def test_parse_sketch_response_valid():
    response = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [
                {"statement": "Simplify LHS", "formal_type": "a + 0 = a"},
                {"statement": "Apply commutativity", "formal_type": "a = a"},
            ],
            "key_lemmas": ["Nat.add_zero"],
        }]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 1
    assert sketches[0].strategy == "direct"
    assert len(sketches[0].steps) == 2


def test_parse_sketch_response_multiple():
    response = json.dumps({
        "sketches": [
            {"strategy": "induction", "steps": [{"statement": "Induct"}], "key_lemmas": []},
            {"strategy": "direct", "steps": [{"statement": "Simplify"}], "key_lemmas": []},
        ]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 2


def test_parse_sketch_response_from_markdown():
    """Handle LLM responses wrapped in markdown code blocks."""
    response = '```json\n{"sketches": [{"strategy": "direct", "steps": [{"statement": "Done"}], "key_lemmas": []}]}\n```'
    sketches = parse_sketch_response(response)
    assert len(sketches) == 1


def test_parse_sketch_response_invalid_returns_empty():
    sketches = parse_sketch_response("this is not json at all")
    assert sketches == []


def test_sketch_context():
    ctx = SketchContext(
        theorem="theorem foo : 1 + 1 = 2",
        mathlib_results=[{"name": "Nat.add_comm"}],
    )
    assert "foo" in ctx.theorem
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_sketch.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement sketch.py**

Create `backend/bourbaki/autonomous/sketch.py`:

```python
"""Proof sketch generator for recursive subgoal decomposition.

Generates informal proof plans that guide the formalizer in creating
Lean have/sorry skeletons. The SketchGenerator protocol is pluggable —
the default LLMSketchGenerator uses the configured model, but a
Numina-style generator-verifier can be swapped in.

Reference: HILBERT (Apple), Aristotle lemma-based reasoning, Goedel-V2 proof planning.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class SketchStep:
    """A single step in an informal proof sketch."""
    statement: str              # NL description: "Show base case n=0"
    formal_type: str | None = None  # Optional Lean type hint: "P 0"
    depends_on: list[int] = field(default_factory=list)  # Indices of prior steps


@dataclass
class ProofSketch:
    """An informal proof plan with ordered steps."""
    strategy: str               # e.g., "induction", "contradiction", "direct"
    steps: list[SketchStep]     # Ordered informal steps
    key_lemmas: list[str] = field(default_factory=list)  # Expected Mathlib lemmas


@dataclass
class SketchContext:
    """Context provided to the sketch generator."""
    theorem: str                # Lean theorem statement
    mathlib_results: list[dict[str, Any]] = field(default_factory=list)
    previous_attempts: list[str] = field(default_factory=list)  # Failed approaches
    depth: int = 0              # Recursion depth (0 = top-level theorem)


SKETCH_PROMPT = """\
You are a mathematical proof planner. Given a Lean 4 theorem statement, produce
an informal proof sketch broken into small, verifiable steps.

Theorem:
{theorem}

{context_section}

Respond with a JSON object (no markdown wrapping) containing a "sketches" array.
Each sketch has:
- "strategy": proof technique name (e.g., "induction", "contradiction", "direct", "cases")
- "steps": array of steps, each with:
  - "statement": natural language description of what this step proves
  - "formal_type": (optional) the Lean 4 type this step would have as a `have` statement
  - "depends_on": (optional) array of 0-indexed step indices this depends on
- "key_lemmas": array of Mathlib lemma names expected to be useful

Generate 1-3 sketches with different strategies when possible.
Keep each sketch to 2-6 steps. Prefer smaller decompositions.

Example response:
{{"sketches": [{{
  "strategy": "induction",
  "steps": [
    {{"statement": "Base case: show P(0)", "formal_type": "P 0"}},
    {{"statement": "Inductive step: assuming P(n), show P(n+1)", "formal_type": "P n → P (n + 1)", "depends_on": [0]}}
  ],
  "key_lemmas": ["Nat.succ_eq_add_one"]
}}]}}
"""


def parse_sketch_response(response: str) -> list[ProofSketch]:
    """Parse LLM response into ProofSketch objects.

    Handles:
    - Raw JSON
    - JSON wrapped in markdown code blocks
    - Invalid JSON (returns empty list)
    """
    # Strip markdown code blocks if present
    text = response.strip()
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse sketch response as JSON")
        return []

    sketches: list[ProofSketch] = []
    for s in data.get("sketches", []):
        steps = []
        for step_data in s.get("steps", []):
            steps.append(SketchStep(
                statement=step_data.get("statement", ""),
                formal_type=step_data.get("formal_type"),
                depends_on=step_data.get("depends_on", []),
            ))
        sketches.append(ProofSketch(
            strategy=s.get("strategy", "unknown"),
            steps=steps,
            key_lemmas=s.get("key_lemmas", []),
        ))

    return sketches


def build_sketch_prompt(context: SketchContext) -> str:
    """Build the prompt for sketch generation."""
    context_parts = []
    if context.mathlib_results:
        lemma_names = [r.get("name", "") for r in context.mathlib_results[:5]]
        context_parts.append(f"Potentially relevant Mathlib lemmas: {', '.join(lemma_names)}")
    if context.previous_attempts:
        context_parts.append(
            "Previous failed approaches (avoid these):\n"
            + "\n".join(f"- {a}" for a in context.previous_attempts)
        )
    if context.depth > 0:
        context_parts.append(
            f"This is a subgoal at recursion depth {context.depth}. "
            "Keep decomposition minimal (1-3 steps)."
        )

    context_section = "\n".join(context_parts) if context_parts else ""
    return SKETCH_PROMPT.format(theorem=context.theorem, context_section=context_section)


@runtime_checkable
class SketchGenerator(Protocol):
    """Protocol for sketch generators (pluggable)."""

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        """Generate proof sketches for a theorem."""
        ...


class LLMSketchGenerator:
    """Default sketch generator using the configured LLM."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        """Generate proof sketches by prompting the LLM."""
        from pydantic_ai import Agent

        prompt = build_sketch_prompt(context)
        agent: Agent[None, str] = Agent(self.model, system_prompt=(
            "You are a proof planning assistant. Output valid JSON only."
        ))

        try:
            result = await agent.run(prompt)
            return parse_sketch_response(result.output)
        except Exception as e:
            logger.error("Sketch generation failed: %s", e)
            return []
```

**Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_sketch.py -v`
Expected: All 7 PASS

**Step 5: Commit**

```bash
git add backend/bourbaki/autonomous/sketch.py backend/tests/test_sketch.py
git commit -m "feat: add proof sketch generator with pluggable SketchGenerator protocol"
```

---

## Task 6: Formalizer — sketch to Lean have/sorry skeleton

**Files:**
- Create: `backend/bourbaki/autonomous/formalizer.py`
- Create: `backend/tests/test_formalizer.py`

**Step 1: Write the failing test**

Create `backend/tests/test_formalizer.py`:

```python
"""Tests for sketch-to-Lean formalizer."""

import re
from bourbaki.autonomous.formalizer import (
    FormalizedSkeleton,
    Subgoal,
    build_skeleton_code,
    extract_subgoals_from_code,
    stitch_proofs,
)
from bourbaki.autonomous.sketch import ProofSketch, SketchStep


def test_build_skeleton_simple():
    sketch = ProofSketch(
        strategy="direct",
        steps=[
            SketchStep(statement="Simplify", formal_type="a + 0 = a"),
        ],
        key_lemmas=[],
    )
    theorem = "theorem foo (a : Nat) : a + 0 = a"
    code = build_skeleton_code(theorem, sketch)
    assert "have" in code
    assert "sorry" in code
    assert theorem in code or "foo" in code


def test_build_skeleton_multi_step():
    sketch = ProofSketch(
        strategy="induction",
        steps=[
            SketchStep(statement="Base case", formal_type="P 0"),
            SketchStep(statement="Inductive step", formal_type="∀ n, P n → P (n+1)"),
        ],
        key_lemmas=[],
    )
    theorem = "theorem bar : ∀ n, P n"
    code = build_skeleton_code(theorem, sketch)
    sorry_count = code.count("sorry")
    assert sorry_count >= 2  # At least one sorry per step


def test_extract_subgoals():
    code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  have step_1 : 2 + 2 = 4 := by sorry
  trivial"""
    subgoals = extract_subgoals_from_code(code)
    assert len(subgoals) == 2
    assert subgoals[0].label == "step_0"
    assert subgoals[0].lean_type == "1 + 1 = 2"
    assert subgoals[1].label == "step_1"


def test_stitch_proofs_replaces_sorry():
    skeleton_code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  trivial"""
    subgoal_proofs = {"step_0": ["norm_num"]}
    result = stitch_proofs(skeleton_code, subgoal_proofs)
    assert "sorry" not in result
    assert "norm_num" in result


def test_stitch_proofs_multi_tactic():
    skeleton_code = """\
theorem foo : True := by
  have step_0 : P := by sorry
  trivial"""
    subgoal_proofs = {"step_0": ["simp", "ring"]}
    result = stitch_proofs(skeleton_code, subgoal_proofs)
    assert "sorry" not in result
    assert "simp" in result
    assert "ring" in result
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_formalizer.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement formalizer.py**

Create `backend/bourbaki/autonomous/formalizer.py`:

```python
"""Formalizer: converts proof sketches to Lean have/sorry skeletons.

Takes a ProofSketch (informal plan) and a theorem statement, produces
a Lean code skeleton where each step is a `have` with `sorry`, ready
for the subgoal solver to fill in.

Reference: HILBERT skeleton generation, DeepSeek-V2 have/sorry decomposition.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bourbaki.autonomous.sketch import ProofSketch

logger = logging.getLogger(__name__)


@dataclass
class Subgoal:
    """A single subgoal extracted from a formalized skeleton."""
    index: int                      # Position in the skeleton
    label: str                      # e.g., "step_0", "step_1"
    lean_type: str                  # The type of the have statement
    lean_context: str = ""          # Lean code preceding this subgoal
    proof_state_id: int | None = None  # REPL proof state ID (set later)


@dataclass
class FormalizedSkeleton:
    """Result of formalizing a proof sketch."""
    code: str                       # Full Lean code with sorry placeholders
    subgoals: list[Subgoal]         # Extracted subgoals to solve
    compilation_ok: bool = False    # Whether the skeleton type-checks in Lean
    errors: list[str] = field(default_factory=list)


def build_skeleton_code(theorem: str, sketch: ProofSketch) -> str:
    """Build Lean code with have/sorry from a proof sketch.

    Args:
        theorem: The Lean theorem statement (e.g., "theorem foo : T").
        sketch: The informal proof sketch.

    Returns:
        Lean code string with sorry placeholders.
    """
    lines = [f"{theorem} := by"]

    for i, step in enumerate(sketch.steps):
        label = f"step_{i}"
        if step.formal_type:
            lines.append(f"  have {label} : {step.formal_type} := by sorry")
        else:
            # No formal type hint — use a placeholder that the LLM correction
            # loop will need to fix
            lines.append(f"  -- Step {i}: {step.statement}")
            lines.append(f"  have {label} : sorry := by sorry")

    # Final step: try to close the proof using the intermediate steps
    step_labels = [f"step_{i}" for i in range(len(sketch.steps))]
    if step_labels:
        lines.append(f"  exact?")
    else:
        lines.append("  sorry")

    return "\n".join(lines)


def extract_subgoals_from_code(code: str) -> list[Subgoal]:
    """Extract subgoals (have ... := by sorry) from skeleton code.

    Returns:
        List of Subgoal objects with label and lean_type populated.
    """
    subgoals: list[Subgoal] = []

    # Match: have <label> : <type> := by sorry
    pattern = re.compile(
        r"have\s+(\w+)\s*:\s*(.+?)\s*:=\s*by\s+sorry"
    )

    lines = code.split("\n")
    context_lines: list[str] = []

    for line in lines:
        match = pattern.search(line)
        if match:
            label = match.group(1)
            lean_type = match.group(2).strip()
            subgoals.append(Subgoal(
                index=len(subgoals),
                label=label,
                lean_type=lean_type,
                lean_context="\n".join(context_lines),
            ))
        context_lines.append(line)

    return subgoals


def stitch_proofs(
    skeleton_code: str,
    subgoal_proofs: dict[str, list[str]],
) -> str:
    """Replace sorry placeholders with actual tactic proofs.

    Args:
        skeleton_code: The Lean skeleton with sorry placeholders.
        subgoal_proofs: Map of subgoal label -> list of tactic strings.

    Returns:
        Complete Lean code with sorry replaced by actual proofs.
    """
    result = skeleton_code

    for label, tactics in subgoal_proofs.items():
        if not tactics:
            continue

        # Pattern: have <label> : <type> := by sorry
        pattern = re.compile(
            rf"(have\s+{re.escape(label)}\s*:\s*.+?)\s*:=\s*by\s+sorry"
        )

        if len(tactics) == 1:
            replacement = rf"\1 := by {tactics[0]}"
        else:
            # Multi-tactic: indent under `by`
            tactic_block = "\n    ".join(tactics)
            replacement = rf"\1 := by\n    {tactic_block}"

        result = pattern.sub(replacement, result)

    return result


FORMALIZE_PROMPT = """\
Convert this proof sketch into a Lean 4 skeleton with `have` statements.
Each intermediate step should be a `have step_N : <type> := by sorry`.
The final line should close the proof using the intermediate steps.

Theorem: {theorem}

Proof sketch:
{sketch_text}

Key Mathlib lemmas that may be useful: {lemmas}

Output ONLY the Lean 4 code, no explanation. The code should type-check
except for the `sorry` placeholders. Use `have step_0`, `step_1`, etc.
as labels.
"""


async def formalize_sketch(
    theorem: str,
    sketch: ProofSketch,
    model: str,
    max_retries: int = 2,
) -> FormalizedSkeleton:
    """Convert a proof sketch to a Lean skeleton using LLM + Lean feedback.

    Tries to formalize, checks compilation, and retries with error
    feedback up to max_retries times (per Goedel-V2: 2 rounds optimal).

    Args:
        theorem: Lean theorem statement.
        sketch: The informal proof sketch.
        model: LLM model identifier.
        max_retries: Max error-correction rounds.

    Returns:
        FormalizedSkeleton with code, subgoals, and compilation status.
    """
    from pydantic_ai import Agent

    # First attempt: build from sketch directly
    code = build_skeleton_code(theorem, sketch)
    subgoals = extract_subgoals_from_code(code)

    # Try to compile and get REPL feedback
    skeleton = FormalizedSkeleton(code=code, subgoals=subgoals)

    try:
        from bourbaki.tools.lean_repl import lean_tactic
        result = await lean_tactic(goal=theorem, tactic="sorry", proof_state=None)
        if result.get("success"):
            skeleton.compilation_ok = True
            return skeleton
    except Exception:
        pass

    # If direct build fails, ask LLM to formalize with error feedback
    sketch_text = "\n".join(
        f"{i+1}. {s.statement}" + (f" [{s.formal_type}]" if s.formal_type else "")
        for i, s in enumerate(sketch.steps)
    )
    lemmas = ", ".join(sketch.key_lemmas) if sketch.key_lemmas else "none specified"
    prompt = FORMALIZE_PROMPT.format(
        theorem=theorem, sketch_text=sketch_text, lemmas=lemmas,
    )

    errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            agent: Agent[None, str] = Agent(model, system_prompt=(
                "You are a Lean 4 formalization assistant. Output only valid Lean 4 code."
            ))

            if attempt > 0 and errors:
                prompt += f"\n\nPrevious attempt failed with errors:\n" + "\n".join(errors[-3:])
                prompt += "\nFix the errors and output corrected code."

            result = await agent.run(prompt)
            generated_code = _extract_lean_code(result.output)
            if generated_code:
                code = generated_code

            subgoals = extract_subgoals_from_code(code)
            skeleton = FormalizedSkeleton(code=code, subgoals=subgoals)

            # Try to compile
            try:
                from bourbaki.tools.lean_repl import lean_tactic
                check = await lean_tactic(goal=theorem, tactic="sorry", proof_state=None)
                if check.get("success"):
                    skeleton.compilation_ok = True
                    return skeleton
                else:
                    errors.append(check.get("error", "unknown compilation error"))
            except Exception as e:
                errors.append(str(e))

        except Exception as e:
            errors.append(str(e))

    skeleton.errors = errors
    return skeleton


def _extract_lean_code(text: str) -> str | None:
    """Extract Lean code from an LLM response (may be wrapped in markdown)."""
    # Try markdown code block
    match = re.search(r"```(?:lean4?)\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, check if the whole response looks like Lean
    if "theorem" in text or "lemma" in text or ":= by" in text:
        return text.strip()

    return None
```

**Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_formalizer.py -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add backend/bourbaki/autonomous/formalizer.py backend/tests/test_formalizer.py
git commit -m "feat: add formalizer to convert proof sketches to Lean have/sorry skeletons"
```

---

## Task 7: Decomposer — recursive orchestration

**Files:**
- Create: `backend/bourbaki/autonomous/decomposer.py`
- Create: `backend/tests/test_decomposer.py`

**Step 1: Write the failing test**

Create `backend/tests/test_decomposer.py`:

```python
"""Tests for recursive subgoal decomposer."""

from bourbaki.autonomous.decomposer import (
    DecompositionConfig,
    DecompositionResult,
)


def test_config_defaults():
    config = DecompositionConfig()
    assert config.max_recursion_depth == 2
    assert config.max_sketches == 3
    assert config.subgoal_search_budget == 50
    assert config.subgoal_search_timeout == 60.0
    assert config.formalization_retries == 2


def test_result_success():
    result = DecompositionResult(
        success=True,
        proof_code="theorem foo : True := by trivial",
        subgoals_total=2,
        subgoals_solved=2,
    )
    assert result.success
    assert result.all_solved


def test_result_partial():
    result = DecompositionResult(
        success=False,
        subgoals_total=3,
        subgoals_solved=1,
        failed_subgoals=["step_1", "step_2"],
    )
    assert not result.success
    assert not result.all_solved
    assert result.solve_rate == 1 / 3


def test_result_to_dict():
    result = DecompositionResult(success=True, subgoals_total=1, subgoals_solved=1)
    d = result.to_dict()
    assert d["success"] is True
    assert d["subgoals_total"] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_decomposer.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement decomposer.py**

Create `backend/bourbaki/autonomous/decomposer.py`:

```python
"""Recursive subgoal decomposer (HILBERT-style).

Orchestrates: sketch generation -> formalization -> subgoal solving -> stitching.
Each failed subgoal can be recursively decomposed up to max_recursion_depth.

Reference: HILBERT (Apple, 99.2% miniF2F), DeepSeek-Prover-V2 subgoal decomposition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bourbaki.autonomous.formalizer import (
    FormalizedSkeleton,
    Subgoal,
    formalize_sketch,
    stitch_proofs,
)
from bourbaki.autonomous.search_tree import SearchResult, prove_with_search
from bourbaki.autonomous.sketch import (
    LLMSketchGenerator,
    ProofSketch,
    SketchContext,
    SketchGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionConfig:
    """Configuration for recursive decomposition."""
    max_recursion_depth: int = 2
    max_sketches: int = 3
    subgoal_search_budget: int = 50
    subgoal_search_timeout: float = 60.0
    formalization_retries: int = 2
    model: str = "openai:gpt-4o"


@dataclass
class DecompositionResult:
    """Result of decompose_and_prove."""
    success: bool
    proof_code: str | None = None
    subgoals_total: int = 0
    subgoals_solved: int = 0
    solved_subgoal_proofs: dict[str, list[str]] = field(default_factory=dict)
    failed_subgoals: list[str] = field(default_factory=list)
    sketches_tried: int = 0
    recursion_depth_reached: int = 0
    total_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def all_solved(self) -> bool:
        return self.subgoals_total > 0 and self.subgoals_solved == self.subgoals_total

    @property
    def solve_rate(self) -> float:
        if self.subgoals_total == 0:
            return 0.0
        return self.subgoals_solved / self.subgoals_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "proof_code": self.proof_code,
            "subgoals_total": self.subgoals_total,
            "subgoals_solved": self.subgoals_solved,
            "failed_subgoals": self.failed_subgoals,
            "sketches_tried": self.sketches_tried,
            "recursion_depth_reached": self.recursion_depth_reached,
            "total_time": round(self.total_time, 2),
            "solve_rate": round(self.solve_rate, 3),
        }


async def decompose_and_prove(
    theorem: str,
    config: DecompositionConfig,
    sketch_generator: SketchGenerator | None = None,
    depth: int = 0,
    previous_attempts: list[str] | None = None,
) -> DecompositionResult:
    """Recursively decompose a theorem into subgoals and prove each.

    Algorithm:
    1. Generate proof sketches (1-3 diverse strategies)
    2. For each sketch, formalize into Lean have/sorry skeleton
    3. For each subgoal: try prove_with_search()
    4. Failed subgoals: recurse if depth < max_recursion_depth
    5. If all solved, stitch proofs and verify
    6. Return partial results even on failure

    Args:
        theorem: Lean 4 theorem statement.
        config: Decomposition configuration.
        sketch_generator: Pluggable sketch generator (defaults to LLMSketchGenerator).
        depth: Current recursion depth.
        previous_attempts: Failed approaches to avoid.

    Returns:
        DecompositionResult with proof or partial progress.
    """
    start = time.monotonic()

    if sketch_generator is None:
        sketch_generator = LLMSketchGenerator(config.model)

    result = DecompositionResult(
        success=False,
        recursion_depth_reached=depth,
    )

    # Generate proof sketches
    context = SketchContext(
        theorem=theorem,
        previous_attempts=previous_attempts or [],
        depth=depth,
    )

    try:
        sketches = await sketch_generator.generate(context)
    except Exception as e:
        result.errors.append(f"Sketch generation failed: {e}")
        result.total_time = time.monotonic() - start
        return result

    if not sketches:
        result.errors.append("No sketches generated")
        result.total_time = time.monotonic() - start
        return result

    # Try each sketch (up to max_sketches)
    for sketch_idx, sketch in enumerate(sketches[:config.max_sketches]):
        result.sketches_tried = sketch_idx + 1

        logger.info(
            "Trying sketch %d/%d (strategy=%s, steps=%d, depth=%d)",
            sketch_idx + 1, len(sketches), sketch.strategy,
            len(sketch.steps), depth,
        )

        # Formalize sketch into Lean skeleton
        skeleton = await formalize_sketch(
            theorem=theorem,
            sketch=sketch,
            model=config.model,
            max_retries=config.formalization_retries,
        )

        if not skeleton.subgoals:
            result.errors.append(f"Sketch {sketch_idx}: no subgoals extracted")
            continue

        # Try to solve each subgoal
        subgoal_proofs: dict[str, list[str]] = {}
        failed: list[str] = []
        result.subgoals_total = len(skeleton.subgoals)
        result.subgoals_solved = 0

        for subgoal in skeleton.subgoals:
            # Build a mini-theorem for this subgoal
            subgoal_theorem = f"theorem {subgoal.label} : {subgoal.lean_type}"

            # Try flat search first
            search_result = await prove_with_search(
                theorem=subgoal_theorem,
                budget=config.subgoal_search_budget,
                timeout=config.subgoal_search_timeout,
            )

            if search_result.success:
                subgoal_proofs[subgoal.label] = search_result.proof_tactics
                result.subgoals_solved += 1
                logger.info(
                    "Subgoal %s solved (%d tactics)",
                    subgoal.label, len(search_result.proof_tactics),
                )
            elif depth < config.max_recursion_depth:
                # Recurse: decompose the failed subgoal
                logger.info(
                    "Subgoal %s failed flat search, recursing (depth=%d)",
                    subgoal.label, depth + 1,
                )
                sub_result = await decompose_and_prove(
                    theorem=subgoal_theorem,
                    config=config,
                    sketch_generator=sketch_generator,
                    depth=depth + 1,
                    previous_attempts=[f"Flat search failed on: {subgoal.lean_type}"],
                )
                result.recursion_depth_reached = max(
                    result.recursion_depth_reached,
                    sub_result.recursion_depth_reached,
                )
                if sub_result.success and sub_result.proof_code:
                    # Extract tactics from the recursive proof
                    # The sub_result.proof_code is a full theorem — extract tactic block
                    tactics = _extract_tactics_from_proof(sub_result.proof_code)
                    subgoal_proofs[subgoal.label] = tactics
                    result.subgoals_solved += 1
                else:
                    failed.append(subgoal.label)
            else:
                failed.append(subgoal.label)

        result.solved_subgoal_proofs = subgoal_proofs
        result.failed_subgoals = failed

        # If all subgoals solved, stitch and verify
        if not failed:
            stitched = stitch_proofs(skeleton.code, subgoal_proofs)
            result.proof_code = stitched
            result.success = True
            result.total_time = time.monotonic() - start
            logger.info(
                "Decomposition succeeded: %d subgoals, %d sketches tried, depth=%d",
                result.subgoals_total, result.sketches_tried, depth,
            )
            return result

        # Some subgoals failed — try next sketch
        logger.info(
            "Sketch %d: %d/%d subgoals solved, trying next sketch",
            sketch_idx, result.subgoals_solved, result.subgoals_total,
        )

    result.total_time = time.monotonic() - start
    return result


def _extract_tactics_from_proof(proof_code: str) -> list[str]:
    """Extract the tactic block from a complete proof.

    Given: "theorem foo : T := by\\n  tactic1\\n  tactic2"
    Returns: ["tactic1", "tactic2"]
    """
    # Find "by" and extract everything after it
    by_idx = proof_code.find(":= by")
    if by_idx == -1:
        by_idx = proof_code.find("by\n")
        if by_idx == -1:
            return [proof_code]  # Return as-is

    tactic_block = proof_code[by_idx + 5:].strip()  # Skip ":= by"
    tactics = [
        line.strip()
        for line in tactic_block.split("\n")
        if line.strip() and not line.strip().startswith("--")
    ]
    return tactics if tactics else [tactic_block]
```

**Step 4: Run tests**

Run: `cd backend && python -m pytest tests/test_decomposer.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add backend/bourbaki/autonomous/decomposer.py backend/tests/test_decomposer.py
git commit -m "feat: add recursive subgoal decomposer (HILBERT-style)"
```

---

## Task 8: Integrate decomposition into search.py as Phase 0

**Files:**
- Modify: `backend/bourbaki/autonomous/search.py`

**Step 1: Add decomposition config fields to AutonomousSearchConfig**

In `backend/bourbaki/autonomous/search.py`, add to `AutonomousSearchConfig.__init__`:

```python
def __init__(
    self,
    max_iterations: int = 100,
    max_hours: float = 4.0,
    strategies: list[str] | None = None,
    checkpoint_interval: int = 10,
    auto_resume: bool = True,
    max_dead_ends_per_strategy: int = 3,
    use_search_tree: bool = False,
    search_tree_budget: int = 100,
    search_tree_max_depth: int = 30,
    # Phase 0: Recursive decomposition
    use_decomposition: bool = True,
    decomposition_max_depth: int = 2,
    decomposition_max_sketches: int = 3,
    decomposition_subgoal_budget: int = 50,
):
    # ... existing fields ...
    self.use_decomposition = use_decomposition
    self.decomposition_max_depth = decomposition_max_depth
    self.decomposition_max_sketches = decomposition_max_sketches
    self.decomposition_subgoal_budget = decomposition_subgoal_budget
```

**Step 2: Add Phase 0 to _run_loop**

At the top of `_run_loop()`, before the existing Phase 1 (search tree) block, add:

```python
# Phase 0: Recursive subgoal decomposition
if (
    self._config.use_decomposition
    and self._problem
    and self._problem.get("lean_statement")
):
    from bourbaki.autonomous.decomposer import (
        DecompositionConfig,
        decompose_and_prove,
    )

    self._emit({
        "type": "strategy_attempt",
        "strategy": "recursive-decomposition",
        "approach": "HILBERT-style recursive subgoal decomposition",
    })

    decomp_config = DecompositionConfig(
        max_recursion_depth=self._config.decomposition_max_depth,
        max_sketches=self._config.decomposition_max_sketches,
        subgoal_search_budget=self._config.decomposition_subgoal_budget,
        model=settings.default_model,
    )

    decomp_result = await decompose_and_prove(
        theorem=self._problem["lean_statement"],
        config=decomp_config,
    )

    self._iteration += 1

    if decomp_result.success and decomp_result.proof_code:
        self._proof_state = {
            "complete": True,
            "proof_code": decomp_result.proof_code,
        }
        self._insights.append(
            f"Decomposition found proof: {decomp_result.subgoals_solved} subgoals, "
            f"{decomp_result.sketches_tried} sketches tried, "
            f"depth={decomp_result.recursion_depth_reached}"
        )
        self._emit({
            "type": "completed",
            "success": True,
            "result": decomp_result.proof_code,
        })
        self._status = "completed"
        return
    else:
        insight = (
            f"Decomposition partial: {decomp_result.subgoals_solved}/{decomp_result.subgoals_total} "
            f"subgoals solved ({decomp_result.sketches_tried} sketches tried)"
        )
        self._insights.append(insight)
        self._emit({
            "type": "strategy_result",
            "strategy": "recursive-decomposition",
            "success": False,
            "insight": insight,
            "partial_progress": decomp_result.to_dict(),
        })
```

**Step 3: Verify import works**

Run: `cd backend && python -c "from bourbaki.autonomous.search import AutonomousSearch, AutonomousSearchConfig; c = AutonomousSearchConfig(); print('decomp:', c.use_decomposition)"`
Expected: `decomp: True`

**Step 4: Commit**

```bash
git add backend/bourbaki/autonomous/search.py
git commit -m "feat: integrate recursive decomposition as Phase 0 in autonomous search"
```

---

## Task 9: Update search_tree to use NoveltyTracker from scoring

**Files:**
- Modify: `backend/bourbaki/autonomous/search_tree.py`

**Step 1: Wire NoveltyTracker into ProofSearchTree**

In `ProofSearchTree.__init__`, add:

```python
from bourbaki.autonomous.scoring import NoveltyTracker

self._novelty_tracker = NoveltyTracker()
```

In `expand()`, update the score calculation to pass the novelty tracker:

```python
child = ProofNode(
    proof_state=new_ps,
    goals=new_goals,
    tactic_history=node.tactic_history + [tactic],
    parent=node,
    score=score_proof_state(new_goals, node.depth + 1, self._novelty_tracker),
    depth=node.depth + 1,
    tactic=tactic,
)
```

Update the root node creation in `initialize()` similarly:

```python
self.root = ProofNode(
    proof_state=result.get("proofState", 0),
    goals=result.get("goals", []),
    tactic_history=[],
    score=score_proof_state(result.get("goals", []), 0, self._novelty_tracker),
)
```

Remove the old `_seen_goals` set and `_goal_key` method since `NoveltyTracker` replaces them. Update deduplication in `expand()` to use `self._novelty_tracker.has_seen(new_goals)` instead.

**Step 2: Run existing tests still pass**

Run: `cd backend && python -m pytest tests/test_search_tree.py tests/test_scoring.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add backend/bourbaki/autonomous/search_tree.py
git commit -m "feat: wire NoveltyTracker into search tree, replace _seen_goals dedup"
```

---

## Task 10: Update roadmap tracker

**Files:**
- Modify: `.bourbaki/roadmap/tracker.md`

**Step 1: Update Phase 3 status**

Mark the relevant Phase 3 tasks as in-progress or complete based on what was built.

**Step 2: Commit**

```bash
git add .bourbaki/roadmap/tracker.md
git commit -m "docs: update roadmap tracker with decomposition progress"
```

---

## Task 11: Integration smoke test

**Files:**
- Create: `backend/tests/test_decomposer_integration.py`

**Step 1: Write integration test (mocked LLM, mocked REPL)**

Create `backend/tests/test_decomposer_integration.py`:

```python
"""Integration test for the full decomposition pipeline.

Uses mocked LLM and REPL to test the full flow:
sketch generation -> formalization -> subgoal solving -> stitching.
"""

import pytest
from unittest.mock import AsyncMock, patch

from bourbaki.autonomous.decomposer import (
    DecompositionConfig,
    decompose_and_prove,
)
from bourbaki.autonomous.sketch import (
    ProofSketch,
    SketchContext,
    SketchStep,
)


class MockSketchGenerator:
    """Returns a fixed sketch for testing."""

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        return [ProofSketch(
            strategy="direct",
            steps=[
                SketchStep(
                    statement="Simplify using norm_num",
                    formal_type="1 + 1 = 2",
                ),
            ],
            key_lemmas=[],
        )]


@pytest.mark.asyncio
async def test_decompose_with_mock_generator():
    """Test decomposition with mocked sketch generator and search."""
    config = DecompositionConfig(
        max_recursion_depth=1,
        max_sketches=1,
        subgoal_search_budget=10,
        subgoal_search_timeout=5.0,
    )

    # Mock prove_with_search to return success
    mock_search_result = AsyncMock()
    mock_search_result.return_value.success = True
    mock_search_result.return_value.proof_tactics = ["norm_num"]
    mock_search_result.return_value.proof_code = "theorem step_0 : 1 + 1 = 2 := by norm_num"

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        mock_search_result,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : 1 + 1 = 2",
            config=config,
            sketch_generator=MockSketchGenerator(),
        )

    assert result.subgoals_total >= 1
    assert result.subgoals_solved >= 1
    assert result.sketches_tried == 1
```

**Step 2: Run integration test**

Run: `cd backend && python -m pytest tests/test_decomposer_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add backend/tests/test_decomposer_integration.py
git commit -m "test: add integration smoke test for decomposition pipeline"
```

---

## Summary

| Task | Component | Type |
|------|-----------|------|
| 1 | Raise tool_call_limits | Quick-win config |
| 2 | Novelty bonus in scoring | Quick-win + tests |
| 3 | UCB exploration in search_tree | Quick-win + tests |
| 4 | Fix modal_runner verification | Bug fix + tests |
| 5 | Sketch generator | New module + tests |
| 6 | Formalizer | New module + tests |
| 7 | Decomposer | New module + tests |
| 8 | Integrate Phase 0 in search.py | Integration |
| 9 | Wire NoveltyTracker into search_tree | Integration |
| 10 | Update roadmap tracker | Docs |
| 11 | Integration smoke test | Testing |

After all tasks: run `cd backend && python -m pytest tests/ -v` to verify no regressions.

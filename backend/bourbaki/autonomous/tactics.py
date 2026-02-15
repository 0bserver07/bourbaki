"""Tactic candidate generation for proof search.

Generates ranked lists of candidate tactics to try at each proof state,
based on goal structure and available Mathlib lemmas.
"""

from __future__ import annotations

import re
from typing import Any


# Standard automation tactics — always worth trying
AUTOMATION_TACTICS = [
    "simp",
    "ring",
    "omega",
    "norm_num",
    "linarith",
    "decide",
    "aesop",
    "trivial",
    "assumption",
    "contradiction",
    "rfl",
]

# Goal pattern → candidate tactics
# Patterns are checked against the goal string
_GOAL_TACTIC_MAP: list[tuple[str, list[str]]] = [
    # Quantifiers
    (r"∀", ["intro", "intros"]),
    (r"∃", ["use", "refine ⟨?_, ?_⟩", "exact ⟨_, _⟩"]),
    # Connectives
    (r"∧", ["constructor", "exact ⟨_, _⟩", "And.intro"]),
    (r"∨", ["left", "right", "Or.inl", "Or.inr"]),
    (r"↔", ["constructor", "Iff.intro"]),
    (r"¬", ["intro", "push_neg"]),
    (r"False", ["contradiction", "exact absurd", "exfalso"]),
    (r"True", ["trivial", "exact True.intro"]),
    # Equality
    (r"=", ["rfl", "ring", "simp", "omega", "norm_num", "ext", "funext", "congr"]),
    # Inequality / ordering
    (r"[<≤>≥]", ["linarith", "omega", "norm_num", "nlinarith", "positivity", "gcongr"]),
    (r"≠", ["intro", "push_neg", "omega", "norm_num"]),
    # Membership / set operations
    (r"∈", ["simp", "exact mem_of", "apply mem_of"]),
    (r"⊆", ["intro", "simp"]),
    # Natural numbers
    (r"Nat\.", ["omega", "simp [Nat.]", "norm_num", "induction"]),
    (r"Fin\.", ["simp", "omega", "fin_cases"]),
    # Division / modular arithmetic
    (r"[%∣]", ["omega", "norm_num", "simp [Nat.dvd_iff_mod_eq_zero]"]),
    (r"Nat\.Prime", ["norm_num [Nat.Prime]", "decide"]),
    # Algebraic structures
    (r"Group\.|Ring\.|Field\.", ["group", "ring", "field_simp"]),
    # Summation / product
    (r"Finset\.sum|∑", ["simp [Finset.sum]", "ring", "norm_num"]),
    (r"Finset\.prod|∏", ["simp [Finset.prod]", "ring"]),
    # Function application
    (r"fun\s", ["ext", "funext", "simp"]),
    # Lists / arrays
    (r"List\.", ["simp [List.]", "induction", "cases"]),
]

# Compiled patterns
_COMPILED_PATTERNS = [(re.compile(p), tactics) for p, tactics in _GOAL_TACTIC_MAP]


def generate_candidates(
    goals: list[str],
    depth: int = 0,
    mathlib_results: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Generate ranked candidate tactics for the current proof state.

    Args:
        goals: Remaining goal strings from Lean.
        depth: Current depth in search tree (affects tactic ordering).
        mathlib_results: Results from mathlib_search for the current goal.

    Returns:
        Ordered list of tactic strings to try.
    """
    if not goals:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(tactic: str) -> None:
        if tactic not in seen:
            seen.add(tactic)
            candidates.append(tactic)

    primary_goal = goals[0]

    # 1. Goal-aware tactics (highest priority — targeted at the goal structure)
    for pattern, tactics in _COMPILED_PATTERNS:
        if pattern.search(primary_goal):
            for t in tactics:
                _add(t)

    # 2. Mathlib lemma application (if we have search results)
    if mathlib_results:
        for result in mathlib_results[:5]:
            name = result.get("name", "")
            if name:
                _add(f"exact {name}")
                _add(f"apply {name}")

    # 3. Standard automation (always worth trying)
    for t in AUTOMATION_TACTICS:
        _add(t)

    # 4. Structural tactics for deeper exploration
    if depth < 3:
        # At shallow depth, try more exploratory tactics
        _add("simp_all")
        _add("push_neg")
        _add("contrapose")
        _add("by_contra")

    # 5. Induction variants (if goal mentions natural numbers or lists)
    if re.search(r"(?:Nat|ℕ|List|Fin)", primary_goal):
        # Try to extract the variable name for induction
        var_match = re.search(r"([a-z_]\w*)\s*:\s*(?:Nat|ℕ)", primary_goal)
        if var_match:
            var = var_match.group(1)
            _add(f"induction {var}")
            _add(f"cases {var}")
        else:
            _add("induction n")
            _add("cases n")

    return candidates


def generate_mathlib_queries(goals: list[str]) -> list[tuple[str, str]]:
    """Generate mathlib_search queries based on goal structure.

    Returns list of (query, mode) tuples for mathlib_search.
    """
    if not goals:
        return []

    queries: list[tuple[str, str]] = []
    primary_goal = goals[0]

    # Extract the goal type (everything after ⊢)
    goal_type_match = re.search(r"⊢\s*(.+)", primary_goal)
    goal_type = goal_type_match.group(1).strip() if goal_type_match else primary_goal

    # Type signature search — most precise
    queries.append((goal_type, "type"))

    # Natural language search — broader
    # Simplify the goal for natural language
    nl_query = goal_type
    nl_query = re.sub(r"[∀∃].*?,\s*", "", nl_query)  # Strip quantifiers
    nl_query = nl_query.replace("→", "implies").replace("↔", "iff")
    nl_query = nl_query.replace("∧", "and").replace("∨", "or")
    if len(nl_query) > 10:
        queries.append((nl_query[:100], "natural"))

    return queries

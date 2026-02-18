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
    "nlinarith",
    "decide",
    "aesop",
    "trivial",
    "assumption",
    "contradiction",
    "rfl",
    "positivity",
    "norm_cast",
    "field_simp",
    "push_cast",
    "ring_nf",
    "simp_all",
]

# Goal pattern → candidate tactics
# Patterns are checked against the goal string
_GOAL_TACTIC_MAP: list[tuple[str, list[str]]] = [
    # Quantifiers
    (r"∀", ["intro", "intros"]),
    (r"∃", ["use", "refine ⟨?_, ?_⟩", "exact ⟨_, _⟩"]),
    # Connectives
    (r"∧", ["constructor", "exact ⟨_, _⟩", "refine ⟨?_, ?_⟩"]),
    (r"∨", ["left", "right", "Or.inl", "Or.inr"]),
    (r"↔", ["constructor", "Iff.intro"]),
    (r"¬", ["intro", "push_neg"]),
    (r"False", ["contradiction", "exact absurd", "exfalso"]),
    (r"True", ["trivial", "exact True.intro"]),
    # Equality
    (r"=", ["rfl", "ring", "simp", "omega", "norm_num", "ext", "funext",
            "congr", "norm_cast", "push_cast", "ring_nf"]),
    # Inequality / ordering
    (r"[<≤>≥]", ["linarith", "omega", "norm_num", "nlinarith", "positivity",
                  "gcongr", "norm_cast"]),
    (r"≠", ["intro", "push_neg", "omega", "norm_num"]),
    # Membership / set operations
    (r"∈", ["simp", "exact mem_of", "apply mem_of"]),
    (r"⊆", ["intro", "simp"]),
    # Natural numbers
    (r"Nat\.", ["omega", "simp [Nat.]", "norm_num", "induction", "norm_cast"]),
    (r"Fin\.", ["simp", "omega", "fin_cases"]),
    # Division / modular arithmetic
    (r"[%∣]", ["omega", "norm_num", "simp [Nat.dvd_iff_mod_eq_zero]",
               "decide", "norm_num [Nat.Prime]"]),
    (r"Nat\.Prime", ["norm_num [Nat.Prime]", "decide"]),
    # Type casts / coercions
    (r"[↑↓]|Int\.ofNat|Nat\.cast|Int\.toNat", ["norm_cast", "push_cast", "simp"]),
    (r"(?:ℤ|Int)\.", ["omega", "norm_num", "push_cast", "norm_cast"]),
    (r"(?:ℚ|Rat)\.", ["field_simp", "ring", "norm_num", "push_cast"]),
    (r"(?:ℝ|Real)\.", ["field_simp", "ring", "norm_num", "nlinarith", "positivity"]),
    # Powers / exponentials
    (r"\^", ["ring", "norm_num", "nlinarith", "simp [pow_succ]",
             "simp [Nat.pow_succ]"]),
    # Absolute value
    (r"abs|‖", ["simp [abs_le]", "norm_num", "nlinarith"]),
    # Algebraic structures
    (r"Group\.|Ring\.|Field\.", ["group", "ring", "field_simp"]),
    # Summation / product
    (r"Finset\.sum|∑", ["simp [Finset.sum]", "ring", "norm_num"]),
    (r"Finset\.prod|∏", ["simp [Finset.prod]", "ring"]),
    # Function application
    (r"fun\s", ["ext", "funext", "simp"]),
    # Lists / arrays
    (r"List\.", ["simp [List.]", "induction", "cases"]),
    # GCD / LCM
    (r"gcd|lcm", ["omega", "norm_num", "simp [Nat.gcd]", "decide"]),
    # Modular arithmetic
    (r"Zmod|ZMod", ["simp", "decide", "norm_num"]),
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
        _add("norm_cast")
        _add("push_cast")
        _add("field_simp")

    if depth < 5:
        # Slightly deeper — try rewriting and simplification combos
        _add("simp only [not_lt, not_le]")
        _add("ring_nf")

    # 5. Induction variants (if goal mentions natural numbers or lists)
    if re.search(r"(?:Nat|ℕ|List|Fin)", primary_goal):
        # Try to extract the variable name for induction
        var_match = re.search(r"([a-z_]\w*)\s*:\s*(?:Nat|ℕ)", primary_goal)
        if var_match:
            var = var_match.group(1)
            _add(f"induction {var}")
            _add(f"induction {var} with\n| zero => simp\n| succ n ih => simp_all")
            _add(f"cases {var}")
            _add(f"cases {var} with\n| zero => simp\n| succ n => simp_all")
        else:
            _add("induction n")
            _add("cases n")

    # 6. Hypothesis manipulation (if hypotheses exist in goal)
    hyp_match = re.findall(r"([a-z_]\w*)\s*:", primary_goal)
    if hyp_match and depth < 4:
        for h in hyp_match[:3]:  # Limit to avoid combinatorial explosion
            _add(f"simp at {h}")
            _add(f"rw [{h}]")

    return candidates


def generate_mathlib_queries(goals: list[str]) -> list[tuple[str, str]]:
    """Generate mathlib_search queries based on goal structure.

    Returns list of (query, mode) tuples for mathlib_search.
    Includes semantic, type, and natural language queries.
    """
    if not goals:
        return []

    queries: list[tuple[str, str]] = []
    primary_goal = goals[0]

    # Extract the goal type (everything after ⊢)
    goal_type_match = re.search(r"⊢\s*(.+)", primary_goal)
    goal_type = goal_type_match.group(1).strip() if goal_type_match else primary_goal

    # Semantic search — best for goal-state-aware retrieval (hybrid ranking)
    queries.append((goal_type, "semantic"))

    # Type signature search — most precise for exact type matching
    queries.append((goal_type, "type"))

    # Natural language search — broader, catches differently-named lemmas
    nl_query = goal_type
    nl_query = re.sub(r"[∀∃].*?,\s*", "", nl_query)  # Strip quantifiers
    nl_query = nl_query.replace("→", "implies").replace("↔", "iff")
    nl_query = nl_query.replace("∧", "and").replace("∨", "or")
    if len(nl_query) > 10:
        queries.append((nl_query[:100], "natural"))

    return queries

"""Multi-agent role definitions for coordinated theorem proving.

Each role defines a specialized agent with a specific tool subset
and system prompt addendum, enabling divide-and-conquer proof strategies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRole:
    """Definition of a specialized agent role."""
    name: str
    description: str
    tools: list[str]
    system_prompt_addendum: str


STRATEGIST = AgentRole(
    name="strategist",
    description="Plans proof strategies and decomposes theorems into subgoals.",
    tools=["symbolic_compute", "paper_search", "skill_invoke", "web_search", "autoformalize"],
    system_prompt_addendum=(
        "You are the Strategist. Your job is to analyze the theorem, "
        "identify the proof technique (induction, contradiction, etc.), "
        "and decompose the problem into subgoals. You do NOT write Lean code — "
        "you produce a proof sketch with clear steps for the Prover to formalize."
    ),
)

SEARCHER = AgentRole(
    name="searcher",
    description="Finds relevant Mathlib lemmas and references for proof construction.",
    tools=["mathlib_search", "web_search"],
    system_prompt_addendum=(
        "You are the Searcher. Given a list of subgoals or proof steps, "
        "find the most relevant Mathlib lemmas using all search modes "
        "(semantic, type, natural, name). Return a ranked list of lemma names "
        "with their type signatures for each subgoal."
    ),
)

PROVER = AgentRole(
    name="prover",
    description="Constructs Lean 4 proofs using tactics guided by strategy and lemmas.",
    tools=["lean_tactic", "mathlib_search"],
    system_prompt_addendum=(
        "You are the Prover. Given a proof strategy and relevant lemmas, "
        "construct the proof using lean_tactic step by step. Focus on applying "
        "the suggested tactics and lemmas. If a tactic fails, try alternatives "
        "from the candidate list before requesting help from the Strategist."
    ),
)

VERIFIER = AgentRole(
    name="verifier",
    description="Verifies complete Lean 4 proofs with lean_prover.",
    tools=["lean_prover"],
    system_prompt_addendum=(
        "You are the Verifier. Your job is to verify complete Lean 4 proofs "
        "using lean_prover. Check that the proof compiles, has no sorry's, "
        "and all goals are closed. Report any errors with clear diagnostics."
    ),
)

ALL_ROLES = [STRATEGIST, SEARCHER, PROVER, VERIFIER]

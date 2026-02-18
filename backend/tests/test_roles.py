"""Tests for multi-agent role definitions."""

from bourbaki.agent.roles import STRATEGIST, SEARCHER, PROVER, VERIFIER, ALL_ROLES


def test_all_roles_defined():
    assert len(ALL_ROLES) == 4
    names = {r.name for r in ALL_ROLES}
    assert names == {"strategist", "searcher", "prover", "verifier"}


def test_strategist_tools():
    assert "symbolic_compute" in STRATEGIST.tools
    assert "paper_search" in STRATEGIST.tools
    assert "skill_invoke" in STRATEGIST.tools
    # Should NOT have prover tools
    assert "lean_tactic" not in STRATEGIST.tools
    assert "lean_prover" not in STRATEGIST.tools


def test_searcher_tools():
    assert "mathlib_search" in SEARCHER.tools
    assert "web_search" in SEARCHER.tools
    assert "lean_prover" not in SEARCHER.tools


def test_prover_tools():
    assert "lean_tactic" in PROVER.tools
    assert "mathlib_search" in PROVER.tools
    assert "paper_search" not in PROVER.tools


def test_verifier_tools():
    assert "lean_prover" in VERIFIER.tools
    assert len(VERIFIER.tools) == 1  # Verifier only needs lean_prover


def test_roles_have_prompt_addendums():
    for role in ALL_ROLES:
        assert len(role.system_prompt_addendum) > 20
        assert role.description

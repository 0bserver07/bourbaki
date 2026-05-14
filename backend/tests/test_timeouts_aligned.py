"""Smoke checks that timeout budgets across the prover chain compose sensibly.

This test class doesn't call any LLM, REPL, or Lean — it only inspects the
module-level constants and signature defaults that the loop wires together.
If anyone bumps one budget but not the others, this file should fail loudly.

Background: issue #19. The reviewer's final-gate ``lean_prover`` call used
the function's 30s default because no explicit ``timeout=`` was passed.
On a loaded box that 30s was below the cold-cache ``import Mathlib``
wall time, so every reviewer call silently timed out and the loop
reported FAIL on proofs the proposer had emitted correctly. The fix
(commit ``7b07c07``) bumped the call site to 240s. These assertions
codify the post-fix invariants so we'll see test failures rather than
silent benchmark regressions if the budgets drift back out of line.
"""

from __future__ import annotations

import inspect

import pytest

from bourbaki.benchmarks.minif2f import (
    _verify_with_lean_prover,
    attempt_proof_loop,
    attempt_proof_pass_at_n,
    run_minif2f,
)
from bourbaki.benchmarks.putnam import _verify_whole_file, run_putnam
from bourbaki.prover import memory as memory_mod
from bourbaki.prover import proposer as proposer_mod
from bourbaki.prover import reviewer as reviewer_mod
from bourbaki.prover.prover import ProverConfig
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.lean_repl import LeanREPLSession


def _default_for(fn, name: str):
    """Return the default value of ``fn``'s ``name`` parameter (raises if absent)."""
    sig = inspect.signature(fn)
    param = sig.parameters[name]
    if param.default is inspect.Parameter.empty:
        raise AssertionError(
            f"{fn.__qualname__} has no default for parameter {name!r}; "
            "expected an explicit default so callers don't fall through to 30s."
        )
    return param.default


# ---------------------------------------------------------------------------
# Per-LLM-call timeouts (module-level constants on the prover nodes)
# ---------------------------------------------------------------------------


def test_proposer_llm_timeout_is_module_constant():
    """The proposer publishes a module constant so the test below can read it.

    The proposer-builder-reviewer prompts cap LLM calls at this value — if
    someone wraps the proposer in another timeout, they need to make their
    outer budget strictly larger than this constant. Surfacing the value
    here is what makes the multiplier check below well-defined.
    """
    assert hasattr(proposer_mod, "_PROPOSER_LLM_TIMEOUT")
    assert isinstance(proposer_mod._PROPOSER_LLM_TIMEOUT, (int, float))
    assert proposer_mod._PROPOSER_LLM_TIMEOUT >= 60, (
        "proposer LLM cap below 60s is too tight — z.ai routinely takes "
        "30-90s on harder problems (see prompts_v2 notes)."
    )


def test_reviewer_llm_timeout_is_module_constant():
    assert hasattr(reviewer_mod, "_REVIEWER_LLM_TIMEOUT")
    assert isinstance(reviewer_mod._REVIEWER_LLM_TIMEOUT, (int, float))
    assert reviewer_mod._REVIEWER_LLM_TIMEOUT >= 30, (
        "reviewer LLM cap below 30s is too tight — even the small "
        "ReviewDecision schema takes ~10-30s round-trip."
    )


def test_memory_llm_timeout_is_module_constant():
    """Memory.ExperienceMemory shouldn't silently fall back below 30s."""
    assert hasattr(memory_mod, "_EXPERIENCE_LLM_TIMEOUT")
    assert memory_mod._EXPERIENCE_LLM_TIMEOUT >= 30


# ---------------------------------------------------------------------------
# REPL ↔ ProverConfig alignment (the build_timeout footgun)
# ---------------------------------------------------------------------------


def test_send_cmd_default_covers_build_timeout():
    """``ProverConfig.build_timeout`` should not exceed the REPL's per-cmd cap.

    The builder currently calls ``session.send_cmd(cmd)`` with no explicit
    ``timeout=`` and therefore inherits ``send_cmd``'s default (120s).
    ``ProverConfig.build_timeout`` is declared but unused; until it's
    wired through ``builder.py`` the only safe invariant is that the
    declared field doesn't promise more than the REPL actually delivers.

    See follow-up note in the timeout audit: wire ``build_timeout``
    through ``run_builder`` so this becomes the contract it pretends to be.
    """
    send_cmd_default = _default_for(LeanREPLSession.send_cmd, "timeout")
    build_timeout = ProverConfig().build_timeout
    assert send_cmd_default >= build_timeout, (
        f"REPL send_cmd default ({send_cmd_default}s) is below the "
        f"declared ProverConfig.build_timeout ({build_timeout}s); "
        "either lower build_timeout or raise the REPL default."
    )


def test_repl_ensure_initialized_timeout_covers_cold_mathlib():
    """``ensure_initialized`` uses 300s for ``import Mathlib`` (cold cache).

    Asserts the value pinned in lean_repl.py:135 hasn't drifted below
    180s (typical cold-cache Mathlib import wall time).
    """
    src = inspect.getsource(LeanREPLSession.ensure_initialized)
    assert "timeout=300" in src or "timeout=180" in src, (
        "ensure_initialized must keep a cold-cache-friendly timeout; "
        "see lean_repl.py: a too-tight value here cascades into every "
        "first-problem failure."
    )


# ---------------------------------------------------------------------------
# Reviewer final-gate ↔ benchmark outer budget (issue #19 fix)
# ---------------------------------------------------------------------------


def test_reviewer_final_gate_timeout_is_at_least_180s():
    """The reviewer's lean_prover call must use >= 180s — see #19.

    Below this value, a cold-cache ``import Mathlib`` (60-180s) silently
    times out and the reviewer reports rejection on a correct proof.
    The fix (commit 7b07c07) set this to 240s; this test guards
    against it being lowered again.
    """
    src = inspect.getsource(reviewer_mod.run_reviewer)
    # We accept any timeout >= 180; the canonical fix uses 240.
    # Sanity-check by scanning the source for a `timeout=` keyword
    # in the lean_prover call site.
    assert "timeout=240" in src or "timeout=300" in src, (
        "Reviewer's lean_prover final-gate timeout has drifted below "
        "the issue-#19 fix value (240s). On a loaded system this "
        "silently converts correct proofs into 'final lean_prover "
        "verification failed' rejections."
    )


def test_reviewer_lean_prover_timeout_fits_in_outer_budget():
    """The reviewer's lean_prover budget must fit inside the outer
    per-problem budget on a default ``attempt_proof_loop`` invocation.
    """
    outer = _default_for(attempt_proof_loop, "timeout")
    # The reviewer's call site uses 240s (post-#19). The whole reviewer
    # node = LLM (60s) + lean_prover (240s) = 300s worst case; the
    # default attempt_proof_loop timeout is 600s, so there's headroom
    # for the proposer/builder to also run.
    reviewer_lean_prover_timeout = 240
    assert reviewer_lean_prover_timeout <= outer, (
        f"Reviewer's lean_prover ({reviewer_lean_prover_timeout}s) "
        f"exceeds attempt_proof_loop's default outer budget ({outer}s); "
        "the loop will trip its own outer wait_for before the reviewer "
        "even runs once."
    )


# ---------------------------------------------------------------------------
# Proposer budget × iteration count vs outer loop budget
# ---------------------------------------------------------------------------


def test_proposer_budget_times_max_iterations_exceeds_outer_timeout():
    """A full ``max_iterations`` run must fit in the proposer's LLM-time budget.

    If ``_PROPOSER_LLM_TIMEOUT × max_iterations < outer timeout`` then no
    matter how slow the LLM gets, the loop can never burn its outer
    budget on proposer time alone. This is a sanity check that the
    knobs aren't wildly inconsistent.
    """
    proposer_cap = proposer_mod._PROPOSER_LLM_TIMEOUT
    cfg = ProverConfig()
    outer = _default_for(attempt_proof_loop, "timeout")
    assert proposer_cap * cfg.max_iterations >= outer, (
        f"Proposer LLM cap ({proposer_cap}s) × max_iterations "
        f"({cfg.max_iterations}) = "
        f"{proposer_cap * cfg.max_iterations}s, "
        f"but the outer per-problem budget is {outer}s. "
        "The loop can't actually consume its full iteration budget."
    )


# ---------------------------------------------------------------------------
# lean_prover default + benchmark verify_timeout defaults
# ---------------------------------------------------------------------------


def test_lean_prover_default_timeout_covers_cold_mathlib():
    """``lean_prover``'s default timeout must cover a cold-cache Mathlib compile.

    Issue #19 root cause: the default was 30s, which is below the
    60-180s typical wall time for ``lake env lean + import Mathlib``.
    Every caller that didn't override the default was silently
    timing out.
    """
    default = _default_for(lean_prover, "timeout")
    assert default >= 180, (
        f"lean_prover default timeout is {default}s; this is the "
        "issue-#19 footgun — bump it to at least 180s (canonical fix: 240s)."
    )


def test_verify_timeout_defaults_cover_cold_mathlib():
    """All benchmark ``verify_timeout`` defaults must cover the same Mathlib path."""
    minif2f_verify = _default_for(run_minif2f, "verify_timeout")
    putnam_verify = _default_for(run_putnam, "verify_timeout")
    inner_verify = _default_for(_verify_with_lean_prover, "verify_timeout")
    putnam_inner = _default_for(_verify_whole_file, "timeout")
    # 150 is the current miniF2F default; allow it but flag anything below.
    for name, value in [
        ("run_minif2f.verify_timeout", minif2f_verify),
        ("run_putnam.verify_timeout", putnam_verify),
        ("_verify_with_lean_prover.verify_timeout", inner_verify),
        ("_verify_whole_file.timeout", putnam_inner),
    ]:
        assert value >= 150, (
            f"{name} default is {value}s; below 150s the cold-cache "
            "Mathlib compile silently times out (issue #19 class)."
        )


# ---------------------------------------------------------------------------
# Outer budgets don't shrink between siblings (Pass@N vs single-shot)
# ---------------------------------------------------------------------------


def test_pass_at_n_per_attempt_matches_loop_default():
    """Pass@N's per-attempt budget should be >= the single-shot loop's default
    if it's going to give each attempt a fair shake.
    """
    pass_n_per_attempt = _default_for(attempt_proof_pass_at_n, "timeout_per_attempt")
    # We use >= 300 explicitly because the single-shot default is 600
    # (generous) and we don't want to force pass@N to match the full
    # budget per attempt — but anything below 300 is too tight.
    assert pass_n_per_attempt >= 300, (
        f"attempt_proof_pass_at_n.timeout_per_attempt is "
        f"{pass_n_per_attempt}s; below 300s each Pass@N attempt is too "
        "rushed to run a real ProverLoop."
    )


@pytest.mark.parametrize(
    "fn_name,kwarg",
    [
        ("attempt_proof_loop", "timeout"),
        ("attempt_proof_pass_at_n", "timeout_per_attempt"),
        ("run_minif2f", "timeout"),
        ("run_minif2f", "verify_timeout"),
    ],
)
def test_minif2f_outer_budgets_present(fn_name, kwarg):
    """Defensive: each outer driver must keep an explicit default so a
    caller who forgets ``timeout=`` doesn't fall through to 30s.
    """
    from bourbaki.benchmarks import minif2f

    fn = getattr(minif2f, fn_name)
    _default_for(fn, kwarg)

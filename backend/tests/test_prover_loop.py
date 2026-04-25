"""Integration tests for ProverLoop.run.

Each node (proposer/builder/reviewer) has its own unit tests; these tests
verify the loop driver wires them together correctly and routes per
``proposer-builder-loop.md`` §3.

All node calls are mocked — no LLM, no Lean REPL, no network.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from bourbaki.benchmarks.loader import MiniF2FProblem
from bourbaki.prover import feedback as fb
from bourbaki.prover.prover import ProverConfig, ProverLoop
from bourbaki.prover.state import ProposalMessage


def _make_problem() -> MiniF2FProblem:
    return MiniF2FProblem(
        id="mathd_test_1",
        source="mathd",
        split="valid",
        statement="theorem mathd_test_1 : 1 = 1 := sorry",
        imports=["Mathlib"],
        file_path="/dev/null",
        full_lean_code=(
            "import Mathlib\n\ntheorem mathd_test_1 : 1 = 1 := sorry\n"
        ),
    )


def _make_loop(monkeypatch, *, proposer_returns, builder_returns, reviewer_returns):
    """Build a ProverLoop with patched node delegators."""
    cfg = ProverConfig(memory_cls="MemorylessMemory", max_iterations=5)
    session = AsyncMock()  # never actually called because _builder is patched
    loop = ProverLoop(config=cfg, session=session)

    proposer_mock = AsyncMock(side_effect=list(proposer_returns))
    builder_mock = AsyncMock(side_effect=list(builder_returns))
    reviewer_mock = AsyncMock(side_effect=list(reviewer_returns))

    monkeypatch.setattr(loop, "_proposer", proposer_mock)
    monkeypatch.setattr(loop, "_builder", builder_mock)
    monkeypatch.setattr(loop, "_reviewer", reviewer_mock)
    return loop, proposer_mock, builder_mock, reviewer_mock


@pytest.mark.asyncio
async def test_loop_terminates_on_review_approved(monkeypatch):
    proposal = ProposalMessage(
        reasoning="rfl works", code="theorem mathd_test_1 : 1 = 1 := rfl", iteration=1
    )
    loop, p, b, r = _make_loop(
        monkeypatch,
        proposer_returns=[proposal],
        builder_returns=[fb.build_success()],
        reviewer_returns=[fb.review_approved("clean")],
    )

    state = await loop.run(_make_problem())

    assert p.await_count == 1
    assert b.await_count == 1
    assert r.await_count == 1
    assert state.approved is True
    assert state.verified is True
    # Regression: final_proof_code must be the assembled standalone source
    # (preamble + proposal), NOT just proposal.code. The outer benchmark
    # verifier checks proof_code directly; if the preamble (with
    # set_option maxHeartbeats 0) is missing, default heartbeats can flip
    # an honest success into a phantom false positive.
    assert state.final_proof_code is not None
    assert proposal.code in state.final_proof_code
    assert "import Mathlib" in state.final_proof_code


@pytest.mark.asyncio
async def test_loop_terminates_on_proposer_terminal_feedback(monkeypatch):
    """Proposer returning a terminal FeedbackMessage (e.g. max_iterations) ends the loop."""
    loop, p, b, r = _make_loop(
        monkeypatch,
        proposer_returns=[fb.max_iterations(5)],
        builder_returns=[],
        reviewer_returns=[],
    )

    state = await loop.run(_make_problem())

    assert p.await_count == 1
    assert b.await_count == 0
    assert r.await_count == 0
    assert state.approved is False
    assert state.verified is False


@pytest.mark.asyncio
async def test_loop_retries_on_build_failure_then_succeeds(monkeypatch):
    """Build failure → memory → next proposer iteration → eventual approval."""
    p1 = ProposalMessage(reasoning="try x", code="theorem mathd_test_1 : 1 = 1 := by sorry", iteration=1)
    p2 = ProposalMessage(reasoning="try rfl", code="theorem mathd_test_1 : 1 = 1 := rfl", iteration=2)

    loop, p, b, r = _make_loop(
        monkeypatch,
        proposer_returns=[p1, p2],
        builder_returns=[fb.build_failed("type mismatch"), fb.build_success()],
        reviewer_returns=[fb.review_approved("ok")],
    )

    state = await loop.run(_make_problem())

    assert p.await_count == 2
    assert b.await_count == 2
    assert r.await_count == 1
    assert state.verified is True
    assert state.iteration == 2


@pytest.mark.asyncio
async def test_loop_retries_on_review_rejected(monkeypatch):
    """Reviewer rejection → memory → next proposer iteration."""
    p1 = ProposalMessage(reasoning="r1", code="theorem mathd_test_1 : 1 = 1 := rfl", iteration=1)
    p2 = ProposalMessage(reasoning="r2", code="theorem mathd_test_1 : 1 = 1 := rfl", iteration=2)

    loop, p, b, r = _make_loop(
        monkeypatch,
        proposer_returns=[p1, p2],
        builder_returns=[fb.build_success(), fb.build_success()],
        reviewer_returns=[fb.review_rejected("statement modified"), fb.review_approved("ok")],
    )

    state = await loop.run(_make_problem())

    assert p.await_count == 2
    assert b.await_count == 2
    assert r.await_count == 2
    assert state.verified is True


@pytest.mark.asyncio
async def test_loop_state_initialized_from_problem(monkeypatch):
    """State.preamble extracted from full_lean_code; problem_id wired."""
    loop, _, _, _ = _make_loop(
        monkeypatch,
        proposer_returns=[fb.max_iterations(5)],
        builder_returns=[],
        reviewer_returns=[],
    )
    problem = _make_problem()

    state = await loop.run(problem)

    assert state.problem_id == "mathd_test_1"
    assert state.target_theorem.startswith("theorem mathd_test_1")
    assert state.full_file == problem.full_lean_code
    assert state.max_iterations == 5
    assert "import Mathlib" in state.preamble


@pytest.mark.asyncio
async def test_loop_target_theorem_includes_sorry_placeholder(monkeypatch):
    """``state.target_theorem`` must carry ``:= sorry`` so the proposer's
    <target> block matches the file content. The loader strips the placeholder
    when populating ``problem.statement``; the loop adds it back.
    """
    loop, _, _, _ = _make_loop(
        monkeypatch,
        proposer_returns=[fb.max_iterations(5)],
        builder_returns=[],
        reviewer_returns=[],
    )

    state = await loop.run(_make_problem())

    assert ":= by" in state.target_theorem
    assert "sorry" in state.target_theorem


@pytest.mark.asyncio
async def test_loop_terminates_on_repeated_parsing_failures(monkeypatch):
    """Repeated ``structured_output_parsing_failed`` (non-terminal feedback,
    not a ProposalMessage) must not loop forever. The driver's local attempt
    counter has to bite even when ``state.iteration`` never advances.
    """
    parse_fails = [fb.structured_output_parsing_failed(f"fail {i}") for i in range(20)]
    loop, p, b, r = _make_loop(
        monkeypatch,
        proposer_returns=parse_fails,
        builder_returns=[],
        reviewer_returns=[],
    )
    # max_iterations = 5 from _make_loop default.
    state = await loop.run(_make_problem())

    # Loop must have called the proposer at most max_iterations + 1 times
    # (the +1 is the call that produces the terminal max_iterations feedback).
    assert p.await_count <= 5
    assert b.await_count == 0
    assert r.await_count == 0
    # Final state never advanced past iteration 0 (no ProposalMessage ever appended).
    assert state.iteration == 0
    assert state.verified is False
    # Last feedback should be the synthetic max_iterations terminator.
    assert state.last_feedback is not None
    assert state.last_feedback.kind == "max_iterations"


def test_extract_preamble_pulls_imports_and_opens():
    from bourbaki.prover.prover import _extract_preamble

    full = (
        "import Mathlib\n"
        "open Nat\n"
        "set_option maxHeartbeats 400000\n"
        "\n"
        "theorem mathd_test_1 : 1 = 1 := by\n"
        "  sorry\n"
    )
    statement = "theorem mathd_test_1 : 1 = 1"
    pre = _extract_preamble(full, statement)
    assert "import Mathlib" in pre
    assert "open Nat" in pre
    assert "set_option" in pre
    assert "theorem" not in pre


def test_extract_preamble_returns_empty_when_statement_missing():
    from bourbaki.prover.prover import _extract_preamble

    pre = _extract_preamble("import Mathlib\n\n", "theorem absent : True")
    assert pre == ""


def test_route_proposer_proposal_continues():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.messages.append(
        ProposalMessage(reasoning="r", code="c", iteration=1)
    )
    assert loop._route_proposer(state) == "continue"


def test_route_proposer_terminal_feedback_ends():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.messages.append(fb.max_iterations(5))
    assert loop._route_proposer(state) == "end"


def test_route_proposer_non_proposal_retries():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.messages.append(fb.structured_output_parsing_failed("e"))
    assert loop._route_proposer(state) == "retry"


def test_route_builder_success_continues():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.last_feedback = fb.build_success()
    assert loop._route_builder(state) == "continue"


def test_route_builder_failure_retries():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.last_feedback = fb.build_failed("err")
    assert loop._route_builder(state) == "retry"


def test_route_builder_terminal_feedback_ends():
    """Regression: builder issuing terminal feedback (e.g. ``max_iterations``)
    must end the loop instead of routing back to ``retry``.
    """
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.last_feedback = fb.max_iterations(5)
    assert loop._route_builder(state) == "end"


def test_route_reviewer_terminal_feedback_ends():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.last_feedback = fb.max_iterations(5)
    assert loop._route_reviewer(state) == "end"


def test_route_reviewer_approved_continues():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.last_feedback = fb.review_approved()
    assert loop._route_reviewer(state) == "continue"


def test_route_reviewer_rejected_retries():
    cfg = ProverConfig()
    loop = ProverLoop(cfg, session=AsyncMock())
    from bourbaki.prover.state import ProverState

    state = ProverState(problem_id="x", target_theorem="t")
    state.last_feedback = fb.review_rejected("nope")
    assert loop._route_reviewer(state) == "retry"

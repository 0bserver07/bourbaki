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


# ---------------------------------------------------------------------------
# Observability: loop entry/exit + route-decision logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_logs_entry_exit_and_route_decisions(monkeypatch, caplog):
    """The loop must emit INFO logs for entry, exit, and every routing
    decision (proposer/builder/reviewer) — these paths were silent."""
    import logging

    proposal = ProposalMessage(
        reasoning="rfl works", code="theorem mathd_test_1 : 1 = 1 := rfl", iteration=1
    )
    loop, _, _, _ = _make_loop(
        monkeypatch,
        proposer_returns=[proposal],
        builder_returns=[fb.build_success()],
        reviewer_returns=[fb.review_approved("clean")],
    )

    with caplog.at_level(logging.INFO, logger="bourbaki.prover.prover"):
        await loop.run(_make_problem())

    messages = [r.getMessage() for r in caplog.records]
    joined = "\n".join(messages)
    # entry names the problem and the proposer/reviewer path
    assert any(
        "entering for problem 'mathd_test_1'" in m and "proposer=single" in m
        for m in messages
    )
    assert any("exiting 'mathd_test_1'" in m for m in messages)
    # one route line per node, each naming the node and outcome
    assert any("node=proposer outcome=continue" in m for m in messages)
    assert any("node=builder outcome=continue" in m for m in messages)
    assert any("node=reviewer outcome=continue" in m for m in messages)
    # exit line carries verified status
    assert "verified=True" in joined


@pytest.mark.asyncio
async def test_loop_logs_max_iterations_exit(monkeypatch, caplog):
    """The attempts-exceeded branch must log an INFO line, not exit silently."""
    import logging

    parse_fails = [fb.structured_output_parsing_failed(f"f{i}") for i in range(20)]
    loop, _, _, _ = _make_loop(
        monkeypatch,
        proposer_returns=parse_fails,
        builder_returns=[],
        reviewer_returns=[],
    )

    with caplog.at_level(logging.INFO, logger="bourbaki.prover.prover"):
        await loop.run(_make_problem())

    messages = [r.getMessage() for r in caplog.records]
    assert any("exceeded max_iterations" in m for m in messages)
    # the retry route on the proposer should also be logged
    assert any("node=proposer outcome=retry" in m for m in messages)


@pytest.mark.asyncio
async def test_loop_logs_builder_retry_route(monkeypatch, caplog):
    """A build failure must produce a builder route=retry INFO log."""
    import logging

    p1 = ProposalMessage(
        reasoning="x", code="theorem mathd_test_1 : 1 = 1 := by sorry", iteration=1
    )
    p2 = ProposalMessage(
        reasoning="rfl", code="theorem mathd_test_1 : 1 = 1 := rfl", iteration=2
    )
    loop, _, _, _ = _make_loop(
        monkeypatch,
        proposer_returns=[p1, p2],
        builder_returns=[fb.build_failed("type mismatch"), fb.build_success()],
        reviewer_returns=[fb.review_approved("ok")],
    )

    with caplog.at_level(logging.INFO, logger="bourbaki.prover.prover"):
        await loop.run(_make_problem())

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "node=builder outcome=retry" in m and "feedback_kind=build_failed" in m
        for m in messages
    )


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


# ---------------------------------------------------------------------------
# Per-LLM-call timeout threading (proposer / reviewer)
#
# Regression for the Kimi K2.6 1T / gpt-oss:120b timeout footgun: the
# hardcoded 90s _PROPOSER_LLM_TIMEOUT was too tight for frontier
# proposer models. ProverConfig now exposes proposer_llm_timeout and
# reviewer_llm_timeout, which must reach ``run_proposer(llm_timeout=...)``
# and ``run_reviewer(llm_timeout=...)`` respectively.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_forwards_proposer_llm_timeout_to_run_proposer(monkeypatch):
    """``ProverLoop._proposer`` must pass ``cfg.proposer_llm_timeout`` as the
    ``llm_timeout`` kwarg to :func:`run_proposer`.
    """
    from bourbaki.prover import prover as prover_mod

    captured: dict = {}

    async def fake_run_proposer(state, **kwargs):
        captured.update(kwargs)
        # End the loop immediately on the next route check.
        return fb.max_iterations(5)

    monkeypatch.setattr(prover_mod, "run_proposer", fake_run_proposer)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_llm_timeout=600.0,
    )
    loop = ProverLoop(config=cfg, session=AsyncMock())
    await loop.run(_make_problem())

    assert captured.get("llm_timeout") == 600.0


@pytest.mark.asyncio
async def test_loop_forwards_reviewer_llm_timeout_to_run_reviewer(monkeypatch):
    """``ProverLoop._reviewer`` must pass ``cfg.reviewer_llm_timeout`` as the
    ``llm_timeout`` kwarg to :func:`run_reviewer`.
    """
    from bourbaki.prover import prover as prover_mod

    proposal = ProposalMessage(
        reasoning="rfl works",
        code="theorem mathd_test_1 : 1 = 1 := rfl",
        iteration=1,
    )

    async def fake_run_proposer(state, **kwargs):
        return proposal

    async def fake_run_builder(state, session, **kwargs):
        return fb.build_success()

    captured: dict = {}

    async def fake_run_reviewer(state, **kwargs):
        captured.update(kwargs)
        return fb.review_approved("ok")

    monkeypatch.setattr(prover_mod, "run_proposer", fake_run_proposer)
    monkeypatch.setattr(prover_mod, "run_builder", fake_run_builder)
    monkeypatch.setattr(prover_mod, "run_reviewer", fake_run_reviewer)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        reviewer_llm_timeout=450.0,
    )
    loop = ProverLoop(config=cfg, session=AsyncMock())
    await loop.run(_make_problem())

    assert captured.get("llm_timeout") == 450.0


@pytest.mark.asyncio
async def test_loop_defaults_forward_module_constants(monkeypatch):
    """When the caller leaves ``ProverConfig`` defaults alone, the loop
    must still forward the module-constant values explicitly — not rely
    on ``run_proposer`` / ``run_reviewer`` falling through to their own
    defaults. Future refactors could drop those module defaults and we'd
    silently lose coverage.
    """
    from bourbaki.prover import prover as prover_mod
    from bourbaki.prover import proposer as proposer_mod
    from bourbaki.prover import reviewer as reviewer_mod

    proposal = ProposalMessage(
        reasoning="r", code="theorem mathd_test_1 : 1 = 1 := rfl", iteration=1
    )
    proposer_captured: dict = {}
    reviewer_captured: dict = {}

    async def fake_run_proposer(state, **kwargs):
        proposer_captured.update(kwargs)
        return proposal

    async def fake_run_builder(state, session, **kwargs):
        return fb.build_success()

    async def fake_run_reviewer(state, **kwargs):
        reviewer_captured.update(kwargs)
        return fb.review_approved("ok")

    monkeypatch.setattr(prover_mod, "run_proposer", fake_run_proposer)
    monkeypatch.setattr(prover_mod, "run_builder", fake_run_builder)
    monkeypatch.setattr(prover_mod, "run_reviewer", fake_run_reviewer)

    cfg = ProverConfig(memory_cls="MemorylessMemory", max_iterations=5)
    loop = ProverLoop(config=cfg, session=AsyncMock())
    await loop.run(_make_problem())

    assert proposer_captured["llm_timeout"] == proposer_mod._PROPOSER_LLM_TIMEOUT
    assert reviewer_captured["llm_timeout"] == reviewer_mod._REVIEWER_LLM_TIMEOUT


# ---------------------------------------------------------------------------
# Multi-provider parallel-fanout proposer routing
#
# ProverConfig.proposer_configs gates which proposer entrypoint
# ProverLoop._proposer calls. None / empty -> legacy run_proposer; non-empty
# list -> parallel_propose with the supplied configs. Reviewer is unaffected.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_calls_run_proposer_when_proposer_configs_is_none(monkeypatch):
    """``ProverConfig(proposer_configs=None)`` keeps the single-model path:
    ``ProverLoop._proposer`` must invoke :func:`run_proposer` and must NOT
    call :func:`parallel_propose`.
    """
    from bourbaki.prover import prover as prover_mod

    run_proposer_mock = AsyncMock(return_value=fb.max_iterations(5))
    parallel_propose_mock = AsyncMock()

    monkeypatch.setattr(prover_mod, "run_proposer", run_proposer_mock)
    monkeypatch.setattr(prover_mod, "parallel_propose", parallel_propose_mock)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=None,
    )
    loop = ProverLoop(config=cfg, session=AsyncMock())
    await loop.run(_make_problem())

    assert run_proposer_mock.await_count == 1
    assert parallel_propose_mock.await_count == 0


@pytest.mark.asyncio
async def test_loop_calls_parallel_propose_when_proposer_configs_set(monkeypatch):
    """``ProverConfig(proposer_configs=[c1, c2])`` routes through
    :func:`parallel_propose`, passing the supplied configs unchanged.
    :func:`run_proposer` must NOT be called for the proposer step.
    """
    from bourbaki.prover import prover as prover_mod
    from bourbaki.prover.parallel_proposer import ProposerConfig

    c1 = ProposerConfig(model="glm:glm-5.1", llm_timeout=90.0)
    c2 = ProposerConfig(model="ollama-cloud:kimi-k2.6", llm_timeout=600.0)

    run_proposer_mock = AsyncMock()
    parallel_propose_mock = AsyncMock(return_value=fb.max_iterations(5))

    monkeypatch.setattr(prover_mod, "run_proposer", run_proposer_mock)
    monkeypatch.setattr(prover_mod, "parallel_propose", parallel_propose_mock)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=[c1, c2],
    )
    loop = ProverLoop(config=cfg, session=AsyncMock())
    await loop.run(_make_problem())

    assert run_proposer_mock.await_count == 0
    assert parallel_propose_mock.await_count == 1
    # parallel_propose(state, configs) — second positional arg is the list.
    assert parallel_propose_mock.await_args is not None
    _, call_args = parallel_propose_mock.await_args.args
    assert call_args == [c1, c2]


def test_run_minif2f_loop_proposer_configs_plumbed_into_prover_config():
    """``run_minif2f(loop_proposer_configs=...)`` must reach the constructed
    :class:`ProverConfig`. We mock :class:`LeanREPLSession` (so the runner
    takes the loop path), then capture the ``config`` object that the
    runner forwards into :func:`attempt_proof_loop`.
    """
    import asyncio as _asyncio
    import unittest.mock as _mock

    from bourbaki.benchmarks import minif2f as minif2f_mod
    from bourbaki.prover.parallel_proposer import ProposerConfig

    cfgs = [
        ProposerConfig(model="glm:glm-5.1", llm_timeout=90.0),
        ProposerConfig(model="ollama-cloud:gpt-oss:120b", llm_timeout=120.0),
    ]
    captured: dict = {}

    class _StubREPLSession:
        """Minimal stand-in. Accepts the kwargs ``run_minif2f`` passes
        and survives ``ensure_initialized`` so the runner picks the
        loop path rather than the lean_prover fallback.
        """

        _initialized = True
        env_id = 0

        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

        async def ensure_initialized(self):
            self._initialized = True

    async def fake_attempt_proof_loop(problem, session, config=None, timeout=300):
        assert config is not None
        captured["proposer_configs"] = config.proposer_configs
        return minif2f_mod.ProblemResult(
            problem_id=problem.id,
            source=problem.source,
            solved=False,
            duration_seconds=0.0,
        )

    with _mock.patch.object(minif2f_mod, "LeanREPLSession", _StubREPLSession), \
         _mock.patch.object(minif2f_mod, "attempt_proof_loop", fake_attempt_proof_loop), \
         _mock.patch.object(minif2f_mod, "load_minif2f_problems", return_value=[
             MiniF2FProblem(
                 id="mathd_test_1",
                 source="mathd",
                 split="valid",
                 statement="theorem mathd_test_1 : 1 = 1 := sorry",
                 imports=["Mathlib"],
                 file_path="/dev/null",
                 full_lean_code="import Mathlib\n\ntheorem mathd_test_1 : 1 = 1 := sorry\n",
             )
         ]):
        _asyncio.run(
            minif2f_mod.run_minif2f(
                use_loop=True,
                loop_proposer_configs=cfgs,
                verify=False,
            )
        )

    assert captured["proposer_configs"] == cfgs

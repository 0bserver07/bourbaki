"""Tests for Pass@N composition with the parallel-fanout proposer.

Pass@N is a wrapper around ``attempt_proof_loop``: each sample-of-N is one
full proposer-builder-reviewer cycle.  When the loop's proposer is itself
a parallel fanout (``ProverConfig.proposer_configs`` is non-empty),
Pass@N gives ``n_pass`` independent fanout races per problem.  These tests
codify the composition invariants so a future refactor can't silently
break it.

All tests mock ``attempt_proof_loop`` (or ``parallel_propose`` further
down the stack) — no LLM, no Lean REPL, no network.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from unittest.mock import AsyncMock

from bourbaki.benchmarks import minif2f as minif2f_mod
from bourbaki.benchmarks.loader import MiniF2FProblem
from bourbaki.benchmarks.minif2f import (
    ProblemResult,
    attempt_proof_pass_at_n,
)
from bourbaki.prover.parallel_proposer import ProposerConfig
from bourbaki.prover.prover import ProverConfig


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


def _failed_result(problem_id: str = "mathd_test_1") -> ProblemResult:
    return ProblemResult(
        problem_id=problem_id,
        source="mathd",
        solved=False,
        repl_reported=False,
        verified=False,
        proof_code=None,
        error="loop failed",
        tactics_used=0,
        duration_seconds=1.0,
        attempts=0,
    )


def _success_result(problem_id: str = "mathd_test_1") -> ProblemResult:
    return ProblemResult(
        problem_id=problem_id,
        source="mathd",
        solved=True,
        repl_reported=True,
        verified=True,
        proof_code="import Mathlib\n\ntheorem mathd_test_1 : 1 = 1 := rfl\n",
        error=None,
        tactics_used=1,
        duration_seconds=1.0,
        attempts=1,
    )


def _fanout_configs() -> list[ProposerConfig]:
    """Two-leg fanout, the shape ``--smart-fanout minimal`` produces."""
    return [
        ProposerConfig(model="glm:glm-5.1", llm_timeout=90.0),
        ProposerConfig(model="ollama-cloud:gpt-oss:120b", llm_timeout=180.0),
    ]


# ---------------------------------------------------------------------------
# 1. pass_n=1 with fanout: same path as a single-pass fanout call
# ---------------------------------------------------------------------------


async def test_pass_n_1_with_fanout_calls_loop_once(monkeypatch):
    """Pass@1 with a fanout configured must invoke ``attempt_proof_loop``
    exactly once with the ``ProverConfig`` carrying the fanout — no extra
    attempts, no unwrap of the config.  Degenerates to single-pass fanout.
    """
    captured: list[ProverConfig | None] = []

    async def fake_loop(problem, session, config=None, timeout=300):
        captured.append(config)
        return _success_result()

    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", fake_loop)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=_fanout_configs(),
    )
    session = AsyncMock()

    result = await attempt_proof_pass_at_n(
        _make_problem(), session, config=cfg, n=1, timeout_per_attempt=30,
    )

    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded is cfg
    assert forwarded is not None and forwarded.proposer_configs is not None
    assert len(forwarded.proposer_configs) == 2
    assert result.solved is True
    assert result.attempts_pass_n == 1


# ---------------------------------------------------------------------------
# 2. pass_n=3 with fanout: first attempt fails, second succeeds
# ---------------------------------------------------------------------------


async def test_pass_n_with_fanout_early_exit_on_success(monkeypatch):
    """Three Pass@N samples where the second attempt succeeds: result must
    report ``attempts_pass_n=2``, the third sample must NOT run, and the
    fanout config must be threaded into every attempt."""
    captured: list[ProverConfig | None] = []

    side_effect = [_failed_result(), _success_result(), _failed_result()]

    async def fake_loop(problem, session, config=None, timeout=300):
        captured.append(config)
        return side_effect.pop(0)

    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", fake_loop)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=_fanout_configs(),
    )
    session = AsyncMock()

    result = await attempt_proof_pass_at_n(
        _make_problem(), session, config=cfg, n=3, timeout_per_attempt=30,
    )

    assert len(captured) == 2  # Third sample never runs.
    assert result.solved is True
    assert result.verified is True
    assert result.attempts_pass_n == 2
    # Every attempt sees the SAME ProverConfig with the SAME proposer_configs.
    for forwarded in captured:
        assert forwarded is cfg
        assert forwarded is not None and forwarded.proposer_configs is not None
        assert len(forwarded.proposer_configs) == 2


# ---------------------------------------------------------------------------
# 3. pass_n=3 with fanout: all attempts fail
# ---------------------------------------------------------------------------


async def test_pass_n_with_fanout_all_fail(monkeypatch):
    """Three Pass@N samples where every attempt fails: result must report
    ``attempts_pass_n=3``, ``verified=False``, and the loop is invoked the
    full N times."""
    n = 3
    failures = [
        ProblemResult(
            problem_id="mathd_test_1",
            source="mathd",
            solved=False,
            error=f"fanout-fail-{i}",
        )
        for i in range(1, n + 1)
    ]
    mock_loop = AsyncMock(side_effect=failures)
    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", mock_loop)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=_fanout_configs(),
    )
    session = AsyncMock()

    result = await attempt_proof_pass_at_n(
        _make_problem(), session, config=cfg, n=n, timeout_per_attempt=30,
    )

    assert mock_loop.await_count == n
    assert result.solved is False
    assert result.verified is False
    assert result.attempts_pass_n == n
    # The LAST failure is what comes back (so logs can grep its error).
    assert result.error == f"fanout-fail-{n}"


# ---------------------------------------------------------------------------
# 4. proposer_configs carried verbatim into each call
# ---------------------------------------------------------------------------


async def test_pass_n_carries_proposer_configs_to_each_call(monkeypatch):
    """Every Pass@N attempt must receive the same ``proposer_configs``
    content — the loop driver must NOT mutate or rebuild it between samples.

    Pydantic's ``ProverConfig`` copies list inputs on construction (frozen
    ``ProposerConfig`` instances are reused but the list wrapper is
    rebuilt) so we assert equality + per-leg field stability rather than
    list identity.  Stability across N calls is what matters: each Pass@N
    attempt must see the same fanout legs in the same order.
    """
    seen_configs_per_call: list[list[ProposerConfig] | None] = []

    async def fake_loop(problem, session, config=None, timeout=300):
        seen_configs_per_call.append(
            config.proposer_configs if config is not None else None
        )
        return _failed_result()

    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", fake_loop)

    fanout = _fanout_configs()
    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=fanout,
    )
    session = AsyncMock()

    n = 4
    await attempt_proof_pass_at_n(
        _make_problem(), session, config=cfg, n=n, timeout_per_attempt=30,
    )

    assert len(seen_configs_per_call) == n
    first_configs = seen_configs_per_call[0]
    assert first_configs is not None
    for call_configs in seen_configs_per_call:
        # Same content across every Pass@N attempt — no mutation, no rebuild.
        assert call_configs == first_configs
        # Per-leg fields preserved.
        assert call_configs is not None
        assert call_configs[0].model == "glm:glm-5.1"
        assert call_configs[0].llm_timeout == 90.0
        assert call_configs[1].model == "ollama-cloud:gpt-oss:120b"
        assert call_configs[1].llm_timeout == 180.0


# ---------------------------------------------------------------------------
# 5. Memory is fresh on each Pass@N attempt
# ---------------------------------------------------------------------------


async def test_pass_n_memory_reset_between_attempts(monkeypatch):
    """ProverLoop builds its memory in ``__init__``.  attempt_proof_loop
    constructs a fresh ProverLoop per call, so each Pass@N attempt gets a
    new BaseMemory instance — prior failed attempts can't leak experience
    into the next sample.  This test would catch a regression that moves
    memory construction to module scope or caches it across calls.
    """
    from bourbaki.prover import prover as prover_mod

    # Keep strong references to every memory object _build_memory returns,
    # otherwise Python may garbage-collect a previous instance before the
    # next attempt runs and reuse its ``id`` — masking the test.
    memories_seen: list[object] = []
    real_build_memory = prover_mod._build_memory

    def spy_build_memory(config):
        m = real_build_memory(config)
        memories_seen.append(m)
        return m

    monkeypatch.setattr(prover_mod, "_build_memory", spy_build_memory)

    # Hold on to ProverLoop instances too so their underlying memory isn't
    # garbage-collected between attempts.
    loops_seen: list[object] = []

    async def fake_loop_passthrough(problem, session, config=None, timeout=300):
        # Build a real ProverLoop to trigger _build_memory, then return a
        # failure so Pass@N continues.  We don't actually run the loop —
        # ProverLoop.__init__ alone is enough to evidence memory rebuild.
        from bourbaki.prover.prover import ProverLoop
        loop_instance = ProverLoop(
            config=config or ProverConfig(memory_cls="MemorylessMemory"),
            session=session,
        )
        loops_seen.append(loop_instance)
        return _failed_result()

    monkeypatch.setattr(
        minif2f_mod, "attempt_proof_loop", fake_loop_passthrough,
    )

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=_fanout_configs(),
    )
    session = AsyncMock()

    n = 3
    await attempt_proof_pass_at_n(
        _make_problem(), session, config=cfg, n=n, timeout_per_attempt=30,
    )

    # _build_memory was called exactly N times, each producing a distinct
    # object — proves memory is NOT shared across Pass@N attempts.
    assert len(memories_seen) == n
    assert len(loops_seen) == n
    # Pairwise distinct references — no memory shared between attempts.
    for a_idx in range(n):
        for b_idx in range(a_idx + 1, n):
            assert memories_seen[a_idx] is not memories_seen[b_idx]
            assert loops_seen[a_idx].memory is not loops_seen[b_idx].memory  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 6. CLI: --pass-n + --smart-fanout balanced compose without argparse clash
# ---------------------------------------------------------------------------


def _load_minif2f_runner_module():
    """Load ``run_minif2f_subset`` from ``backend/scripts`` as a module."""
    import sys

    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "run_minif2f_subset.py"
    )
    scripts_dir = script_path.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "run_minif2f_subset_pass_n_under_test", script_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_pass_n_with_smart_fanout_balanced_composes(monkeypatch):
    """``--pass-n 4 --smart-fanout balanced --race-mode first_verified``
    must parse without an argparse clash and resolve to a non-empty
    ``ProposerConfig`` list with ``verify_with_lean=True`` on each leg.

    Argparse already rejects ``--smart-fanout`` together with
    ``--proposer-models`` (CLI-side mutual exclusion).  ``--pass-n`` is
    orthogonal to both — the combination is the headline composition we
    need to support.
    """
    mod = _load_minif2f_runner_module()

    # Stub smart_default_fanout so we don't depend on env vars.
    from bourbaki.prover import default_fanout as df_mod
    fake_configs = [
        ProposerConfig(model="glm:glm-5.1", label="glm-5.1"),
        ProposerConfig(
            model="ollama-cloud:gpt-oss:120b", label="gpt-oss-120b",
        ),
    ]

    def fake_smart_default_fanout(*, tier, verify_with_lean=False, **kwargs):
        assert tier == "balanced"
        assert verify_with_lean is True
        # Mirror the real function: verified-mode flips the flag on each leg.
        if verify_with_lean:
            from dataclasses import replace
            return [replace(c, verify_with_lean=True) for c in fake_configs]
        return list(fake_configs)

    monkeypatch.setattr(
        df_mod, "smart_default_fanout", fake_smart_default_fanout,
    )

    # Provide a logger sink for _resolve_proposer_configs.
    logger = logging.getLogger("test_cli_pass_n_with_smart_fanout_balanced_composes")

    # Build the args object argparse would have produced.
    import argparse
    args = argparse.Namespace(
        smart_fanout="balanced",
        proposer_models=None,
        race_mode="first_verified",
        pass_n=4,
    )

    resolved = mod._resolve_proposer_configs(args, logger)

    assert resolved is not None
    assert len(resolved) == 2
    assert all(c.verify_with_lean for c in resolved)
    # The pass_n arg is independent — no field clash on ProverConfig.
    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=20,
        proposer_configs=resolved,
        proposer_race_mode=args.race_mode,
    )
    assert cfg.proposer_configs is not None and len(cfg.proposer_configs) == 2
    assert cfg.proposer_race_mode == "first_verified"


# ---------------------------------------------------------------------------
# 7. Logging: pass-n attempt 2/4 line surfaces during multi-sample fanout runs
# ---------------------------------------------------------------------------


async def test_pass_n_fanout_logs_each_attempt(monkeypatch, caplog):
    """With ``n > 1`` and a fanout configured, each attempt must emit an
    INFO log line identifying which sample-of-N is launching and how many
    legs the fanout has.  A log reader skimming a multi-hour run shouldn't
    have to count proposer logs to figure out which Pass@N sample is in
    flight.
    """
    # All attempts fail so the loop runs the full N times.
    mock_loop = AsyncMock(
        side_effect=[_failed_result() for _ in range(4)],
    )
    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", mock_loop)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        max_iterations=5,
        proposer_configs=_fanout_configs(),
    )
    session = AsyncMock()

    caplog.set_level(logging.INFO, logger="bourbaki.benchmarks.minif2f")
    await attempt_proof_pass_at_n(
        _make_problem(), session, config=cfg, n=4, timeout_per_attempt=30,
    )

    fanout_logs = [
        rec.getMessage()
        for rec in caplog.records
        if "pass-n attempt" in rec.getMessage()
    ]
    assert len(fanout_logs) == 4
    # First attempt logs "attempt 1/4 ... [2 legs]"; last logs "attempt 4/4".
    assert "1/4" in fanout_logs[0]
    assert "2 legs" in fanout_logs[0]
    assert "4/4" in fanout_logs[3]
    # Single-model path produces a different log line — must NOT fire when
    # a fanout is configured.
    assert not any(
        "single-model loop" in msg for msg in fanout_logs
    )


async def test_pass_n_single_model_logs_each_attempt(monkeypatch, caplog):
    """Same logging guarantee for the single-model loop path (no fanout
    configured): each attempt logs which sample-of-N is launching.  The
    log line differs ("single-model loop" vs "fanout") so a log reader can
    tell at a glance which proposer flavor the run used.
    """
    mock_loop = AsyncMock(
        side_effect=[_failed_result() for _ in range(3)],
    )
    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", mock_loop)

    cfg = ProverConfig(memory_cls="MemorylessMemory", max_iterations=5)
    session = AsyncMock()

    caplog.set_level(logging.INFO, logger="bourbaki.benchmarks.minif2f")
    await attempt_proof_pass_at_n(
        _make_problem(), session, config=cfg, n=3, timeout_per_attempt=30,
    )

    pass_n_logs = [
        rec.getMessage()
        for rec in caplog.records
        if "pass-n attempt" in rec.getMessage()
    ]
    assert len(pass_n_logs) == 3
    assert all("single-model loop" in msg for msg in pass_n_logs)
    # Fanout-flavoured log must NOT appear.
    assert not any("legs" in msg for msg in pass_n_logs)


async def test_pass_n_1_does_not_log_attempt_banner(monkeypatch, caplog):
    """Pass@1 is the degenerate case — the per-attempt banner adds noise
    without information.  Guarded by ``n > 1`` in the implementation;
    this test pins the gate so a future refactor can't quietly start
    spamming attempt-of-1 lines into every single-pass benchmark run.
    """
    monkeypatch.setattr(
        minif2f_mod, "attempt_proof_loop",
        AsyncMock(return_value=_success_result()),
    )
    session = AsyncMock()

    caplog.set_level(logging.INFO, logger="bourbaki.benchmarks.minif2f")
    await attempt_proof_pass_at_n(
        _make_problem(), session, config=None, n=1, timeout_per_attempt=30,
    )

    banner_logs = [
        rec for rec in caplog.records
        if "pass-n attempt" in rec.getMessage()
    ]
    assert banner_logs == []


# ---------------------------------------------------------------------------
# Putnam-side: attempts_pass_n is forwarded from miniF2F result to Putnam result
# ---------------------------------------------------------------------------


def test_loop_result_to_putnam_carries_attempts_pass_n():
    """Regression: the miniF2F → Putnam result conversion previously dropped
    ``attempts_pass_n``, silently losing the sample-of-N index when Pass@N
    ran on Putnam.  The converter must forward the field unchanged so the
    saved Putnam JSON surfaces which attempt won.
    """
    from bourbaki.benchmarks.putnam import _loop_result_to_putnam
    from bourbaki.benchmarks.putnam_loader import PutnamProblem

    putnam_problem = PutnamProblem(
        id="putnam_2020_a1",
        year=2020,
        section="a",
        problem_number="a1",
        statement="theorem putnam_2020_a1 : 1 = 1",
        imports=["Mathlib"],
        preamble="",
        setup_block="",
        file_path="/dev/null",
        full_lean_code="import Mathlib\n\ntheorem putnam_2020_a1 : 1 = 1 := sorry\n",
        has_answer=False,
        answer_is_sorry=False,
        answer_name=None,
        docstring=None,
    )
    loop_result = ProblemResult(
        problem_id="putnam_2020_a1",
        source="putnam",
        solved=True,
        verified=True,
        attempts_pass_n=3,
    )
    putnam_result = _loop_result_to_putnam(putnam_problem, loop_result)
    assert putnam_result.attempts_pass_n == 3
    # And it lands in the serialized dict, so the saved JSON has the field.
    d = putnam_result.to_dict()
    assert d["attempts_pass_n"] == 3

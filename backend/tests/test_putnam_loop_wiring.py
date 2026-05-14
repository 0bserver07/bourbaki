"""Tests for wiring :class:`ProverLoop` into :func:`run_putnam` (#16).

These tests mock :func:`bourbaki.benchmarks.minif2f.attempt_proof_loop` (and
:func:`attempt_proof_pass_at_n`) at the ``putnam`` module's import site
plus :class:`bourbaki.tools.lean_repl.LeanREPLSession` — so no LLM, no Lean
REPL, no network are required.  The wiring's job is to:

1. Route to the loop path when ``use_loop=True``;
2. Route to Pass@N when ``use_loop=True`` and ``pass_n > 1``;
3. Stay on the existing REPL fallback when ``use_loop=False``;
4. Pass a :class:`PutnamProblem` through ``attempt_proof_loop`` without
   tripping any field-access ``AttributeError`` (duck-compatibility with
   :class:`MiniF2FProblem`).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bourbaki.benchmarks import putnam as putnam_mod
from bourbaki.benchmarks import minif2f as minif2f_mod
from bourbaki.benchmarks.minif2f import ProblemResult as MiniF2FResult
from bourbaki.benchmarks.putnam import run_putnam
from bourbaki.benchmarks.putnam_loader import PutnamProblem


def _make_putnam_problem(
    problem_id: str = "putnam_2020_a1",
    year: int = 2020,
    section: str = "a",
    has_answer: bool = False,
    answer_is_sorry: bool = False,
) -> PutnamProblem:
    """Synthetic PutnamProblem with the minimum fields the loop reads.

    Doesn't depend on the upstream putnam-bench checkout — see issue #16
    Step 4: PutnamProblem only needs ``id``, ``source`` (property),
    ``statement``, and ``full_lean_code`` for ``attempt_proof_loop``.
    """
    return PutnamProblem(
        id=problem_id,
        year=year,
        section=section,
        problem_number=f"{section}1",
        statement=f"theorem {problem_id} : 1 = 1",
        imports=["Mathlib"],
        preamble="",
        setup_block="",
        file_path="/dev/null",
        full_lean_code=(
            f"import Mathlib\n\ntheorem {problem_id} : 1 = 1 :=\n  sorry\n"
        ),
        has_answer=has_answer,
        answer_is_sorry=answer_is_sorry,
        answer_name=None,
        docstring=None,
    )


def _success_loop_result(problem_id: str) -> MiniF2FResult:
    return MiniF2FResult(
        problem_id=problem_id,
        source="putnam",
        solved=True,
        repl_reported=True,
        verified=True,
        proof_code=f"import Mathlib\n\ntheorem {problem_id} : 1 = 1 := rfl\n",
        error=None,
        tactics_used=1,
        duration_seconds=0.5,
        attempts=1,
    )


def _fake_repl_session_factory():
    """Return a constructor that produces an AsyncMock impersonating
    :class:`LeanREPLSession`.

    The mock has the methods ``run_putnam`` calls during setup —
    ``start``, ``ensure_initialized``, ``stop`` — plus the
    ``_initialized`` attribute the runner checks.  None of these touch a
    real Lean process.
    """
    def _factory(*_args, **_kwargs):
        session = AsyncMock()
        session._initialized = True
        session.env_id = 1
        # Methods used by attempt_putnam_repl (which we should NOT hit when
        # use_loop=True is set).  Default to a "no proof state" response.
        session.send_cmd = AsyncMock(return_value={"sorries": []})
        return session
    return _factory


def _patch_loader_with(problems, monkeypatch):
    """Patch ``load_putnam_problems`` so ``run_putnam`` sees the canned list
    instead of trying to read the upstream putnam-bench checkout."""
    monkeypatch.setattr(
        putnam_mod, "load_putnam_problems", lambda *a, **kw: problems,
    )


@pytest.mark.asyncio
async def test_run_putnam_use_loop_routes_to_attempt_proof_loop(monkeypatch):
    """When ``use_loop=True`` and ``pass_n=1``, run_putnam awaits
    :func:`attempt_proof_loop` (NOT the Pass@N wrapper, NOT the REPL
    fallback) with a :class:`ProverConfig` built from the ``loop_*`` args.
    """
    problem = _make_putnam_problem("putnam_2020_a1")
    _patch_loader_with([problem], monkeypatch)

    mock_loop = AsyncMock(return_value=_success_loop_result(problem.id))
    mock_pass_at_n = AsyncMock()
    mock_repl = AsyncMock()
    monkeypatch.setattr(putnam_mod, "attempt_proof_loop", mock_loop)
    monkeypatch.setattr(putnam_mod, "attempt_proof_pass_at_n", mock_pass_at_n)
    monkeypatch.setattr(putnam_mod, "attempt_putnam_repl", mock_repl)
    monkeypatch.setattr(
        putnam_mod, "LeanREPLSession", _fake_repl_session_factory(),
    )
    # Skip whole-file verification — proof_code is a stub.
    monkeypatch.setattr(
        putnam_mod, "_verify_whole_file",
        AsyncMock(return_value={"verified": True, "error": None}),
    )

    result = await run_putnam(
        problem_ids=[problem.id],
        timeout=30,
        verify_proofs=False,
        use_loop=True,
        loop_max_iterations=7,
        loop_model="glm:glm-5.1",
        loop_memory="MemorylessMemory",
        loop_memory_k=3,
        loop_enable_mathlib_search=True,
        pass_n=1,
    )

    assert mock_loop.await_count == 1
    assert mock_pass_at_n.await_count == 0
    assert mock_repl.await_count == 0

    # Verify the config built from loop_* args was passed through.
    call = mock_loop.await_args
    passed_problem = call.args[0] if call.args else call.kwargs["problem"]
    assert passed_problem.id == problem.id
    cfg = call.kwargs.get("config")
    assert cfg is not None
    assert cfg.model == "glm:glm-5.1"
    assert cfg.max_iterations == 7
    assert cfg.memory_cls == "MemorylessMemory"
    assert cfg.memory_k == 3
    assert cfg.enable_mathlib_search is True
    assert call.kwargs.get("timeout") == 30

    # The result is a Putnam-flavoured ProblemResult, not the miniF2F one.
    assert result.total == 1
    assert result.results[0].problem_id == problem.id
    assert result.results[0].year == 2020
    assert result.results[0].section == "a"
    assert result.results[0].verified is True

    # Reproducibility config must include the new knobs.
    assert result.config["use_loop"] is True
    assert result.config["loop_max_iterations"] == 7
    assert result.config["loop_model"] == "glm:glm-5.1"
    assert result.config["loop_memory"] == "MemorylessMemory"
    assert result.config["loop_memory_k"] == 3
    assert result.config["loop_enable_mathlib_search"] is True
    assert result.config["pass_n"] == 1


@pytest.mark.asyncio
async def test_run_putnam_pass_n_routes_to_pass_at_n(monkeypatch):
    """When ``use_loop=True`` and ``pass_n > 1``, run_putnam awaits
    :func:`attempt_proof_pass_at_n` (NOT the single-attempt loop)."""
    problem = _make_putnam_problem("putnam_2020_b1", section="b")
    _patch_loader_with([problem], monkeypatch)

    mock_loop = AsyncMock()
    mock_pass_at_n = AsyncMock(
        return_value=_success_loop_result(problem.id),
    )
    mock_repl = AsyncMock()
    monkeypatch.setattr(putnam_mod, "attempt_proof_loop", mock_loop)
    monkeypatch.setattr(putnam_mod, "attempt_proof_pass_at_n", mock_pass_at_n)
    monkeypatch.setattr(putnam_mod, "attempt_putnam_repl", mock_repl)
    monkeypatch.setattr(
        putnam_mod, "LeanREPLSession", _fake_repl_session_factory(),
    )
    monkeypatch.setattr(
        putnam_mod, "_verify_whole_file",
        AsyncMock(return_value={"verified": True, "error": None}),
    )

    result = await run_putnam(
        problem_ids=[problem.id],
        timeout=42,
        verify_proofs=False,
        use_loop=True,
        pass_n=4,
    )

    assert mock_pass_at_n.await_count == 1
    assert mock_loop.await_count == 0
    assert mock_repl.await_count == 0

    call = mock_pass_at_n.await_args
    assert call.kwargs.get("n") == 4
    assert call.kwargs.get("timeout_per_attempt") == 42
    cfg = call.kwargs.get("config")
    assert cfg is not None
    # Defaults that weren't overridden in the caller.
    assert cfg.model == "glm:glm-5.1"
    assert cfg.max_iterations == 50

    assert result.config["pass_n"] == 4
    assert result.config["use_loop"] is True


@pytest.mark.asyncio
async def test_run_putnam_use_loop_false_does_not_call_loop(monkeypatch):
    """Default path (``use_loop=False``): the REPL fallback runs and the
    loop entrypoints are NEVER awaited."""
    problem = _make_putnam_problem("putnam_2020_a2")
    _patch_loader_with([problem], monkeypatch)

    mock_loop = AsyncMock()
    mock_pass_at_n = AsyncMock()
    monkeypatch.setattr(putnam_mod, "attempt_proof_loop", mock_loop)
    monkeypatch.setattr(putnam_mod, "attempt_proof_pass_at_n", mock_pass_at_n)

    repl_called = {"n": 0}

    async def fake_repl(problem, session, **kwargs):
        repl_called["n"] += 1
        return putnam_mod.ProblemResult(
            problem_id=problem.id,
            year=problem.year,
            section=problem.section,
            has_answer=problem.has_answer,
            solved=False,
            error="stub",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(putnam_mod, "attempt_putnam_repl", fake_repl)
    monkeypatch.setattr(
        putnam_mod, "LeanREPLSession", _fake_repl_session_factory(),
    )

    result = await run_putnam(
        problem_ids=[problem.id],
        timeout=15,
        verify_proofs=False,
        # use_loop default is False
    )

    assert mock_loop.await_count == 0
    assert mock_pass_at_n.await_count == 0
    assert repl_called["n"] == 1
    assert result.total == 1
    # Default config knobs should reflect off-state.
    assert result.config["use_loop"] is False
    assert result.config["loop_max_iterations"] is None
    assert result.config["pass_n"] is None


@pytest.mark.asyncio
async def test_putnam_problem_is_duck_compatible_with_attempt_proof_loop(
    monkeypatch,
):
    """Regression: passing a :class:`PutnamProblem` through the loop
    entrypoint must not blow up on missing fields.  The loop reads ``.id``,
    ``.source``, ``.statement``, and ``.full_lean_code``; all four exist on
    PutnamProblem (``source`` is a property returning ``"putnam"``).
    """
    problem = _make_putnam_problem("putnam_2023_a1")

    # Confirm the attributes the loop relies on are accessible.
    assert problem.id == "putnam_2023_a1"
    assert problem.source == "putnam"  # property
    assert problem.statement.startswith("theorem ")
    assert "import Mathlib" in problem.full_lean_code

    captured = {}

    async def fake_loop(p, session, *, config=None, timeout=600):
        # Trip any AttributeError just by reading the fields the real loop
        # reads.  If PutnamProblem ever loses any of these, this fails.
        captured["id"] = p.id
        captured["source"] = p.source
        captured["statement"] = p.statement
        captured["full_lean_code"] = p.full_lean_code
        return _success_loop_result(p.id)

    monkeypatch.setattr(putnam_mod, "attempt_proof_loop", fake_loop)
    monkeypatch.setattr(
        putnam_mod, "LeanREPLSession", _fake_repl_session_factory(),
    )
    monkeypatch.setattr(
        putnam_mod, "_verify_whole_file",
        AsyncMock(return_value={"verified": True, "error": None}),
    )
    _patch_loader_with([problem], monkeypatch)

    result = await run_putnam(
        problem_ids=[problem.id],
        verify_proofs=False,
        use_loop=True,
        pass_n=1,
    )

    assert captured["id"] == problem.id
    assert captured["source"] == "putnam"
    assert captured["statement"] == problem.statement
    assert captured["full_lean_code"] == problem.full_lean_code

    # And the conversion back to a Putnam-flavoured ProblemResult preserves
    # the Putnam-only columns from the source problem.
    assert result.results[0].year == problem.year
    assert result.results[0].section == problem.section
    assert result.results[0].has_answer == problem.has_answer


@pytest.mark.asyncio
async def test_run_putnam_use_loop_skips_answer_sorry_problems(monkeypatch):
    """When a problem is answer-sorry and ``exclude_answer`` is True
    (default), the loop is NOT invoked — we still skip those problems."""
    problem = _make_putnam_problem(
        "putnam_2023_a1", has_answer=True, answer_is_sorry=True,
    )
    _patch_loader_with([problem], monkeypatch)

    mock_loop = AsyncMock()
    mock_pass_at_n = AsyncMock()
    monkeypatch.setattr(putnam_mod, "attempt_proof_loop", mock_loop)
    monkeypatch.setattr(putnam_mod, "attempt_proof_pass_at_n", mock_pass_at_n)
    monkeypatch.setattr(putnam_mod, "attempt_putnam_repl", AsyncMock())
    monkeypatch.setattr(
        putnam_mod, "LeanREPLSession", _fake_repl_session_factory(),
    )

    result = await run_putnam(
        problem_ids=[problem.id],
        verify_proofs=False,
        use_loop=True,
        pass_n=1,
        exclude_answer=True,  # default
    )

    # Answer-sorry path skips before any prover engagement.
    assert mock_loop.await_count == 0
    assert mock_pass_at_n.await_count == 0
    assert len(result.results) == 1
    assert result.results[0].skipped is True

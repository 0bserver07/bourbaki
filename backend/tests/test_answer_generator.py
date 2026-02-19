"""Tests for the PutnamBench answer generation pipeline."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bourbaki.benchmarks.answer_generator import (
    AnswerAttempt,
    build_answer_prompt,
    build_answer_verification_code,
    build_full_proof_code,
    extract_answer_type,
    extract_reference_answer,
    insert_answer,
    _clean_llm_answer,
    generate_answer,
)
from bourbaki.benchmarks.putnam_loader import PutnamProblem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_LEAN_SIMPLE = textwrap.dedent("""\
    import Mathlib

    abbrev putnam_1963_b1_solution : ℤ := sorry
    -- 2
    /--
    For what integer a does x^2-x+a divide x^13+x+90?
    -/
    theorem putnam_1963_b1
    : ∀ a : ℤ, (X^2 - X + (C a)) ∣ (X ^ 13 + X + (C 90)) ↔ a = putnam_1963_b1_solution :=
    sorry
""")

SAMPLE_LEAN_FUNCTION = textwrap.dedent("""\
    import Mathlib

    abbrev putnam_1962_a5_solution : ℕ → ℕ := sorry
    -- fun n : ℕ => n * (n + 1) * 2^(n - 2)
    /--
    Evaluate in closed form the sum.
    -/
    theorem putnam_1962_a5
    : ∀ n ≥ 2, putnam_1962_a5_solution n = ∑ k ∈ Finset.Icc 1 n, Nat.choose n k * k^2 :=
    sorry
""")

SAMPLE_LEAN_NONCOMPUTABLE = textwrap.dedent("""\
    import Mathlib

    noncomputable abbrev putnam_2024_a1_solution : Set ℕ := sorry
    ---{1}
    /--
    Determine all positive integers n.
    -/
    theorem putnam_2024_a1 :
        {n : ℕ | 0 < n} = putnam_2024_a1_solution :=
      sorry
""")

SAMPLE_LEAN_NO_COMMENT = textwrap.dedent("""\
    import Mathlib

    abbrev putnam_2000_a1_solution : ℕ := sorry

    theorem putnam_2000_a1 : putnam_2000_a1_solution = 42 :=
    sorry
""")


@pytest.fixture
def simple_problem(tmp_path: Path) -> PutnamProblem:
    """A simple answer-type problem with integer answer."""
    src = tmp_path / "putnam_1963_b1.lean"
    src.write_text(SAMPLE_LEAN_SIMPLE)
    return PutnamProblem(
        id="putnam_1963_b1",
        year=1963,
        section="b",
        problem_number="b1",
        statement=(
            "theorem putnam_1963_b1\n"
            ": ∀ a : ℤ, (X^2 - X + (C a)) ∣ (X ^ 13 + X + (C 90)) "
            "↔ a = putnam_1963_b1_solution"
        ),
        imports=["import Mathlib"],
        preamble="",
        setup_block="abbrev putnam_1963_b1_solution : ℤ := sorry\n-- 2",
        file_path=str(src),
        full_lean_code=SAMPLE_LEAN_SIMPLE,
        has_answer=True,
        answer_is_sorry=True,
        answer_name="putnam_1963_b1_solution",
        docstring="For what integer a does x^2-x+a divide x^13+x+90?",
    )


@pytest.fixture
def function_problem(tmp_path: Path) -> PutnamProblem:
    """An answer-type problem with a function answer."""
    src = tmp_path / "putnam_1962_a5.lean"
    src.write_text(SAMPLE_LEAN_FUNCTION)
    return PutnamProblem(
        id="putnam_1962_a5",
        year=1962,
        section="a",
        problem_number="a5",
        statement=(
            "theorem putnam_1962_a5\n"
            ": ∀ n ≥ 2, putnam_1962_a5_solution n = "
            "∑ k ∈ Finset.Icc 1 n, Nat.choose n k * k^2"
        ),
        imports=["import Mathlib"],
        preamble="",
        setup_block=(
            "abbrev putnam_1962_a5_solution : ℕ → ℕ := sorry\n"
            "-- fun n : ℕ => n * (n + 1) * 2^(n - 2)"
        ),
        file_path=str(src),
        full_lean_code=SAMPLE_LEAN_FUNCTION,
        has_answer=True,
        answer_is_sorry=True,
        answer_name="putnam_1962_a5_solution",
        docstring="Evaluate in closed form the sum.",
    )


@pytest.fixture
def noncomputable_problem(tmp_path: Path) -> PutnamProblem:
    """An answer-type problem with noncomputable abbrev."""
    src = tmp_path / "putnam_2024_a1.lean"
    src.write_text(SAMPLE_LEAN_NONCOMPUTABLE)
    return PutnamProblem(
        id="putnam_2024_a1",
        year=2024,
        section="a",
        problem_number="a1",
        statement=(
            "theorem putnam_2024_a1 :\n"
            "    {n : ℕ | 0 < n} = putnam_2024_a1_solution"
        ),
        imports=["import Mathlib"],
        preamble="",
        setup_block="noncomputable abbrev putnam_2024_a1_solution : Set ℕ := sorry\n---{1}",
        file_path=str(src),
        full_lean_code=SAMPLE_LEAN_NONCOMPUTABLE,
        has_answer=True,
        answer_is_sorry=True,
        answer_name="putnam_2024_a1_solution",
        docstring="Determine all positive integers n.",
    )


@pytest.fixture
def no_answer_problem() -> PutnamProblem:
    """A non-answer problem (no abbrev)."""
    return PutnamProblem(
        id="putnam_1962_a1",
        year=1962,
        section="a",
        problem_number="a1",
        statement="theorem putnam_1962_a1 (n : ℕ) : True",
        imports=["import Mathlib"],
        preamble="",
        setup_block="",
        file_path="/nonexistent/path.lean",
        full_lean_code="import Mathlib\n\ntheorem putnam_1962_a1 (n : ℕ) : True :=\n  sorry",
        has_answer=False,
        answer_is_sorry=False,
        answer_name=None,
    )


# ---------------------------------------------------------------------------
# Tests: Reference answer extraction
# ---------------------------------------------------------------------------

class TestExtractReferenceAnswer:
    def test_simple_integer(self) -> None:
        result = extract_reference_answer(
            SAMPLE_LEAN_SIMPLE, "putnam_1963_b1_solution",
        )
        assert result == "2"

    def test_function_answer(self) -> None:
        result = extract_reference_answer(
            SAMPLE_LEAN_FUNCTION, "putnam_1962_a5_solution",
        )
        assert result == "fun n : ℕ => n * (n + 1) * 2^(n - 2)"

    def test_noncomputable_triple_dash(self) -> None:
        result = extract_reference_answer(
            SAMPLE_LEAN_NONCOMPUTABLE, "putnam_2024_a1_solution",
        )
        assert result == "{1}"

    def test_no_comment(self) -> None:
        result = extract_reference_answer(
            SAMPLE_LEAN_NO_COMMENT, "putnam_2000_a1_solution",
        )
        assert result is None

    def test_wrong_name(self) -> None:
        result = extract_reference_answer(
            SAMPLE_LEAN_SIMPLE, "putnam_9999_a1_solution",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Answer type extraction
# ---------------------------------------------------------------------------

class TestExtractAnswerType:
    def test_simple_type(self) -> None:
        result = extract_answer_type(SAMPLE_LEAN_SIMPLE, "putnam_1963_b1_solution")
        assert result == "ℤ"

    def test_function_type(self) -> None:
        result = extract_answer_type(SAMPLE_LEAN_FUNCTION, "putnam_1962_a5_solution")
        assert result == "ℕ → ℕ"

    def test_set_type(self) -> None:
        result = extract_answer_type(
            SAMPLE_LEAN_NONCOMPUTABLE, "putnam_2024_a1_solution",
        )
        assert result == "Set ℕ"

    def test_nonexistent(self) -> None:
        result = extract_answer_type(SAMPLE_LEAN_SIMPLE, "nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Answer insertion
# ---------------------------------------------------------------------------

class TestInsertAnswer:
    def test_simple_insert(self) -> None:
        setup = "abbrev putnam_1963_b1_solution : ℤ := sorry\n-- 2"
        result = insert_answer(setup, "putnam_1963_b1_solution", "2")
        assert ":= 2" in result
        assert "sorry" not in result.split("\n")[0]  # sorry removed from abbrev line

    def test_function_insert(self) -> None:
        setup = "abbrev putnam_1962_a5_solution : ℕ → ℕ := sorry"
        answer = "fun n : ℕ => n * (n + 1) * 2^(n - 2)"
        result = insert_answer(setup, "putnam_1962_a5_solution", answer)
        assert answer in result
        assert ":= sorry" not in result

    def test_noncomputable_insert(self) -> None:
        setup = "noncomputable abbrev putnam_2024_a1_solution : Set ℕ := sorry"
        result = insert_answer(setup, "putnam_2024_a1_solution", "{1}")
        assert ":= {1}" in result
        assert "sorry" not in result

    def test_preserves_comment(self) -> None:
        setup = "abbrev putnam_1963_b1_solution : ℤ := sorry\n-- 2"
        result = insert_answer(setup, "putnam_1963_b1_solution", "2")
        assert "-- 2" in result

    def test_no_match_returns_original(self) -> None:
        setup = "abbrev putnam_1963_b1_solution : ℤ := sorry"
        result = insert_answer(setup, "nonexistent_name", "42")
        assert result == setup  # unchanged


# ---------------------------------------------------------------------------
# Tests: Build verification code
# ---------------------------------------------------------------------------

class TestBuildVerificationCode:
    def test_includes_all_parts(self, simple_problem: PutnamProblem) -> None:
        code = build_answer_verification_code(simple_problem, "2")
        assert "import Mathlib" in code
        assert "putnam_1963_b1_solution" in code
        assert ":= 2" in code
        assert "theorem putnam_1963_b1" in code
        # The theorem should still have sorry (proof not yet done)
        assert "sorry" in code

    def test_answer_substituted(self, noncomputable_problem: PutnamProblem) -> None:
        code = build_answer_verification_code(noncomputable_problem, "{1}")
        assert ":= {1}" in code
        # The abbrev should no longer have := sorry
        lines = code.split("\n")
        for line in lines:
            if "abbrev" in line and "putnam_2024_a1_solution" in line:
                assert "sorry" not in line


class TestBuildFullProofCode:
    def test_includes_answer_and_proof(self, simple_problem: PutnamProblem) -> None:
        proof = "theorem putnam_1963_b1 : ∀ a : ℤ, True := by simp"
        code = build_full_proof_code(simple_problem, "2", proof)
        assert "import Mathlib" in code
        assert ":= 2" in code
        assert "by simp" in code


# ---------------------------------------------------------------------------
# Tests: LLM prompt construction
# ---------------------------------------------------------------------------

class TestBuildAnswerPrompt:
    def test_contains_answer_type(self, simple_problem: PutnamProblem) -> None:
        prompt = build_answer_prompt(simple_problem, "ℤ")
        assert "ℤ" in prompt
        assert "putnam_1963_b1_solution" in prompt

    def test_contains_theorem(self, simple_problem: PutnamProblem) -> None:
        prompt = build_answer_prompt(simple_problem, "ℤ")
        assert "theorem putnam_1963_b1" in prompt

    def test_contains_docstring(self, simple_problem: PutnamProblem) -> None:
        prompt = build_answer_prompt(simple_problem, "ℤ")
        assert "integer a" in prompt  # from the docstring

    def test_contains_reference_when_provided(
        self, simple_problem: PutnamProblem,
    ) -> None:
        prompt = build_answer_prompt(simple_problem, "ℤ", reference_answer="2")
        assert "Reference Answer" in prompt
        assert "2" in prompt

    def test_no_reference_section_when_none(
        self, simple_problem: PutnamProblem,
    ) -> None:
        prompt = build_answer_prompt(simple_problem, "ℤ", reference_answer=None)
        assert "Reference Answer" not in prompt


# ---------------------------------------------------------------------------
# Tests: LLM answer cleanup
# ---------------------------------------------------------------------------

class TestCleanLLMAnswer:
    def test_plain_answer(self) -> None:
        assert _clean_llm_answer("42") == "42"

    def test_markdown_fencing(self) -> None:
        assert _clean_llm_answer("```lean\n42\n```") == "42"

    def test_markdown_lean4(self) -> None:
        assert _clean_llm_answer("```lean4\nfun n => n + 1\n```") == "fun n => n + 1"

    def test_strips_whitespace(self) -> None:
        assert _clean_llm_answer("  42  \n") == "42"

    def test_removes_abbrev_prefix(self) -> None:
        raw = "abbrev putnam_2023_a1_solution : ℕ := 18"
        assert _clean_llm_answer(raw) == "18"

    def test_removes_noncomputable_abbrev_prefix(self) -> None:
        raw = "noncomputable abbrev putnam_2024_a1_solution : Set ℕ := {1}"
        assert _clean_llm_answer(raw) == "{1}"


# ---------------------------------------------------------------------------
# Tests: Full pipeline (mocked)
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_returns_none_for_non_answer_problem(
        self, no_answer_problem: PutnamProblem,
    ) -> None:
        result = await generate_answer(no_answer_problem)
        assert result is None

    @pytest.mark.asyncio
    async def test_reference_answer_tried_first(
        self, simple_problem: PutnamProblem,
    ) -> None:
        """When the reference answer is available and proof search succeeds,
        the result should use the reference answer."""
        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.proof_tactics = ["omega"]
        mock_search_result.proof_code = "theorem putnam_1963_b1 := by omega"

        mock_verify_result = {
            "proofComplete": True,
            "success": True,
        }

        with patch(
            "bourbaki.autonomous.search_tree.prove_with_search",
            new_callable=AsyncMock,
            return_value=mock_search_result,
        ), patch(
            "bourbaki.tools.lean_prover.lean_prover",
            new_callable=AsyncMock,
            return_value=mock_verify_result,
        ):
            result = await generate_answer(simple_problem)

        assert result is not None
        assert result.answer_code == "2"
        assert result.source == "reference"
        assert result.theorem_solved is True
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_llm_fallback_when_no_reference(
        self, tmp_path: Path,
    ) -> None:
        """When there's no reference answer, LLM answers are tried."""
        src = tmp_path / "putnam_2000_a1.lean"
        src.write_text(SAMPLE_LEAN_NO_COMMENT)

        problem = PutnamProblem(
            id="putnam_2000_a1",
            year=2000,
            section="a",
            problem_number="a1",
            statement="theorem putnam_2000_a1 : putnam_2000_a1_solution = 42",
            imports=["import Mathlib"],
            preamble="",
            setup_block="abbrev putnam_2000_a1_solution : ℕ := sorry",
            file_path=str(src),
            full_lean_code=SAMPLE_LEAN_NO_COMMENT,
            has_answer=True,
            answer_is_sorry=True,
            answer_name="putnam_2000_a1_solution",
        )

        mock_search_result = MagicMock()
        mock_search_result.success = True
        mock_search_result.proof_tactics = ["rfl"]
        mock_search_result.proof_code = "theorem putnam_2000_a1 := by rfl"

        mock_verify_result = {"proofComplete": True, "success": True}

        # Mock the LLM agent to return "42"
        mock_agent_instance = MagicMock()
        mock_run_result = MagicMock()
        mock_run_result.output = "42"
        mock_agent_instance.run = AsyncMock(return_value=mock_run_result)

        with patch(
            "bourbaki.autonomous.search_tree.prove_with_search",
            new_callable=AsyncMock,
            return_value=mock_search_result,
        ), patch(
            "bourbaki.tools.lean_prover.lean_prover",
            new_callable=AsyncMock,
            return_value=mock_verify_result,
        ), patch(
            "pydantic_ai.Agent",
            return_value=mock_agent_instance,
        ):
            result = await generate_answer(problem, max_attempts=1)

        assert result is not None
        assert result.answer_code == "42"
        assert result.source == "llm"
        assert result.theorem_solved is True

    @pytest.mark.asyncio
    async def test_failed_proof_search_returns_best_attempt(
        self, simple_problem: PutnamProblem,
    ) -> None:
        """When proof search fails, the best attempt is returned."""
        mock_search_result = MagicMock()
        mock_search_result.success = False
        mock_search_result.proof_tactics = []
        mock_search_result.proof_code = None

        with patch(
            "bourbaki.autonomous.search_tree.prove_with_search",
            new_callable=AsyncMock,
            return_value=mock_search_result,
        ):
            result = await generate_answer(
                simple_problem, max_attempts=0,
            )

        assert result is not None
        assert result.theorem_solved is False
        assert result.verified is False
        assert result.source == "reference"

    @pytest.mark.asyncio
    async def test_missing_source_file(self) -> None:
        """When the source file doesn't exist, an error AnswerAttempt is returned."""
        problem = PutnamProblem(
            id="putnam_9999_a1",
            year=9999,
            section="a",
            problem_number="a1",
            statement="theorem putnam_9999_a1 : True",
            imports=["import Mathlib"],
            preamble="",
            setup_block="abbrev putnam_9999_a1_solution : ℕ := sorry",
            file_path="/nonexistent/file.lean",
            full_lean_code="",
            has_answer=True,
            answer_is_sorry=True,
            answer_name="putnam_9999_a1_solution",
        )
        result = await generate_answer(problem)
        assert result is not None
        assert result.error is not None
        assert "source file" in result.error.lower()


# ---------------------------------------------------------------------------
# Tests: AnswerAttempt dataclass
# ---------------------------------------------------------------------------

class TestAnswerAttempt:
    def test_default_values(self) -> None:
        attempt = AnswerAttempt(
            answer_code="42",
            theorem_solved=True,
            verified=True,
        )
        assert attempt.proof_tactics == []
        assert attempt.error is None
        assert attempt.source == "unknown"

    def test_all_fields(self) -> None:
        attempt = AnswerAttempt(
            answer_code="fun n => n + 1",
            theorem_solved=True,
            verified=False,
            proof_tactics=["omega", "ring"],
            error="Verification failed",
            source="llm",
        )
        assert attempt.answer_code == "fun n => n + 1"
        assert len(attempt.proof_tactics) == 2
        assert attempt.source == "llm"

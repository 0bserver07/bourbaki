"""Tests for the PutnamBench loader and benchmark runner utilities."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bourbaki.benchmarks.putnam_loader import (
    PutnamProblem,
    _parse_lean_file,
    get_putnam_stats,
    load_putnam_problems,
    DEFAULT_PUTNAM_DIR,
)
from bourbaki.benchmarks.putnam import (
    INVALID_TACTICS,
    _contains_invalid_tactic,
    ProblemResult,
    PutnamBenchmarkResult,
)


# ---------------------------------------------------------------------------
# Fixtures: temporary Lean files
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_putnam_dir(tmp_path: Path) -> Path:
    """Create a mock PutnamBench directory with sample files."""
    src = tmp_path / "lean4" / "src"
    src.mkdir(parents=True)

    # Simple theorem (no answer, no setup)
    (src / "putnam_1962_a1.lean").write_text(textwrap.dedent("""\
        import Mathlib

        open MeasureTheory

        /--
        Given five points in a plane, show something.
        -/
        theorem putnam_1962_a1
        (S : Set (ℝ × ℝ))
        (hS : S.ncard = 5)
        : True :=
        sorry
    """))

    # Answer-type problem with abbrev
    (src / "putnam_2023_a1.lean").write_text(textwrap.dedent("""\
        import Mathlib

        open Nat

        abbrev putnam_2023_a1_solution : ℕ := sorry
        -- 18
        /--
        Find the smallest n such that |f_n''(0)| > 2023.
        -/
        theorem putnam_2023_a1
          (f : ℕ → ℝ → ℝ)
          (hf : ∀ n > 0, f n = fun x : ℝ => x) :
          IsLeast {n | 0 < n} putnam_2023_a1_solution :=
        sorry
    """))

    # Problem with noncomputable abbrev
    (src / "putnam_2024_a1.lean").write_text(textwrap.dedent("""\
        import Mathlib

        noncomputable abbrev putnam_2024_a1_solution : Set ℕ := sorry
        --{1}
        /--
        Determine all positive integers n.
        -/
        theorem putnam_2024_a1 :
            {n : ℕ | 0 < n} = putnam_2024_a1_solution :=
          sorry
    """))

    # Problem with def helper
    (src / "putnam_1997_b5.lean").write_text(textwrap.dedent("""\
        import Mathlib

        open Nat

        def tetration : ℕ → ℕ → ℕ
          | _, 0 => 1
          | a, n + 1 => a ^ tetration a n

        /--
        Some problem about tetration.
        -/
        theorem putnam_1997_b5
            (n : ℕ) :
            tetration 2 n > 0 :=
          sorry
    """))

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseFile:
    def test_simple_theorem(self, tmp_putnam_dir: Path) -> None:
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_1962_a1.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        assert problem.id == "putnam_1962_a1"
        assert problem.year == 1962
        assert problem.section == "a"
        assert problem.problem_number == "a1"
        assert not problem.has_answer
        assert not problem.answer_is_sorry
        assert problem.answer_name is None
        assert "theorem putnam_1962_a1" in problem.statement
        assert "MeasureTheory" in problem.preamble
        assert "import Mathlib" in problem.full_lean_code
        assert problem.docstring is not None
        assert "five points" in problem.docstring

    def test_answer_problem(self, tmp_putnam_dir: Path) -> None:
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_2023_a1.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        assert problem.id == "putnam_2023_a1"
        assert problem.year == 2023
        assert problem.has_answer
        assert problem.answer_is_sorry  # answer is := sorry
        assert problem.answer_name == "putnam_2023_a1_solution"
        assert "putnam_2023_a1_solution" in problem.setup_block

    def test_noncomputable_abbrev(self, tmp_putnam_dir: Path) -> None:
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_2024_a1.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        assert problem.has_answer
        assert problem.answer_is_sorry  # noncomputable abbrev := sorry
        assert problem.answer_name == "putnam_2024_a1_solution"

    def test_def_helper(self, tmp_putnam_dir: Path) -> None:
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_1997_b5.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        assert problem.id == "putnam_1997_b5"
        assert problem.section == "b"
        assert "tetration" in problem.setup_block

    def test_full_lean_code_has_all_parts(self, tmp_putnam_dir: Path) -> None:
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_1997_b5.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        code = problem.full_lean_code
        assert "import Mathlib" in code
        assert "open Nat" in code
        assert "def tetration" in code
        assert "theorem putnam_1997_b5" in code
        assert "sorry" in code

    def test_source_property(self, tmp_putnam_dir: Path) -> None:
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_1962_a1.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        assert problem.source == "putnam"
        assert problem.split == "all"


class TestLoadProblems:
    def test_load_all(self, tmp_putnam_dir: Path) -> None:
        problems = load_putnam_problems(putnam_dir=tmp_putnam_dir)
        assert len(problems) == 4

    def test_filter_by_year(self, tmp_putnam_dir: Path) -> None:
        problems = load_putnam_problems(year_filter=2023, putnam_dir=tmp_putnam_dir)
        assert len(problems) == 1
        assert problems[0].id == "putnam_2023_a1"

    def test_filter_by_section(self, tmp_putnam_dir: Path) -> None:
        problems = load_putnam_problems(section_filter="b", putnam_dir=tmp_putnam_dir)
        assert len(problems) == 1
        assert problems[0].id == "putnam_1997_b5"

    def test_filter_by_ids(self, tmp_putnam_dir: Path) -> None:
        problems = load_putnam_problems(
            problem_ids=["putnam_1962_a1", "putnam_2024_a1"],
            putnam_dir=tmp_putnam_dir,
        )
        assert len(problems) == 2

    def test_filter_by_year_range(self, tmp_putnam_dir: Path) -> None:
        problems = load_putnam_problems(
            year_range=(2020, 2025), putnam_dir=tmp_putnam_dir
        )
        assert len(problems) == 2
        years = {p.year for p in problems}
        assert years == {2023, 2024}

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_putnam_problems(putnam_dir=tmp_path / "nonexistent")


class TestStats:
    def test_stats(self, tmp_putnam_dir: Path) -> None:
        problems = load_putnam_problems(putnam_dir=tmp_putnam_dir)
        stats = get_putnam_stats(problems)
        assert stats["total"] == 4
        assert stats["with_answer"] == 2
        assert stats["answer_sorry"] == 2
        assert stats["pure_theorem"] == 2
        assert "a" in stats["by_section"]
        assert "b" in stats["by_section"]
        assert stats["year_range"] == (1962, 2024)


class TestAnswerIsSorry:
    """Tests for answer_is_sorry detection."""

    def test_answer_sorry_detected(self, tmp_putnam_dir: Path) -> None:
        """Answer problems with := sorry should have answer_is_sorry=True."""
        problems = load_putnam_problems(putnam_dir=tmp_putnam_dir)
        answer_problems = [p for p in problems if p.has_answer]
        assert len(answer_problems) == 2
        for p in answer_problems:
            assert p.answer_is_sorry, f"{p.id} should have answer_is_sorry=True"

    def test_filled_answer_not_sorry(self, tmp_path: Path) -> None:
        """An answer problem with a concrete value should NOT be answer_is_sorry."""
        src = tmp_path / "lean4" / "src"
        src.mkdir(parents=True)
        (src / "putnam_2023_b2.lean").write_text(textwrap.dedent("""\
            import Mathlib

            abbrev putnam_2023_b2_solution : ℕ := 42

            theorem putnam_2023_b2 : putnam_2023_b2_solution = 42 :=
            sorry
        """))
        problem = _parse_lean_file(src / "putnam_2023_b2.lean")
        assert problem is not None
        assert problem.has_answer
        assert not problem.answer_is_sorry  # answer is 42, not sorry
        assert problem.answer_name == "putnam_2023_b2_solution"

    def test_answer_is_sorry_in_to_dict(self, tmp_putnam_dir: Path) -> None:
        """answer_is_sorry should appear in to_dict output."""
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_2023_a1.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        d = problem.to_dict()
        assert "answer_is_sorry" in d
        assert d["answer_is_sorry"] is True

    def test_non_answer_not_sorry(self, tmp_putnam_dir: Path) -> None:
        """Non-answer problems should have answer_is_sorry=False."""
        path = tmp_putnam_dir / "lean4" / "src" / "putnam_1962_a1.lean"
        problem = _parse_lean_file(path)
        assert problem is not None
        assert not problem.answer_is_sorry


class TestTacticFiltering:
    """Tests for the invalid tactic filter."""

    def test_clean_proof_passes(self) -> None:
        code = "theorem foo : True := by simp"
        assert _contains_invalid_tactic(code) is None

    def test_invalid_tactic_detected(self) -> None:
        code = "theorem foo : Nat := by exact Lean.defaultMaxRecDepth"
        result = _contains_invalid_tactic(code)
        assert result == "exact Lean.defaultMaxRecDepth"

    def test_each_invalid_tactic(self) -> None:
        for tactic in INVALID_TACTICS:
            code = f"theorem foo := by {tactic}"
            result = _contains_invalid_tactic(code)
            assert result == tactic, f"Failed to detect: {tactic}"

    def test_partial_match_not_triggered(self) -> None:
        # "exact Real.instAdd" should match, but "exact Real.instAddition" should too
        # because it contains "exact Real.instAdd" as a substring
        code = "theorem foo := by exact Real.add_comm"
        assert _contains_invalid_tactic(code) is None


class TestProblemResultSerialization:
    """Tests for ProblemResult.to_dict with new fields."""

    def test_verified_field(self) -> None:
        r = ProblemResult(
            problem_id="putnam_2024_a1",
            year=2024,
            section="a",
            solved=True,
            has_answer=False,
            verified=True,
            proof_code="theorem foo := by simp",
        )
        d = r.to_dict()
        assert d["verified"] is True
        assert d["has_answer"] is False
        assert "skipped" not in d  # not skipped

    def test_skipped_field(self) -> None:
        r = ProblemResult(
            problem_id="putnam_2023_a1",
            year=2023,
            section="a",
            solved=False,
            has_answer=True,
            skipped=True,
            skip_reason="answer_is_sorry",
        )
        d = r.to_dict()
        assert d["skipped"] is True
        assert d["skip_reason"] == "answer_is_sorry"


class TestBenchmarkResultSerialization:
    """Tests for PutnamBenchmarkResult.to_dict with new fields."""

    def test_new_fields_present(self) -> None:
        b = PutnamBenchmarkResult(
            total=10,
            solved=2,
            pass_rate=0.2,
            theorem_only_total=8,
            theorem_only_solved=2,
            answer_total=5,
            answer_skipped=3,
            verified_count=2,
            tactic_filtered_count=1,
        )
        d = b.to_dict()
        assert d["theorem_only_total"] == 8
        assert d["theorem_only_solved"] == 2
        assert d["theorem_only_pass_rate"] == 0.25
        assert d["answer_total"] == 5
        assert d["answer_skipped"] == 3
        assert d["verified_count"] == 2
        assert d["tactic_filtered_count"] == 1

    def test_zero_theorem_rate(self) -> None:
        b = PutnamBenchmarkResult(theorem_only_total=0, theorem_only_solved=0)
        d = b.to_dict()
        assert d["theorem_only_pass_rate"] == 0.0


class TestRealPutnamBench:
    """Tests against the actual PutnamBench checkout (skipped if not present)."""

    @pytest.fixture
    def putnam_dir(self) -> Path:
        d = DEFAULT_PUTNAM_DIR
        if not (d / "lean4" / "src").is_dir():
            pytest.skip("PutnamBench not cloned")
        return d

    def test_load_count(self, putnam_dir: Path) -> None:
        problems = load_putnam_problems(putnam_dir=putnam_dir)
        # PutnamBench has 672 Lean files
        assert len(problems) >= 650, f"Expected ~672, got {len(problems)}"

    def test_all_have_statements(self, putnam_dir: Path) -> None:
        problems = load_putnam_problems(putnam_dir=putnam_dir)
        for p in problems:
            assert p.statement, f"{p.id} has empty statement"
            assert "theorem" in p.statement, f"{p.id} statement missing 'theorem'"

    def test_year_range(self, putnam_dir: Path) -> None:
        problems = load_putnam_problems(putnam_dir=putnam_dir)
        years = {p.year for p in problems}
        assert min(years) == 1962
        assert max(years) >= 2024

    def test_stats_sanity(self, putnam_dir: Path) -> None:
        problems = load_putnam_problems(putnam_dir=putnam_dir)
        stats = get_putnam_stats(problems)
        assert stats["total"] >= 650
        assert stats["with_answer"] > 0
        assert stats["pure_theorem"] > 0
        assert stats["by_section"]["a"] > 0
        assert stats["by_section"]["b"] > 0

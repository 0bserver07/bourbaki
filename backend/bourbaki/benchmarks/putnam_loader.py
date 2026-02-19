"""PutnamBench problem loader -- parses Lean 4 theorem files into benchmark problems.

PutnamBench (https://github.com/trishullab/PutnamBench) contains 672 formalized
Putnam competition problems from years 1962-2025.  Each file declares a theorem
(and optionally an answer abbreviation) with ``sorry`` as a placeholder proof.

Unlike miniF2F, PutnamBench has:
- No train/test/valid split (all problems in one flat directory)
- ``import Mathlib`` (full) in every file
- Answer-type problems with ``abbrev putnam_YYYY_XN_solution``
- Helper ``def`` blocks that precede the theorem
- Year information embedded in the filename: putnam_YYYY_XN
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default location for PutnamBench checkout
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PUTNAM_DIR = _PROJECT_ROOT / ".bourbaki" / "putnam-bench"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Extract theorem declaration: matches ``theorem putnam_...`` up to ``sorry``
# Handles multiple forms:
#   theorem name ... := sorry
#   theorem name ... := by sorry
#   theorem name ... := by\n  sorry
#   theorem name... :=\n  sorry
#   theorem name:  (colon directly on name line, e.g. putnam_1993_b5)
_THEOREM_SIMPLE_RE = re.compile(
    r"(theorem\s+(putnam_\w+)[\s:].*?)(?:\s*:=\s*(?:by\s+)?sorry)",
    re.DOTALL,
)

# Extract imports
_IMPORT_RE = re.compile(r"^import\s+.+$", re.MULTILINE)

# Extract preamble lines (import, set_option, open, noncomputable, etc.)
_PREAMBLE_RE = re.compile(
    r"^(?:import|set_option|open|noncomputable|section|namespace)\s+.+$",
    re.MULTILINE,
)

# Extract everything *between* imports and the theorem (abbrevs, defs, etc.)
# These are "setup" lines that the REPL needs before the theorem.
_SETUP_BLOCK_RE = re.compile(
    r"^(?:abbrev|noncomputable\s+abbrev|def|noncomputable\s+def|instance|"
    r"variable|attribute|@\[|--\s|/--|section|namespace|end\s).+$",
    re.MULTILINE,
)

# Year from filename: putnam_YYYY_XN  -> YYYY
_YEAR_RE = re.compile(r"^putnam_(\d{4})_([ab]\d+)$")

# Section letter from problem id: putnam_YYYY_a3 -> "a"
_SECTION_RE = re.compile(r"^putnam_\d{4}_([ab])")


@dataclass
class PutnamProblem:
    """A single PutnamBench problem."""

    id: str  # e.g. "putnam_1962_a1"
    year: int  # e.g. 1962
    section: str  # "a" or "b"
    problem_number: str  # e.g. "a1", "b3"
    statement: str  # Full theorem statement (without := sorry)
    imports: list[str]  # Required Lean imports
    preamble: str  # open/set_option lines
    setup_block: str  # abbrev/def lines needed before the theorem
    file_path: str  # Absolute path to source file
    full_lean_code: str  # Complete code for verification (preamble + setup + theorem + sorry)
    has_answer: bool = False  # Whether the problem has an abbrev solution
    answer_is_sorry: bool = False  # True if the answer abbrev is still := sorry
    answer_name: str | None = None  # e.g. "putnam_2023_a1_solution"
    docstring: str | None = None  # Informal problem statement

    # Make it compatible with MiniF2FProblem interface
    @property
    def source(self) -> str:
        return "putnam"

    @property
    def split(self) -> str:
        return "all"  # PutnamBench has no split

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "year": self.year,
            "section": self.section,
            "problem_number": self.problem_number,
            "source": self.source,
            "split": self.split,
            "statement": self.statement,
            "imports": self.imports,
            "file_path": self.file_path,
            "has_answer": self.has_answer,
            "answer_is_sorry": self.answer_is_sorry,
            "answer_name": self.answer_name,
            "docstring": self.docstring,
        }


def load_putnam_problems(
    year_filter: int | None = None,
    section_filter: str | None = None,
    problem_ids: list[str] | None = None,
    putnam_dir: Path | None = None,
    year_range: tuple[int, int] | None = None,
) -> list[PutnamProblem]:
    """Load PutnamBench problems from the local checkout.

    Args:
        year_filter: Filter by specific year (e.g. 2023).
        section_filter: Filter by section ("a" or "b").
        problem_ids: Load only specific problem IDs.
        putnam_dir: Path to PutnamBench checkout (default: .bourbaki/putnam-bench).
        year_range: Filter by year range, inclusive (e.g. (2000, 2025)).

    Returns:
        List of PutnamProblem objects, sorted by year then problem number.
    """
    base = putnam_dir or DEFAULT_PUTNAM_DIR
    src_dir = base / "lean4" / "src"
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"PutnamBench not found at {src_dir}. "
            f"Clone it: git clone https://github.com/trishullab/PutnamBench "
            f"{base}"
        )

    problems: list[PutnamProblem] = []

    for lean_file in sorted(src_dir.glob("putnam_*.lean")):
        problem = _parse_lean_file(lean_file)
        if problem is None:
            continue

        # Apply filters
        if problem_ids and problem.id not in problem_ids:
            continue
        if year_filter is not None and problem.year != year_filter:
            continue
        if section_filter and problem.section != section_filter:
            continue
        if year_range is not None:
            lo, hi = year_range
            if not (lo <= problem.year <= hi):
                continue

        problems.append(problem)

    return problems


def _parse_lean_file(path: Path) -> PutnamProblem | None:
    """Parse a single PutnamBench Lean file into a problem."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    stem = path.stem  # e.g. "putnam_1962_a1"

    # Extract year and section
    year_match = _YEAR_RE.match(stem)
    if not year_match:
        return None
    year = int(year_match.group(1))
    problem_number = year_match.group(2)  # e.g. "a1"

    section_match = _SECTION_RE.match(stem)
    section = section_match.group(1) if section_match else "a"

    # Extract imports
    imports = _IMPORT_RE.findall(content)

    # Extract preamble (open, set_option directives - NOT imports, NOT abbrev/def)
    preamble_lines: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("open", "set_option")):
            preamble_lines.append(stripped)
        if stripped.startswith(("open scoped",)):
            # already captured by startswith("open") above
            pass

    preamble = "\n".join(preamble_lines)

    # Extract the setup block: everything between imports/open and the theorem
    # This includes abbrev, def, variable, etc.
    setup_lines: list[str] = []
    in_setup = False
    in_docstring = False
    in_multiline_def = False
    brace_depth = 0

    for line in content.split("\n"):
        stripped = line.strip()

        # Skip imports and open directives (already in preamble)
        if stripped.startswith(("import", "open", "set_option")):
            continue

        # Stop at the main theorem
        if stripped.startswith("theorem " + stem):
            break

        # Track docstrings (skip them for setup, they're metadata)
        if stripped.startswith("/-"):
            in_docstring = True
        if in_docstring:
            if "-/" in stripped:
                in_docstring = False
            continue

        # Track multi-line def/abbrev blocks
        if stripped.startswith(("abbrev", "noncomputable abbrev", "def", "noncomputable def")):
            in_multiline_def = True
            brace_depth = 0

        if in_multiline_def:
            setup_lines.append(line)
            brace_depth += line.count("{") - line.count("}")
            # A def/abbrev ends when we see sorry, or a blank line after balanced braces
            if "sorry" in stripped or (brace_depth <= 0 and stripped == ""):
                in_multiline_def = False
            continue

        # Also grab standalone comment lines with answer hints (-- 18)
        if stripped.startswith("--") and setup_lines:
            setup_lines.append(line)
            continue

        # Skip blank lines
        if not stripped:
            continue

        # Any other non-empty, non-comment line that's not theorem
        # (e.g. attribute, variable, instance, etc.)
        if not stripped.startswith(("theorem", "lemma")):
            setup_lines.append(line)

    setup_block = "\n".join(setup_lines).strip()

    # Extract docstring
    docstring = None
    ds_match = re.search(r"/--\s*(.*?)\s*-/", content, re.DOTALL)
    if ds_match:
        docstring = ds_match.group(1).strip()

    # Detect answer abbreviations
    has_answer = False
    answer_is_sorry = False
    answer_name = None
    abbrev_match = re.search(
        r"(?:noncomputable\s+)?abbrev\s+(putnam_\w+_solution)\b", content
    )
    if abbrev_match:
        has_answer = True
        answer_name = abbrev_match.group(1)
        # Check if the answer is still := sorry (unfilled placeholder)
        # The abbrev line looks like:
        #   abbrev putnam_XXXX_solution : Type := sorry
        #   noncomputable abbrev putnam_XXXX_solution : Type := sorry
        # We need to match past the type annotation (which contains `:`)
        answer_sorry_re = re.search(
            r"(?:noncomputable\s+)?abbrev\s+" + re.escape(answer_name) + r"\b.*?:=\s*sorry",
            content,
        )
        answer_is_sorry = answer_sorry_re is not None

    # Extract theorem statement
    m = _THEOREM_SIMPLE_RE.search(content)
    if not m:
        return None
    statement = m.group(1).strip()

    # Build the full Lean code for verification
    # Include: imports + preamble + setup_block + theorem + sorry
    parts: list[str] = []
    parts.append("import Mathlib")
    if preamble:
        parts.append(preamble)
    if setup_block:
        parts.append(setup_block)
    parts.append(f"{statement} :=\n  sorry")

    full_code = "\n\n".join(parts)

    return PutnamProblem(
        id=stem,
        year=year,
        section=section,
        problem_number=problem_number,
        statement=statement,
        imports=imports,
        preamble=preamble,
        setup_block=setup_block,
        file_path=str(path),
        full_lean_code=full_code,
        has_answer=has_answer,
        answer_is_sorry=answer_is_sorry,
        answer_name=answer_name,
        docstring=docstring,
    )


def get_putnam_stats(problems: list[PutnamProblem]) -> dict[str, Any]:
    """Get summary statistics for a set of Putnam problems."""
    by_year: dict[int, int] = {}
    by_section: dict[str, int] = {}
    answer_count = 0
    answer_sorry_count = 0

    for p in problems:
        by_year[p.year] = by_year.get(p.year, 0) + 1
        by_section[p.section] = by_section.get(p.section, 0) + 1
        if p.has_answer:
            answer_count += 1
        if p.answer_is_sorry:
            answer_sorry_count += 1

    # Decade breakdown
    by_decade: dict[str, int] = {}
    for y, c in by_year.items():
        decade = f"{(y // 10) * 10}s"
        by_decade[decade] = by_decade.get(decade, 0) + c

    return {
        "total": len(problems),
        "by_year": dict(sorted(by_year.items())),
        "by_decade": dict(sorted(by_decade.items())),
        "by_section": dict(sorted(by_section.items())),
        "with_answer": answer_count,
        "answer_sorry": answer_sorry_count,
        "pure_theorem": len(problems) - answer_count,
        "year_range": (
            (min(by_year.keys()), max(by_year.keys())) if by_year else (0, 0)
        ),
    }

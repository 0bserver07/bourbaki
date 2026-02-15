"""miniF2F problem loader — parses Lean 4 theorem files into benchmark problems.

miniF2F (https://github.com/yangky11/miniF2F-lean4) contains 488 formalized
math problems split into valid (244) and test (244). Each file declares a
theorem with `sorry` as a placeholder proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default location for miniF2F checkout
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MINIF2F_DIR = _PROJECT_ROOT / ".bourbaki" / "miniF2F-lean4"

# Pattern to extract theorem declarations
# Matches: theorem name ... := by sorry  OR  theorem name ... := sorry
_THEOREM_RE = re.compile(
    r"^(theorem\s+\w+.*?)(?::=\s*by\s+sorry|:=\s*sorry)\s*$",
    re.MULTILINE | re.DOTALL,
)

# Simpler pattern: find the theorem line and capture everything up to sorry
_THEOREM_SIMPLE_RE = re.compile(
    r"(theorem\s+(\w+)\s.*?)(?:\s*:=\s*(?:by\s+)?sorry)",
    re.DOTALL,
)

# Extract imports from the file
_IMPORT_RE = re.compile(r"^import\s+.+$", re.MULTILINE)

# Extract preamble lines (import, set_option, open, etc.)
_PREAMBLE_RE = re.compile(r"^(?:import|set_option|open)\s+.+$", re.MULTILINE)

# Extract the problem source from the filename
# e.g., "aime_1983_p1" → "aime", "imo_1964_p4" → "imo"
_SOURCE_RE = re.compile(r"^([a-z]+)_")


@dataclass
class MiniF2FProblem:
    """A single miniF2F benchmark problem."""
    id: str                    # e.g., "aime_1983_p1"
    source: str                # e.g., "aime", "imo", "amc", "mathd"
    split: str                 # "valid" or "test"
    statement: str             # Full theorem statement (without sorry)
    imports: list[str]         # Required Lean imports
    file_path: str             # Path to the source file
    full_lean_code: str        # Complete file content for verification

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "split": self.split,
            "statement": self.statement,
            "imports": self.imports,
            "file_path": self.file_path,
        }


def load_minif2f_problems(
    split: str = "valid",
    source_filter: str | None = None,
    problem_ids: list[str] | None = None,
    minif2f_dir: Path | None = None,
) -> list[MiniF2FProblem]:
    """Load miniF2F problems from the local checkout.

    Args:
        split: "valid" or "test" (or "all" for both).
        source_filter: Filter by source (e.g., "aime", "imo", "amc").
        problem_ids: Load only specific problem IDs.
        minif2f_dir: Path to miniF2F-lean4 checkout (default: .bourbaki/miniF2F-lean4).

    Returns:
        List of MiniF2FProblem objects.
    """
    base = minif2f_dir or DEFAULT_MINIF2F_DIR
    if not base.is_dir():
        raise FileNotFoundError(
            f"miniF2F not found at {base}. "
            f"Clone it: git clone https://github.com/yangky11/miniF2F-lean4 {base}"
        )

    problems: list[MiniF2FProblem] = []
    splits = ["valid", "test"] if split == "all" else [split]

    for s in splits:
        # Try common directory naming conventions
        split_name = s.capitalize()  # "Valid" or "Test"
        candidates = [
            base / "MiniF2F" / split_name,
            base / "Minif2f" / split_name,
            base / "minif2f" / s,
            base / split_name,
            base / s,
        ]
        split_dir = None
        for candidate in candidates:
            if candidate.is_dir():
                split_dir = candidate
                break
        if split_dir is None:
            continue

        for lean_file in sorted(split_dir.glob("*.lean")):
            problem = _parse_lean_file(lean_file, s)
            if problem is None:
                continue

            # Apply filters
            if problem_ids and problem.id not in problem_ids:
                continue
            if source_filter and problem.source != source_filter:
                continue

            problems.append(problem)

    return problems


def _parse_lean_file(path: Path, split: str) -> MiniF2FProblem | None:
    """Parse a single miniF2F Lean file into a problem."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Extract imports
    imports = _IMPORT_RE.findall(content)

    # Extract full preamble (imports + set_option + open)
    preamble_lines = _PREAMBLE_RE.findall(content)

    # Extract theorem statement
    m = _THEOREM_SIMPLE_RE.search(content)
    if not m:
        return None

    statement = m.group(1).strip()

    # Determine source from filename
    stem = path.stem
    source_match = _SOURCE_RE.match(stem)
    source = source_match.group(1) if source_match else "unknown"

    # Build the full Lean code for verification — include all preamble directives
    # (imports, set_option maxHeartbeats, open namespaces) so the proof environment
    # matches what miniF2F expects
    preamble_block = "\n".join(preamble_lines)
    full_code = f"{preamble_block}\n\n{statement} := by\n  sorry"

    return MiniF2FProblem(
        id=stem,
        source=source,
        split=split,
        statement=statement,
        imports=imports,
        file_path=str(path),
        full_lean_code=full_code,
    )


def get_problem_stats(problems: list[MiniF2FProblem]) -> dict[str, Any]:
    """Get summary statistics for a set of problems."""
    sources: dict[str, int] = {}
    for p in problems:
        sources[p.source] = sources.get(p.source, 0) + 1

    return {
        "total": len(problems),
        "by_source": dict(sorted(sources.items(), key=lambda x: -x[1])),
        "splits": {
            "valid": sum(1 for p in problems if p.split == "valid"),
            "test": sum(1 for p in problems if p.split == "test"),
        },
    }

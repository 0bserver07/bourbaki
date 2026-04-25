"""Build standalone Lean 4 source from a theorem stub plus a tactic sequence.

Extracted from ``autonomous.search_tree.ProofSearchTree._build_proof_code`` so
the same assembly logic can be shared by the new prover loop, the legacy
search tree, and any future verification-side caller.

The REPL session has ``import Mathlib`` loaded implicitly, but standalone
verification via ``lean_prover`` needs the full file. This helper guarantees:

- ``import Mathlib`` is prepended (unless the theorem text already imports).
- Any ``open`` / ``set_option`` / ``noncomputable section`` lines that
  appear before the theorem keyword are preserved as a preamble.
- The theorem declaration is followed by ``:= by`` and the indented tactic
  block.
"""

from __future__ import annotations


_THEOREM_KEYWORDS = (
    "theorem ",
    "lemma ",
    "noncomputable theorem ",
    "noncomputable lemma ",
)


def build_proof_code(theorem: str, tactics: list[str]) -> str:
    """Assemble a complete Lean 4 file from ``theorem`` and ``tactics``."""

    tactic_block = "\n  ".join(tactics)

    lines = theorem.split("\n")
    preamble_lines: list[str] = []
    theorem_lines: list[str] = []
    found_theorem = False
    has_import = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import "):
            has_import = True
        if not found_theorem and not stripped.startswith(_THEOREM_KEYWORDS):
            preamble_lines.append(line)
        else:
            found_theorem = True
            theorem_lines.append(line)

    # If nothing looked like a theorem keyword, treat the whole thing as
    # the theorem (backward compat with plain ``theorem foo : T`` input).
    if not theorem_lines:
        theorem_lines = lines
        preamble_lines = []

    theorem_decl = "\n".join(theorem_lines).rstrip()

    parts: list[str] = []

    if not has_import:
        parts.append("import Mathlib")

    preamble = "\n".join(preamble_lines).strip()
    if preamble:
        parts.append(preamble)

    parts.append(f"{theorem_decl} := by\n  {tactic_block}")

    return "\n\n".join(parts)

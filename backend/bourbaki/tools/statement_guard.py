"""Anti-trivialization guard for Lean 4 theorem statements.

A formalization that typechecks, contains no ``sorry``, and preserves its own
signature is approved by the reviewer today regardless of whether it states the
original problem — the reviewer's ``check_1`` compares Lean against Lean, never
against the informal statement (``prover/prompts.py``, ``prover/reviewer.py``).

So ``theorem foo : True := trivial`` passes every gate the loop has.

This module is the cheap deterministic half of that fix: a syntactic filter that
runs before any LLM call and rejects statements that are vacuous by construction.
It makes no judgement about faithfulness — that is the fidelity grader's job. It
only catches targets that could not state *any* problem.

Deliberately conservative: a false reject here silently lowers a benchmark pass
rate, which is exactly the failure mode this work exists to prevent. When the
target cannot be parsed, the verdict is "not trivial" and the statement passes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = [
    "TrivialFinding",
    "TrivializationVerdict",
    "check_trivialization",
]

# Targets that assert nothing. `False` is included because a theorem whose
# target is `False` is either unprovable or evidence of an inconsistent
# axiom set — in neither case is it a formalization of the problem.
_VACUOUS_TARGETS = frozenset({"True", "False"})

# Proof bodies that close a goal without doing mathematical work. Only
# meaningful in combination with a vacuous target; a real statement closed by
# `rfl` is unusual but legitimate, so these never reject on their own.
_TRIVIAL_BODIES = frozenset({"trivial", "rfl", "by trivial", "by rfl"})

_DECL_RE = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|nonrec\s+)*(theorem|lemma|example)\b")

# Bracket pairs that can nest inside a declaration header. The target type is
# the first `:` that appears at depth zero, so binders like `(h : n > 0)` and
# instance binders like `[Fact p.Prime]` must not be mistaken for it.
_OPENERS = "([{⟨⦃"
_CLOSERS = ")]}⟩⦄"


@dataclass
class TrivialFinding:
    """One declaration judged vacuous, with the evidence that judged it."""

    decl_name: str
    reason: str
    target: str
    line: int


@dataclass
class TrivializationVerdict:
    """Result of scanning a Lean source fragment.

    ``is_trivial`` is the blocking signal. ``findings`` explains why, so a
    rejection can be reported rather than being a silent filter.
    """

    is_trivial: bool
    findings: list[TrivialFinding] = field(default_factory=list)

    def summary(self) -> str:
        """One-line reason suitable for feedback or a log."""
        if not self.findings:
            return "no vacuous statement detected"
        return "; ".join(
            f"{f.decl_name or '<anonymous>'} (line {f.line}): {f.reason}"
            for f in self.findings
        )


def _strip_comments(code: str) -> str:
    """Remove ``--`` line comments and ``/- -/`` block comments.

    Block comments nest in Lean 4, so this counts depth rather than regexing.
    Characters are replaced with spaces instead of deleted so that line numbers
    and offsets stay aligned with the original source.
    """
    out: list[str] = []
    depth = 0
    i = 0
    n = len(code)
    while i < n:
        two = code[i : i + 2]
        if depth == 0 and two == "--":
            while i < n and code[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if two == "/-":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if two == "-/" and depth > 0:
            depth -= 1
            out.append("  ")
            i += 2
            continue
        out.append(" " if depth > 0 and code[i] != "\n" else code[i])
        i += 1
    return "".join(out)


def _split_target(header: str) -> tuple[str, str] | None:
    """Split a declaration into (target_type, body).

    ``header`` starts at the ``theorem`` keyword. Returns ``None`` when no
    depth-zero ``:`` is found, which means the declaration is malformed or uses
    a form this parser does not model — in that case the caller passes it.
    """
    depth = 0
    target_start: int | None = None
    i = 0
    n = len(header)
    while i < n:
        ch = header[i]
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth = max(0, depth - 1)
        elif depth == 0:
            # `:=` is assignment, not a type ascription — check it first so a
            # value-style declaration does not read as a target.
            if header.startswith(":=", i):
                if target_start is None:
                    return None
                return header[target_start:i], header[i + 2 :]
            if ch == ":":
                if target_start is None:
                    target_start = i + 1
        i += 1
    if target_start is None:
        return None
    return header[target_start:], ""


def _normalize_target(target: str) -> str:
    """Strip whitespace and redundant outer parentheses from a target type."""
    t = " ".join(target.split())
    # `((True))` is still vacuous. Only peel when the outer parens actually
    # wrap the whole expression.
    while t.startswith("(") and t.endswith(")"):
        depth = 0
        wraps = True
        for idx, ch in enumerate(t):
            if ch in _OPENERS:
                depth += 1
            elif ch in _CLOSERS:
                depth -= 1
                if depth == 0 and idx != len(t) - 1:
                    wraps = False
                    break
        if not wraps:
            break
        t = t[1:-1].strip()
    return t


def _decl_name(header: str) -> str:
    """Extract the declaration name, or "" for anonymous forms like `example`."""
    m = re.match(
        r"\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|nonrec\s+)*"
        r"(?:theorem|lemma|example)\s+([^\s({\[:]+)",
        header,
    )
    return m.group(1) if m else ""


def _iter_declarations(code: str) -> list[tuple[int, str]]:
    """Yield (line_number, source) for each top-level theorem-like declaration."""
    lines = code.splitlines()
    starts: list[int] = [i for i, ln in enumerate(lines) if _DECL_RE.match(ln)]
    decls: list[tuple[int, str]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        decls.append((start + 1, "\n".join(lines[start:end])))
    return decls


def check_trivialization(code: str) -> TrivializationVerdict:
    """Scan Lean source for statements that assert nothing.

    Rejects a declaration when its target type is ``True`` or ``False``. A
    trivial proof body (``trivial``, ``rfl``) is recorded as supporting evidence
    but never rejects on its own — real statements are occasionally closed by
    ``rfl``, and a false reject is worse than a miss.

    Unparseable input passes. This guard exists to catch an obvious degenerate
    case cheaply, not to be a Lean parser.
    """
    if not code or not code.strip():
        return TrivializationVerdict(is_trivial=False)

    stripped = _strip_comments(code)
    findings: list[TrivialFinding] = []

    for line_no, decl in _iter_declarations(stripped):
        split = _split_target(decl)
        if split is None:
            continue
        raw_target, body = split
        target = _normalize_target(raw_target)
        if target not in _VACUOUS_TARGETS:
            continue

        normalized_body = " ".join(body.split())
        reason = f"target type is `{target}`, which states nothing"
        if normalized_body in _TRIVIAL_BODIES:
            reason += f" (closed by `{normalized_body}`)"

        findings.append(
            TrivialFinding(
                decl_name=_decl_name(decl),
                reason=reason,
                target=target,
                line=line_no,
            )
        )

    if findings:
        logger.warning(
            "statement_guard: rejected %d vacuous declaration(s): %s",
            len(findings),
            "; ".join(f.decl_name or "<anonymous>" for f in findings),
        )

    return TrivializationVerdict(is_trivial=bool(findings), findings=findings)

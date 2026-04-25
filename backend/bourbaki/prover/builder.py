"""Builder node for the proposer-builder-reviewer loop.

The builder takes a :class:`ProposalMessage` produced by the proposer and
runs it through the warm :class:`LeanREPLSession` (Mathlib pre-loaded). It
inspects the REPL's response and emits a typed :class:`FeedbackMessage`:

- ``missing_target_theorem`` — proposal does not declare the target theorem
  (terminal; loop ends).
- ``build_failed`` — REPL reported one or more ``severity == "error"``
  messages.
- ``sorries_goal_state`` — proposal compiles but contains ``sorry``s; we
  expose the goals the REPL printed at each sorry location.
- ``axiom_detected`` — proposal contains forbidden ``axiom`` declarations.
- ``search_tactics_detected`` — proposal contains banned ``apply?`` /
  ``exact?`` / blocklisted tactics.
- ``build_success`` — clean compile, no sorries, no banned tactics.

This mirrors ax-prover's ``_builder_node`` but uses the warm REPL instead
of ``lake build`` (saving ~30-90s per iteration). Whole-file verification
via ``lean_prover`` is deferred to the reviewer/verifier stage.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bourbaki.autonomous.tactics import contains_blocklisted_tactic
from bourbaki.prover import feedback
from bourbaki.prover.state import FeedbackMessage, ProverState
from bourbaki.tools.lean_repl import LeanREPLSession

logger = logging.getLogger(__name__)


# Matches a ``theorem`` or ``lemma`` declaration and captures its name.
_DECL_RE = re.compile(r"\b(?:theorem|lemma)\s+(\S+)")

# Matches an ``import`` line (REPL has Mathlib pre-loaded so we strip these).
_IMPORT_LINE_RE = re.compile(r"^\s*import\s+\S+.*$", re.MULTILINE)


async def run_builder(
    state: ProverState, session: LeanREPLSession
) -> FeedbackMessage:
    """Run the builder node on ``state.last_proposal``.

    Returns a :class:`FeedbackMessage` describing the build outcome. The
    caller is responsible for routing on the resulting feedback (e.g.
    ``is_terminal`` ends the loop; ``is_success`` advances to the reviewer).
    """
    proposal = state.last_proposal
    if proposal is None:
        raise ValueError("run_builder expects state.last_proposal to be set")

    code = proposal.code
    target_name = state.problem_id

    # 1) Missing target theorem? (terminal)
    declared = _find_declared_names(code)
    if target_name and target_name not in declared:
        logger.warning(
            "builder: proposed code does not declare target theorem '%s' "
            "(found: %s)",
            target_name,
            sorted(declared),
        )
        return feedback.missing_target_theorem(target_name)

    # 2) Strip imports (REPL has Mathlib pre-loaded).
    stripped_code = _IMPORT_LINE_RE.sub("", code)

    # Build the REPL command. Prepend preamble if non-empty.
    if state.preamble:
        cmd = f"{state.preamble.rstrip()}\n\n{stripped_code.lstrip()}"
    else:
        cmd = stripped_code

    # 3) Send to REPL.
    logger.debug("builder: sending %d chars to REPL", len(cmd))
    result: dict[str, Any] = await session.send_cmd(cmd)

    # 4) Inspect result.

    # 4a) REPL-level error (timeout, pipe corruption). The send_cmd shape
    # is ``{"error": "..."}`` with no ``messages`` / ``sorries`` keys.
    if "error" in result and "messages" not in result and "sorries" not in result:
        return feedback.build_failed(str(result.get("error", "unknown REPL error")))

    # 4b) Compiler errors (severity == "error" in messages).
    messages = result.get("messages") or []
    error_messages = [
        m for m in messages
        if isinstance(m, dict) and m.get("severity") == "error"
    ]
    if error_messages:
        rendered = _render_repl_errors(error_messages)
        return feedback.build_failed(rendered)

    # 4c) Remaining sorries — extract goal states.
    sorries = result.get("sorries") or []
    if sorries:
        goal_states = _extract_sorry_goals(sorries)
        return feedback.sorries_goal_state(len(sorries), goal_states)

    # 4d) Forbidden ``axiom`` declarations (scan post-strip code).
    axiom_count, axiom_locations = _count_pattern(stripped_code, r"\baxiom\b")
    if axiom_count:
        return feedback.axiom_detected(axiom_count, axiom_locations)

    # 4e) Banned search tactics (apply? / exact?).
    search_count, search_locations = _count_pattern(
        stripped_code, r"\b(?:apply|exact)\?"
    )
    if search_count:
        return feedback.search_tactics_detected(search_count, search_locations)

    # 4f) Tactic blocklist (REPL false positives — see autonomous/tactics.py).
    blocked = contains_blocklisted_tactic(stripped_code)
    if blocked is not None:
        # Surface via search_tactics_detected — the prompt copy is correct
        # ("banned at build time, replace with explicit lemma names") and
        # we don't have a dedicated "blocklisted_tactic" feedback factory.
        bcount, blocations = _count_pattern(
            stripped_code, re.escape(blocked)
        )
        # _count_pattern can return 0 if the blocklisted token contains
        # whitespace that doesn't match \b; fall back to a synthetic
        # location string in that case.
        if bcount == 0:
            blocations = f"blocklisted tactic: {blocked}"
            bcount = 1
        return feedback.search_tactics_detected(bcount, blocations)

    # 5) Clean build.
    return feedback.build_success()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_declared_names(code: str) -> set[str]:
    """Return the set of theorem/lemma names declared in ``code``."""
    names: set[str] = set()
    for match in _DECL_RE.finditer(code):
        # Strip trailing punctuation like ``:`` or ``(`` that may have been
        # captured by ``\S+`` (the regex captures up to the next whitespace).
        name = match.group(1)
        # Lean theorem names can include ``.`` (namespaced). Trim anything
        # past the first non-identifier character.
        trimmed = re.match(r"[A-Za-z_][A-Za-z0-9_'.]*", name)
        if trimmed:
            names.add(trimmed.group(0))
    return names


def _render_repl_errors(messages: list[dict[str, Any]]) -> str:
    """Concatenate the ``data`` field of each error message."""
    parts: list[str] = []
    for m in messages:
        data = m.get("data")
        if data is None:
            continue
        pos = m.get("pos")
        if isinstance(pos, dict) and "line" in pos:
            line = pos.get("line")
            col = pos.get("column")
            header = f"line {line}, column {col}: " if col is not None else f"line {line}: "
            parts.append(f"{header}{data}")
        else:
            parts.append(str(data))
    return "\n\n".join(parts) if parts else "(no error data)"


def _extract_sorry_goals(sorries: list[Any]) -> list[str]:
    """Pull the goal-state string out of each sorry entry.

    The lean4-repl shape uses ``goal`` (singular string) for the response
    to a fresh ``cmd`` containing sorries, but some versions return
    ``goals`` (list) — handle both.
    """
    goals: list[str] = []
    for s in sorries:
        if not isinstance(s, dict):
            goals.append(str(s))
            continue
        if "goal" in s and s["goal"]:
            goals.append(str(s["goal"]))
        elif "goals" in s and s["goals"]:
            inner = s["goals"]
            if isinstance(inner, list):
                goals.append("\n".join(str(g) for g in inner))
            else:
                goals.append(str(inner))
        else:
            goals.append("(no goal information returned)")
    return goals


def _count_pattern(code: str, pattern: str) -> tuple[int, str]:
    """Count regex matches in ``code`` and return ``(count, locations)``.

    ``locations`` is a comma-separated string of ``"line N"`` entries
    suitable for inclusion in feedback content.
    """
    try:
        compiled = re.compile(pattern)
    except re.error:
        return 0, ""

    line_numbers: list[int] = []
    for i, line in enumerate(code.splitlines(), start=1):
        for _ in compiled.finditer(line):
            line_numbers.append(i)
    if not line_numbers:
        return 0, ""
    locations = ", ".join(f"line {n}" for n in line_numbers)
    return len(line_numbers), locations

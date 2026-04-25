"""Typed feedback factories for the proposer-builder-reviewer loop.

Each factory returns a :class:`FeedbackMessage` with a `kind` string that
matches its name and a rendered `content` block formatted for the next
proposer call. Kinds map directly to ax-prover's feedback classes
(`BuildSuccess`, `BuildFailed`, `SorriesGoalState`, …) but we render them
into a single text payload since GLM-5.1 reads plain text far better than
nested JSON.
"""

from __future__ import annotations

from bourbaki.prover.state import FeedbackMessage


def build_success() -> FeedbackMessage:
    return FeedbackMessage(
        kind="build_success",
        content="<feedback>The proposed proof builds successfully with no remaining sorries and no banned tactics.</feedback>",
        is_success=True,
    )


def build_failed(error_output: str, max_tokens: int = 4000) -> FeedbackMessage:
    """Render compiler errors. Truncates the middle of long outputs."""

    truncated = _truncate_middle(error_output, max_tokens)
    content = (
        "<feedback>\n"
        "The proposed proof did not build. Compiler output:\n\n"
        f"```\n{truncated}\n```\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="build_failed", content=content)


def sorries_goal_state(sorry_count: int, goal_states: list[str]) -> FeedbackMessage:
    rendered_goals = "\n\n".join(
        f"Goal {i + 1}:\n{goal}" for i, goal in enumerate(goal_states)
    )
    content = (
        "<feedback>\n"
        f"The proposed proof has {sorry_count} remaining `sorry` placeholder(s). "
        "The compiler reports the following goal state(s) at each location:\n\n"
        f"{rendered_goals}\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="sorries_goal_state", content=content)


def axiom_detected(count: int, locations: str) -> FeedbackMessage:
    content = (
        "<feedback>\n"
        f"The proposed proof contains {count} `axiom` declaration(s) at: {locations}. "
        "Axioms are forbidden — the proof must be derived from Mathlib alone.\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="axiom_detected", content=content)


def search_tactics_detected(count: int, locations: str) -> FeedbackMessage:
    content = (
        "<feedback>\n"
        f"The proposed proof uses `apply?` or `exact?` at {count} location(s): {locations}. "
        "Mathlib search tactics are banned at build time. Replace them with explicit lemma names.\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="search_tactics_detected", content=content)


def missing_target_theorem(name: str) -> FeedbackMessage:
    """The proposer typoed the theorem name (or omitted it). The loop should
    let the proposer try again with this feedback in scope so it can correct
    the name — terminal would burn the budget on a single character mistake.
    """
    content = (
        "<feedback>\n"
        f"The proposed code does not declare the target theorem `{name}`. "
        "Make sure your output includes the full theorem declaration with the original signature, "
        "spelled exactly. Re-emit with the correct name on the next iteration.\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="missing_target_theorem", content=content)


def review_approved(comments: str = "") -> FeedbackMessage:
    content = "<feedback>The reviewer approved the proof."
    if comments:
        content += f" Reviewer notes: {comments}"
    content += "</feedback>"
    return FeedbackMessage(kind="review_approved", content=content, is_success=True)


def review_rejected(feedback: str) -> FeedbackMessage:
    content = (
        "<feedback>\n"
        "The reviewer rejected the proposed proof. Reasons:\n"
        f"{feedback}\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="review_rejected", content=content)


def max_iterations(n: int) -> FeedbackMessage:
    content = (
        "<feedback>\n"
        f"Maximum iteration budget ({n}) reached without a verified proof.\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="max_iterations", content=content, is_terminal=True)


def structured_output_parsing_failed(error_message: str) -> FeedbackMessage:
    content = (
        "<feedback>\n"
        "The structured output failed to parse. Re-emit JSON with exactly the four "
        "fields `reasoning`, `imports`, `opens`, `updated_theorem`.\n"
        f"Parser error: {error_message}\n"
        "</feedback>"
    )
    return FeedbackMessage(kind="structured_output_parsing_failed", content=content)


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n\n... [truncated {len(text) - max_chars} characters] ...\n\n{text[-tail:]}"

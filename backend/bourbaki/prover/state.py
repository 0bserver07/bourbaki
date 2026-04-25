"""Pydantic state models for the proposer-builder-reviewer loop.

Mirrors ax-prover's ProverAgentState but flattened for a plain async loop
(no LangGraph). All structured outputs the LLM emits are validated through
these models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProposalMessage(BaseModel):
    """A single proposer output: reasoning + replacement theorem code."""

    reasoning: str
    imports: list[str] = Field(default_factory=list)
    opens: list[str] = Field(default_factory=list)
    code: str
    iteration: int


class FeedbackMessage(BaseModel):
    """Feedback returned by builder / reviewer / memory edges.

    `kind` matches the factory name in `feedback.py`. `is_terminal` ends
    the loop without further proposer calls.
    """

    kind: str
    content: str
    is_success: bool = False
    is_terminal: bool = False


class ProverResult(BaseModel):
    """Structured-output schema enforced on GLM-5.1 during the proposer call.

    Kept deliberately narrow: GLM-5.1's JSON mode is less robust than Claude's,
    so the surface is just four fields.
    """

    reasoning: str
    imports: list[str] = Field(default_factory=list)
    opens: list[str] = Field(default_factory=list)
    updated_theorem: str


class ReviewDecision(BaseModel):
    """Reviewer structured output. Approval is derived from `check_1 AND check_2`;
    `check_3` and `approved` are honeypots — ignored by the caller.
    """

    reasoning: str
    check_1: bool
    check_2: bool
    check_3: bool
    approved: bool


class ProverState(BaseModel):
    """Mutable loop state threaded through proposer / builder / reviewer / memory."""

    problem_id: str
    target_theorem: str
    preamble: str = ""
    full_file: str = ""
    iteration: int = 0
    max_iterations: int = 50
    experience: str = ""
    last_proposal: ProposalMessage | None = None
    last_feedback: FeedbackMessage | None = None
    messages: list[ProposalMessage | FeedbackMessage] = Field(default_factory=list)
    approved: bool = False
    verified: bool = False
    final_proof_code: str | None = None

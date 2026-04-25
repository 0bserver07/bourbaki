"""Memory strategies for the proposer-builder-reviewer loop.

Three policies, mirroring ax-prover:

- :class:`MemorylessMemory` — forget everything, return empty experience.
- :class:`PreviousKMemory` — render the last K (proposal, feedback) pairs verbatim.
- :class:`ExperienceMemory` — call GLM-5.1 once per retry to compress prior lessons
  into an `<experience>...</experience>` block.

All implementations are stubs at this stage (Phase 1 scaffold).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bourbaki.prover.state import ProverState


class BaseMemory(ABC):
    """Strategy that turns recent loop history into a fresh experience string."""

    @abstractmethod
    async def process(self, state: ProverState) -> str:
        """Return the new value for `state.experience`."""


class MemorylessMemory(BaseMemory):
    """Returns an empty experience string. Default for the first cut."""

    async def process(self, state: ProverState) -> str:
        raise NotImplementedError("Phase 2 will implement MemorylessMemory.process")


class PreviousKMemory(BaseMemory):
    """Renders the last K (ProposalMessage, FeedbackMessage) pairs verbatim."""

    def __init__(self, k: int = 2) -> None:
        self.k = k

    async def process(self, state: ProverState) -> str:
        raise NotImplementedError("Phase 2 will implement PreviousKMemory.process")


class ExperienceMemory(BaseMemory):
    """LLM-summarized experience block, regenerated each iteration."""

    def __init__(self, model: str = "glm:glm-5.1") -> None:
        self.model = model

    async def process(self, state: ProverState) -> str:
        raise NotImplementedError("Phase 2 will implement ExperienceMemory.process")

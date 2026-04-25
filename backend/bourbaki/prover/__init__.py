"""Proposer-Builder-Reviewer-Memory loop for Lean 4 proof generation.

Replaces the HILBERT-style sketch/formalize/decompose pipeline with a single
iterative loop driven by Pydantic AI + GLM-5.1 and a warm LeanREPLSession.

See `.bourbaki/plans/proposer-builder-loop.md` for the design doc.
"""

from bourbaki.prover.prover import ProverConfig, ProverLoop
from bourbaki.prover.state import ProverResult

__all__ = ["ProverLoop", "ProverConfig", "ProverResult"]

"""Proposer-Builder-Reviewer-Memory loop for Lean 4 proof generation.

Replaces the HILBERT-style sketch/formalize/decompose pipeline with a single
iterative loop driven by Pydantic AI + GLM-5.1 and a warm LeanREPLSession.

See `.bourbaki/plans/proposer-builder-loop.md` for the design doc.
"""

# z.ai compatibility shim — applies an args_as_dict workaround at first
# import so z.ai's Anthropic-compat endpoint survives pydantic_ai's
# retry message re-mapping. See issue #13 and _pydantic_ai_compat.py.
from bourbaki.prover import _pydantic_ai_compat  # noqa: F401

from bourbaki.prover.prover import ProverConfig, ProverLoop
from bourbaki.prover.state import ProverResult

__all__ = ["ProverLoop", "ProverConfig", "ProverResult"]

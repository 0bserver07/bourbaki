"""Tool registry — builds Pydantic AI FunctionToolset from all tools."""

from __future__ import annotations

from pydantic_ai import FunctionToolset

from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.paper_search import paper_search
from bourbaki.tools.sequence_lookup import sequence_lookup
from bourbaki.tools.symbolic_compute import symbolic_compute
from bourbaki.tools.web_search import web_search


def build_toolset() -> FunctionToolset:
    """Create a FunctionToolset with all Bourbaki tools registered."""
    toolset = FunctionToolset()
    toolset.tool(symbolic_compute)
    toolset.tool(lean_prover)
    toolset.tool(sequence_lookup)
    toolset.tool(paper_search)
    toolset.tool(web_search)
    return toolset

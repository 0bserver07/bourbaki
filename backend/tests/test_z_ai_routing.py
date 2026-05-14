"""Routing and compatibility tests for z.ai backends (issue #13).

Two related concerns covered here:

1. **Routing.** ``_resolve_model_object`` in each of the three prover
   modules (proposer, reviewer, memory) supports four prefix flavours:

   - ``glm:<model>``     → Anthropic-compat endpoint (default for GLM)
   - ``glm-oai:<model>`` → OpenAI-compat endpoint (alternative pool)
   - ``ollama-cloud:<model>`` → Ollama Cloud OpenAI-compat
   - anything else → passed through to pydantic_ai verbatim

2. **The args_as_dict shim** in :mod:`bourbaki.prover._pydantic_ai_compat`.
   pydantic_ai 1.56 crashes when ``ToolCallPart.args`` is anything other
   than a JSON string or a dict (e.g. a list — z.ai's tool-use responses
   sometimes look like that on retry). The shim replaces ``args_as_dict``
   with a defensive version that handles any input shape. The tests below
   exercise that the shim is loaded and behaves correctly.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

# Importing the prover package applies the args_as_dict shim as a side
# effect (via __init__.py). All tests below rely on it.
from bourbaki.prover import memory as memory_mod
from bourbaki.prover import proposer as proposer_mod
from bourbaki.prover import reviewer as reviewer_mod


_ZAI_ANTHROPIC_BASE_URL = "https://api.z.ai/api/anthropic"
_ZAI_OPENAI_BASE_URL = "https://api.z.ai/api/paas/v4/"


# ---------------------------------------------------------------------------
# Routing: glm: → Anthropic-compat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [proposer_mod, reviewer_mod, memory_mod],
    ids=["proposer", "reviewer", "memory"],
)
def test_glm_prefix_routes_to_anthropic_compat(module, monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    model = module._resolve_model_object("glm:glm-5.1")
    assert isinstance(model, AnthropicModel)
    assert str(model.client.base_url).rstrip("/") == _ZAI_ANTHROPIC_BASE_URL


# ---------------------------------------------------------------------------
# Routing: glm-oai: → OpenAI-compat (alternative pool)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [proposer_mod, reviewer_mod, memory_mod],
    ids=["proposer", "reviewer", "memory"],
)
def test_glm_oai_prefix_routes_to_openai_compat(module, monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    model = module._resolve_model_object("glm-oai:glm-5.1")
    assert isinstance(model, OpenAIChatModel)
    assert str(model.client.base_url).rstrip("/") == _ZAI_OPENAI_BASE_URL.rstrip("/")


# ---------------------------------------------------------------------------
# Routing: ollama-cloud: → OpenAI-compat against Ollama Cloud
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [proposer_mod, reviewer_mod, memory_mod],
    ids=["proposer", "reviewer", "memory"],
)
def test_ollama_cloud_prefix_routes_to_openai_compat(module, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    model = module._resolve_model_object("ollama-cloud:gpt-oss-20b")
    assert isinstance(model, OpenAIChatModel)
    assert "ollama.com" in str(model.client.base_url)


# ---------------------------------------------------------------------------
# Routing: pass-through for plain pydantic_ai strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [proposer_mod, reviewer_mod, memory_mod],
    ids=["proposer", "reviewer", "memory"],
)
def test_passthrough_for_non_custom_prefix(module):
    assert module._resolve_model_object("openai:gpt-4o") == "openai:gpt-4o"
    assert module._resolve_model_object("anthropic:claude-sonnet-4-5") == "anthropic:claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# args_as_dict shim — the bug we're working around
# ---------------------------------------------------------------------------


def test_args_as_dict_shim_handles_list_args():
    """z.ai's Anthropic-compat sometimes returns ``input`` as a list.

    Without the shim, pydantic_ai's ``ToolCallPart.args_as_dict`` would
    raise ``TypeError: Expected bytes, bytearray or str`` on retry message
    re-mapping. With the shim, we get a graceful wrap.
    """
    part = ToolCallPart(
        tool_name="final_result",
        args=["unexpected", "list", "from", "z.ai"],  # type: ignore[arg-type]
        tool_call_id="abc",
    )
    out = part.args_as_dict()
    assert isinstance(out, dict)
    assert out == {"_raw": ["unexpected", "list", "from", "z.ai"]}


def test_args_as_dict_shim_handles_none():
    part = ToolCallPart(tool_name="t", args=None, tool_call_id="abc")
    assert part.args_as_dict() == {}


def test_args_as_dict_shim_handles_dict():
    part = ToolCallPart(
        tool_name="t",
        args={"a": 1, "b": "two"},
        tool_call_id="abc",
    )
    assert part.args_as_dict() == {"a": 1, "b": "two"}


def test_args_as_dict_shim_handles_json_string():
    part = ToolCallPart(tool_name="t", args='{"a": 1}', tool_call_id="abc")
    assert part.args_as_dict() == {"a": 1}


def test_args_as_dict_shim_handles_malformed_string():
    part = ToolCallPart(tool_name="t", args="not json {{{", tool_call_id="abc")
    out = part.args_as_dict()
    assert out == {"_raw": "not json {{{"}


def test_args_as_dict_shim_handles_json_scalar_string():
    # Valid JSON, but the parsed value isn't a dict — wrap defensively.
    part = ToolCallPart(tool_name="t", args="42", tool_call_id="abc")
    assert part.args_as_dict() == {"_value": 42}

"""Routing tests for z.ai's GLM models (issue #13).

The prover loop previously routed ``glm:`` model strings through
``AnthropicProvider`` against z.ai's Anthropic-compatible endpoint at
``https://api.z.ai/api/anthropic``. On PutnamBench problems that path
crashes inside ``pydantic_ai/messages.py::args_as_dict`` during retry
re-mapping because z.ai's tool-use response shape feeds a non-string,
non-dict back into a ``ToolCallPart.args``.

This module:

1. Documents the upstream crash with a tiny reproduction so a regression
   is caught in CI even without an API key (see
   ``test_args_as_dict_crashes_on_non_str_non_dict_args``).

2. Locks down our fix: ``_resolve_model_object("glm:glm-5.1")`` must now
   return an ``OpenAIChatModel`` pointed at z.ai's OpenAI-compatible
   endpoint, not an ``AnthropicModel``. Verified across all three prover
   modules — proposer, reviewer, memory — since each carries its own
   resolver (deliberate, per the Phase 2 design doc).
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel

from bourbaki.prover import memory as memory_mod
from bourbaki.prover import proposer as proposer_mod
from bourbaki.prover import reviewer as reviewer_mod


# Expected OpenAI-compatible base URL for z.ai. Trailing slash matches
# z.ai's documented example and the constant in each prover module.
_EXPECTED_BASE_URL = "https://api.z.ai/api/paas/v4/"


# ---------------------------------------------------------------------------
# Upstream crash reproduction (issue #13)
# ---------------------------------------------------------------------------


def test_args_as_dict_crashes_on_non_str_non_dict_args():
    """Regression: this is the exact pydantic_ai bug that bit us.

    ``ToolCallPart.args`` is typed as ``str | dict[str, Any] | None`` but
    ``args_as_dict`` only checks for ``dict``. Any other type (list,
    bytes-like, etc.) falls through to ``pydantic_core.from_json(self.args)``
    which raises ``TypeError: Expected bytes, bytearray or str``.

    The Anthropic adapter triggers it because it stores
    ``response_part.input`` via ``cast(dict[str, Any], item.input)``
    without an actual type conversion (``anthropic.py:577``). When z.ai's
    response feeds back a malformed shape, the re-map on retry crashes.

    Pinning this here so if pydantic_ai patches it upstream we notice and
    can revisit the OpenAI-compat workaround.
    """
    part = ToolCallPart(
        tool_name="final_result",
        args=["unexpected", "list", "from", "z.ai"],  # type: ignore[arg-type]
        tool_call_id="abc",
    )
    with pytest.raises(TypeError, match="Expected bytes, bytearray or str"):
        part.args_as_dict()


# ---------------------------------------------------------------------------
# Resolver lockdown — each prover module
# ---------------------------------------------------------------------------


def _assert_glm_routes_to_openai_compat(resolved):
    """Shared lockdown: resolved model must be OpenAIChatModel pointed at z.ai."""
    assert isinstance(resolved, OpenAIChatModel), (
        f"glm: must route through OpenAIChatModel to dodge the Anthropic-compat "
        f"args_as_dict crash (issue #13); got {type(resolved).__name__}"
    )
    assert resolved.model_name == "glm-5.1"
    # OpenAIChatModel exposes the underlying AsyncOpenAI via .client.
    base_url = str(resolved.client.base_url)
    assert base_url == _EXPECTED_BASE_URL, (
        f"expected z.ai's OpenAI-compat base URL ({_EXPECTED_BASE_URL}), got {base_url}"
    )


def test_proposer_resolves_glm_to_openai_compat_endpoint(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "dummy-for-test")
    resolved = proposer_mod._resolve_model_object("glm:glm-5.1")
    _assert_glm_routes_to_openai_compat(resolved)


def test_reviewer_resolves_glm_to_openai_compat_endpoint(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "dummy-for-test")
    resolved = reviewer_mod._resolve_model_object("glm:glm-5.1")
    _assert_glm_routes_to_openai_compat(resolved)


def test_memory_resolves_glm_to_openai_compat_endpoint(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "dummy-for-test")
    resolved = memory_mod._resolve_model_object("glm:glm-5.1")
    _assert_glm_routes_to_openai_compat(resolved)


def test_resolvers_use_glm_api_key_env_var(monkeypatch):
    """Lock down that the GLM_API_KEY env var is what's actually picked up.

    Previously the AnthropicProvider would fall back to ANTHROPIC_API_KEY
    if GLM_API_KEY was empty — silently using the wrong key. The new
    routing uses OpenAIProvider which respects the passed api_key
    directly, no env-var fallback. We confirm by setting GLM_API_KEY to
    a sentinel and reading it back off the constructed client.
    """
    sentinel = "z-ai-key-sentinel-1234"
    monkeypatch.setenv("GLM_API_KEY", sentinel)
    # Ensure no Anthropic key leaks in and rescues an oversight.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    for mod in (proposer_mod, reviewer_mod, memory_mod):
        resolved = mod._resolve_model_object("glm:glm-5.1")
        # AsyncOpenAI stores the key on the underlying client.
        assert resolved.client.api_key == sentinel, (
            f"{mod.__name__} did not propagate GLM_API_KEY to the OpenAI client"
        )


# ---------------------------------------------------------------------------
# Passthrough — non-glm prefixes must remain untouched
# ---------------------------------------------------------------------------


def test_passthrough_for_non_custom_prefix():
    """Plain provider strings (e.g. ``openai:gpt-4o``) must pass through
    untouched so Pydantic AI handles them natively.
    """
    for mod in (proposer_mod, reviewer_mod, memory_mod):
        assert mod._resolve_model_object("openai:gpt-4o") == "openai:gpt-4o"


def test_ollama_cloud_prefix_still_routes_to_openai_compat(monkeypatch):
    """``ollama-cloud:`` is unrelated to the z.ai fix and should keep
    working. Pin it so a sloppy refactor doesn't break Ollama Cloud users.
    """
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "dummy")
    for mod in (proposer_mod, reviewer_mod, memory_mod):
        resolved = mod._resolve_model_object("ollama-cloud:llama3.1")
        assert isinstance(resolved, OpenAIChatModel)
        assert resolved.model_name == "llama3.1"
        assert "ollama.com" in str(resolved.client.base_url)

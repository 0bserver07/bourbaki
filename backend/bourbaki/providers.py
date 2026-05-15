"""Provider routing: model string → Pydantic AI model object.

One source of truth for which prefix routes to which provider.

This module is imported by the proposer / reviewer / memory nodes of the
prover loop. The legacy ``bourbaki.agent.core._resolve_model_object``
keeps its own local copy because it still uses the deprecated
``OpenAIModel`` alias; migration is tracked separately.
"""

from __future__ import annotations

import os

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider


# Endpoint constants — kept here so all sites use the same URLs.
#
# z.ai has two API surfaces with separate billing pools:
#  - Anthropic-compat (`glm:` prefix, default) — where GLM-5/5.1 resource
#    packages live. Survives pydantic_ai's retry mapping thanks to the
#    args_as_dict shim in ``bourbaki.prover._pydantic_ai_compat``.
#  - OpenAI-compat (`glm-oai:` prefix) — separate billing pool, alternative
#    routing if the Anthropic-compat endpoint becomes unsuitable.
# See issue #13.
ZAI_ANTHROPIC_BASE_URL = "https://api.z.ai/api/anthropic"
ZAI_OPENAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
OLLAMA_LOCAL_DEFAULT = "http://localhost:11434/v1"


def resolve_model_object(model_str: str) -> str | OpenAIChatModel | AnthropicModel:
    """Resolve a Bourbaki model string into a Pydantic AI model.

    Supported prefixes:

    - ``glm:<model>`` — z.ai Anthropic-compat, key=GLM_API_KEY
    - ``glm-oai:<model>`` — z.ai OpenAI-compat, key=GLM_API_KEY
    - ``ollama:<model>`` — local Ollama daemon, ``OLLAMA_BASE_URL``
      override (default ``http://localhost:11434/v1``). The daemon
      transparently proxies ``:cloud`` tags to Ollama Cloud, so frontier
      models (e.g. ``ollama:kimi-k2.6:cloud``) work without an API key.
    - ``ollama-cloud:<model>`` — direct Ollama Cloud at
      ``https://ollama.com/v1`` (key=OLLAMA_CLOUD_API_KEY).
    - anything else — handed verbatim to Pydantic AI
      (e.g. ``openai:gpt-4o``, ``anthropic:claude-sonnet-4-5``).

    Prefix order matters: ``ollama-cloud:`` is checked before ``ollama:``
    (more specific first); a regression guard in
    :mod:`tests.test_z_ai_routing` covers this.
    """

    if model_str.startswith("ollama-cloud:"):
        model_name = model_str.removeprefix("ollama-cloud:")
        api_key = os.environ.get("OLLAMA_CLOUD_API_KEY", "ollama")
        provider = OpenAIProvider(
            base_url=OLLAMA_CLOUD_BASE_URL,
            api_key=api_key,
        )
        return OpenAIChatModel(model_name, provider=provider)

    if model_str.startswith("ollama:"):
        model_name = model_str.removeprefix("ollama:")
        base_url = os.environ.get("OLLAMA_BASE_URL", OLLAMA_LOCAL_DEFAULT)
        provider = OpenAIProvider(
            base_url=base_url,
            api_key="ollama",  # dummy; local daemon ignores
        )
        return OpenAIChatModel(model_name, provider=provider)

    if model_str.startswith("glm-oai:"):
        model_name = model_str.removeprefix("glm-oai:")
        api_key = os.environ.get("GLM_API_KEY", "")
        provider = OpenAIProvider(
            base_url=ZAI_OPENAI_BASE_URL,
            api_key=api_key,
        )
        return OpenAIChatModel(model_name, provider=provider)

    if model_str.startswith("glm:"):
        model_name = model_str.removeprefix("glm:")
        api_key = os.environ.get("GLM_API_KEY", "")
        provider = AnthropicProvider(
            base_url=ZAI_ANTHROPIC_BASE_URL,
            api_key=api_key,
        )
        return AnthropicModel(model_name, provider=provider)

    return model_str

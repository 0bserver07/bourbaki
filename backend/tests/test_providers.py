"""Tests for :mod:`bourbaki.providers` — the single source of truth for
model-string → Pydantic AI model resolution.

The per-module ``_resolve_model_object`` aliases in proposer / reviewer /
memory are exercised by :mod:`tests.test_z_ai_routing`. This file pins
the underlying routing function directly so future moves of the alias
boundary don't silently lose coverage.
"""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from bourbaki.providers import (
    OLLAMA_LOCAL_DEFAULT,
    ZAI_ANTHROPIC_BASE_URL,
    resolve_model_object,
)


def test_resolve_glm_prefix(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    model = resolve_model_object("glm:glm-5.1")
    assert isinstance(model, AnthropicModel)
    assert str(model.client.base_url).rstrip("/") == ZAI_ANTHROPIC_BASE_URL


def test_resolve_ollama_local_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    model = resolve_model_object("ollama:foo")
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "foo"
    assert str(model.client.base_url).rstrip("/") == OLLAMA_LOCAL_DEFAULT.rstrip("/")


def test_resolve_ollama_local_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434/v1")
    model = resolve_model_object("ollama:gpt-oss:120b")
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-oss:120b"
    assert "10.0.0.5" in str(model.client.base_url)


def test_resolve_ollama_cloud(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    model = resolve_model_object("ollama-cloud:gpt-oss:120b")
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-oss:120b"
    assert "ollama.com" in str(model.client.base_url)


def test_resolve_passthrough():
    assert resolve_model_object("openai:gpt-4o") == "openai:gpt-4o"
    assert resolve_model_object("anthropic:claude-sonnet-4-5") == "anthropic:claude-sonnet-4-5"


def test_ollama_cloud_takes_precedence_over_ollama(monkeypatch):
    """``ollama-cloud:foo`` must NOT match the ``ollama:`` branch.

    Regression guard against someone reordering the prefix checks and
    silently routing cloud requests to localhost.
    """
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    model = resolve_model_object("ollama-cloud:kimi-k2.6")
    assert isinstance(model, OpenAIChatModel)
    assert "ollama.com" in str(model.client.base_url)
    assert "localhost" not in str(model.client.base_url)

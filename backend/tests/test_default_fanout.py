"""Unit tests for the smart default fanout selector.

These tests use ``monkeypatch.setenv`` / ``monkeypatch.delenv`` to drive
the env-var probe inside ``smart_default_fanout`` so we never depend on
the test runner's ambient ``.env``. All assertions are on the
:class:`ProposerConfig` shape, not on any provider network call.
"""

from __future__ import annotations

import pytest

from bourbaki.prover.default_fanout import (
    _PROFILES,
    smart_default_fanout,
)
from bourbaki.prover.parallel_proposer import ProposerConfig


# Every env var the fanout selector cares about. We clear all of them at
# the start of each test (via ``_clear_all_keys``) so individual cases
# can opt back in only the keys they want eligible.
_ALL_KEYS: tuple[str, ...] = tuple(
    sorted({p.required_env_var for p in _PROFILES})
)


@pytest.fixture(autouse=True)
def _clear_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every fanout-relevant env var before each test.

    Without this, a developer running the suite with their personal
    .env loaded would see different fanout shapes per run.
    """
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_only_glm_returns_baseline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """With just GLM_API_KEY set, every tier returns one baseline leg.

    There is no other eligible profile to pair with the baseline, so
    even ``max`` collapses to a single leg. The point of the test is to
    confirm we don't synthesize legs for absent providers.
    """
    monkeypatch.setenv("GLM_API_KEY", "x")

    for tier in ("minimal", "balanced", "max"):
        cfgs = smart_default_fanout(tier=tier)
        assert len(cfgs) == 1, f"tier={tier} yielded {len(cfgs)} legs"
        assert cfgs[0].model == "glm:glm-5.1"
        assert cfgs[0].llm_timeout == 90.0


def test_only_ollama_cloud_returns_six_ollama_legs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OLLAMA_CLOUD_API_KEY alone unlocks the full ollama family.

    In ``max`` mode every eligible profile is included. The order
    asserts the deterministic ``_PROFILES`` ordering (frontier first,
    then coder, then the two non-streaming reasoning legs, then the
    two streaming early-cancel reasoning legs).
    """
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "y")

    cfgs = smart_default_fanout(tier="max")
    assert len(cfgs) == 6
    models = [c.model for c in cfgs]
    assert models == [
        "ollama-cloud:gpt-oss:120b",
        "ollama-cloud:qwen3-coder:480b",
        "ollama-native:kimi-k2.6",
        "ollama-native:deepseek-v4-pro",
        "ollama-stream:kimi-k2.6",
        "ollama-stream:deepseek-v4-pro",
    ]


def test_glm_plus_ollama_balanced_returns_three_legs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The realistic user environment: GLM + OLLAMA_CLOUD keys.

    Balanced policy: cheap GLM baseline + the first two diversity legs
    (gpt-oss + qwen3-coder), no Claude available → 3 legs total. The
    ``reasoning-fast`` streaming legs are excluded from balanced (only
    eligible under ``max``) so they don't push out coder/frontier.

    Max policy on the same env yields all 7 eligible profiles
    (1 baseline + 2 ollama-cloud + 2 ollama-native + 2 ollama-stream).
    """
    monkeypatch.setenv("GLM_API_KEY", "x")
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "y")

    balanced = smart_default_fanout(tier="balanced")
    assert [c.model for c in balanced] == [
        "glm:glm-5.1",
        "ollama-cloud:gpt-oss:120b",
        "ollama-cloud:qwen3-coder:480b",
    ]

    full = smart_default_fanout(tier="max")
    assert len(full) == 7


def test_all_keys_max_returns_every_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every key set, ``max`` returns every profile.

    The streaming ``reasoning-fast`` variants race alongside the
    non-streaming reasoning legs (both kimi and deepseek appear twice,
    once per adapter).
    """
    for key in _ALL_KEYS:
        monkeypatch.setenv(key, "k")

    cfgs = smart_default_fanout(tier="max")
    assert len(cfgs) == len(_PROFILES)
    # Anthropic, OpenAI, and streaming legs must be present.
    labels = [c.label for c in cfgs]
    assert "claude-sonnet" in labels
    assert "openai-gpt5-mini" in labels
    assert "kimi-stream" in labels
    assert "deepseek-stream" in labels


def test_include_roles_baseline_only_returns_glm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``include_roles=("baseline",)`` filters down to just the baseline leg.

    Useful in A/B tests where you want to isolate one model family.
    """
    monkeypatch.setenv("GLM_API_KEY", "x")
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "y")

    cfgs = smart_default_fanout(tier="max", include_roles=("baseline",))
    assert len(cfgs) == 1
    assert cfgs[0].model == "glm:glm-5.1"


def test_verify_with_lean_flag_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every returned config has ``verify_with_lean=True`` when requested.

    ProposerConfig is frozen, so this confirms the ``dataclasses.replace``
    path produces fresh instances rather than mutating the module-level
    profile templates.
    """
    monkeypatch.setenv("GLM_API_KEY", "x")
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "y")

    cfgs = smart_default_fanout(tier="balanced", verify_with_lean=True)
    assert all(c.verify_with_lean for c in cfgs)

    # The module-level _PROFILES templates must NOT be mutated as a side
    # effect — they default to verify_with_lean=False.
    assert all(p.config.verify_with_lean is False for p in _PROFILES)


def test_no_env_vars_raises_value_error() -> None:
    """An empty environment produces a clear error message naming the
    env vars the caller could set.

    The autouse fixture has already cleared every relevant key.
    """
    with pytest.raises(ValueError, match="no API keys present"):
        smart_default_fanout()


def test_minimal_tier_caps_at_two_legs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``minimal`` returns at most one baseline plus one diversity leg
    even when many providers are configured (4+ eligible).
    """
    for key in _ALL_KEYS:
        monkeypatch.setenv(key, "k")

    cfgs = smart_default_fanout(tier="minimal")
    assert len(cfgs) == 2
    # Order: baseline first, then the first frontier-class leg.
    assert cfgs[0].model == "glm:glm-5.1"
    assert cfgs[1].model == "ollama-cloud:gpt-oss:120b"


def test_invalid_tier_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tier outside ``{minimal, balanced, max}`` is rejected eagerly,
    before we try to read env vars, so the error doesn't depend on
    which keys happen to be set.
    """
    monkeypatch.setenv("GLM_API_KEY", "x")

    with pytest.raises(ValueError, match="tier="):
        smart_default_fanout(tier="frontier-only")  # type: ignore[arg-type]


def test_include_roles_wipes_eligible_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When include_roles eliminates every eligible leg, raise a distinct
    error so the caller can tell "missing key" from "role mismatch".
    """
    monkeypatch.setenv("GLM_API_KEY", "x")
    # Eligible role is "baseline"; ask for "claude" only — empty result.
    with pytest.raises(ValueError, match="no profiles remained after"):
        smart_default_fanout(tier="max", include_roles=("claude",))


def test_balanced_tier_with_anthropic_adds_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Balanced policy adds the Anthropic leg when its key is set, on top
    of baseline + 2 diversity legs → 4 total.
    """
    monkeypatch.setenv("GLM_API_KEY", "x")
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "y")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "z")

    cfgs = smart_default_fanout(tier="balanced")
    assert len(cfgs) == 4
    assert cfgs[-1].model == "anthropic:claude-sonnet-4-6"


def test_returned_configs_are_proposer_config_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check the return type matches what parallel_propose expects."""
    monkeypatch.setenv("GLM_API_KEY", "x")

    cfgs = smart_default_fanout(tier="minimal")
    assert all(isinstance(c, ProposerConfig) for c in cfgs)

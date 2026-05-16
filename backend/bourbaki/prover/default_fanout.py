"""Smart default fanout config based on available API keys.

Used by the CLI runners when ``--proposer-models`` is omitted; selects a
diverse set of legs from whichever providers have keys configured.

Design
------
The full ``--proposer-models`` flag is rich (per-leg models, timeouts,
verify flags, labels) but most users don't want to type the long form to
get a sensible fanout. ``smart_default_fanout`` reads which API keys are
available in the environment (after ``bourbaki.config.export_api_keys``
has run) and returns a tiered fanout of :class:`ProposerConfig` legs.

Three tiers are supported:

- ``minimal``: 2 legs — one cheap baseline + one frontier leg. Use this
  for fast iteration on the prompt or scaffold; resembles the legacy
  single-model path but adds one diverse peer for free.
- ``balanced``: 3-4 legs — cheap baseline + up to two frontier diversity
  legs + Claude if the key is available. Defaults to this; targets the
  sweet spot of "more correct than single-model, fast enough to run on
  a 35-problem subset in under an hour".
- ``max``: every leg whose key is present (up to 7). Use this when
  wallclock isn't the constraint and you want maximum coverage of model
  failure modes — e.g. a full 244-problem run that bills hours anyway.

Roles
-----
Each profile has a ``role`` tag the tier filter uses to compose the
fanout. Roles are intentionally fuzzy — ``baseline`` is the "always-on"
cheap leg, ``frontier`` covers the heaviest open-weights, ``reasoning``
is for thinking models routed through ``ollama-native:``, ``coder`` for
Lean-tuned coders, ``claude`` / ``openai`` for the proprietary peers
when their keys are available. Callers can override the role mix via
``include_roles`` if they want e.g. only ``claude`` + ``baseline``.

Conventions
-----------
- Profile order in ``_PROFILES`` is the order legs appear in the
  returned list. This keeps fanout composition deterministic across
  runs (helpful for log diffing) and puts the cheapest baseline first.
- Per-leg timeouts mirror the empirically-tuned defaults from the
  ``--proposer-models`` examples in the runner docstrings: 90s for GLM,
  180s for the 100-500B ollama-cloud legs, 300s for the 1T thinking
  models via ``ollama-native:``, 60s for the proprietary peers.
- ``verify_with_lean`` is applied uniformly across all legs when the
  flag is requested. This matches how the runner's
  ``_parse_proposer_models`` treats ``race_mode == "first_verified"`` —
  every leg must compile to be eligible to win.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

from bourbaki.prover.parallel_proposer import ProposerConfig

logger = logging.getLogger(__name__)


# Tier semantics live here so they're a single source of truth across
# the module, tests, and CLI help text.
_VALID_TIERS: tuple[str, ...] = ("minimal", "balanced", "max")


@dataclass(frozen=True)
class ModelProfile:
    """One model that could appear in a fanout.

    Attributes:
        config: The :class:`ProposerConfig` to forward into the fanout
            when this profile is selected. ``verify_with_lean`` is
            overridden by :func:`smart_default_fanout` if the caller
            opts into verified-race mode; the default here is False so
            ``first_completed`` mode behaves identically to a hand-rolled
            ``--proposer-models`` spec.
        required_env_var: Environment variable that must be non-empty
            for this profile to be eligible. Checked via ``os.environ``
            after :func:`bourbaki.config.export_api_keys` has run.
        role: Coarse classifier used by the tier filter:

            - ``baseline``: always-on cheap leg (GLM)
            - ``frontier``: heaviest open-weights (gpt-oss:120b)
            - ``coder``: Lean-tuned coder (qwen3-coder:480b)
            - ``reasoning``: thinking models via ollama-native (kimi, deepseek)
            - ``claude``: Anthropic peer
            - ``openai``: OpenAI peer
    """

    config: ProposerConfig
    required_env_var: str
    role: str


# Order here defines the order of legs in the returned fanout list:
# cheapest/fastest baselines first, then frontier diversity, then
# proprietary peers. Tests assert on this ordering.
_PROFILES: tuple[ModelProfile, ...] = (
    # GLM — cheap baseline, always pulled into any tier when its key is set.
    ModelProfile(
        ProposerConfig(model="glm:glm-5.1", llm_timeout=90.0, label="glm-5.1"),
        required_env_var="GLM_API_KEY",
        role="baseline",
    ),
    # Ollama Cloud — frontier diversity, no reasoning chain, fastest open-weight.
    ModelProfile(
        ProposerConfig(
            model="ollama-cloud:gpt-oss:120b",
            llm_timeout=180.0,
            label="gpt-oss-120b",
        ),
        required_env_var="OLLAMA_CLOUD_API_KEY",
        role="frontier",
    ),
    # Ollama Cloud — Lean-specialized coder model.
    ModelProfile(
        ProposerConfig(
            model="ollama-cloud:qwen3-coder:480b",
            llm_timeout=180.0,
            label="qwen3-coder",
        ),
        required_env_var="OLLAMA_CLOUD_API_KEY",
        role="coder",
    ),
    # Ollama Native — 1T thinking model via /api/chat (needs longer timeout).
    ModelProfile(
        ProposerConfig(
            model="ollama-native:kimi-k2.6",
            llm_timeout=300.0,
            label="kimi-k2.6",
        ),
        required_env_var="OLLAMA_CLOUD_API_KEY",
        role="reasoning",
    ),
    # Ollama Native — DeepSeek thinking model via /api/chat.
    ModelProfile(
        ProposerConfig(
            model="ollama-native:deepseek-v4-pro",
            llm_timeout=300.0,
            label="deepseek-v4-pro",
        ),
        required_env_var="OLLAMA_CLOUD_API_KEY",
        role="reasoning",
    ),
    # Anthropic — strong fast second baseline when ANTHROPIC_API_KEY is set.
    ModelProfile(
        ProposerConfig(
            model="anthropic:claude-sonnet-4-6",
            llm_timeout=60.0,
            label="claude-sonnet",
        ),
        required_env_var="ANTHROPIC_API_KEY",
        role="claude",
    ),
    # OpenAI — fast, cheap proprietary peer when OPENAI_API_KEY is set.
    ModelProfile(
        ProposerConfig(
            model="openai:gpt-5-mini",
            llm_timeout=60.0,
            label="openai-gpt5-mini",
        ),
        required_env_var="OPENAI_API_KEY",
        role="openai",
    ),
)


def _key_present(env_var: str) -> bool:
    """True iff ``env_var`` is set in the process environment to a non-empty value."""
    return bool(os.environ.get(env_var, "").strip())


def _eligible_profiles() -> list[ModelProfile]:
    """Return profiles whose ``required_env_var`` is currently set, preserving order."""
    return [p for p in _PROFILES if _key_present(p.required_env_var)]


def _filter_by_tier(profiles: list[ModelProfile], tier: str) -> list[ModelProfile]:
    """Apply the tier policy to an already-eligible profile list.

    The tier rules are:

    - ``minimal``: keep the first ``baseline`` (if any) and the first
      profile whose role is in {frontier, coder, reasoning, claude,
      openai}. Caps the fanout at 2 legs.
    - ``balanced``: keep the first ``baseline``, up to two of the
      ``frontier`` / ``coder`` / ``reasoning`` legs (in profile order),
      and the first ``claude`` if available. Caps the fanout at 4 legs.
    - ``max``: keep everything that's eligible (no extra filtering).

    Tie-breaks are resolved by profile order in :data:`_PROFILES`, so
    the returned fanout is deterministic.
    """
    if tier == "max":
        return profiles

    if tier == "minimal":
        kept: list[ModelProfile] = []
        # First baseline (if any).
        baseline = next((p for p in profiles if p.role == "baseline"), None)
        if baseline is not None:
            kept.append(baseline)
        # First "anything frontier-ish" — frontier > coder > reasoning >
        # claude > openai by virtue of _PROFILES ordering.
        non_baseline_roles = ("frontier", "coder", "reasoning", "claude", "openai")
        first_other = next(
            (p for p in profiles if p.role in non_baseline_roles), None
        )
        if first_other is not None:
            kept.append(first_other)
        return kept

    # tier == "balanced"
    kept = []
    baseline = next((p for p in profiles if p.role == "baseline"), None)
    if baseline is not None:
        kept.append(baseline)
    # Up to two frontier-diversity legs (frontier/coder/reasoning) in
    # profile order. We don't try to pick one of each role because the
    # most useful diversity in practice is "different model families,
    # whichever was provisioned" — _PROFILES already encodes the order
    # we care about (gpt-oss first, qwen3-coder second).
    diversity_roles = ("frontier", "coder", "reasoning")
    diversity = [p for p in profiles if p.role in diversity_roles][:2]
    kept.extend(diversity)
    # Claude as a quality boost when available.
    claude = next((p for p in profiles if p.role == "claude"), None)
    if claude is not None:
        kept.append(claude)
    return kept


def smart_default_fanout(
    *,
    tier: str = "balanced",
    include_roles: tuple[str, ...] | None = None,
    verify_with_lean: bool = False,
) -> list[ProposerConfig]:
    """Build a fanout list from whichever models have keys configured.

    Reads ``os.environ`` (so call after ``bourbaki.config.export_api_keys``
    has bridged Settings → env vars) to determine which profiles are
    eligible, then applies tier and role filters.

    Args:
        tier: ``"minimal"`` | ``"balanced"`` | ``"max"``. See module
            docstring for semantics. Defaults to ``"balanced"``, which
            is the right tradeoff for a 35-problem benchmark run.
        include_roles: Optional whitelist of roles to keep after the
            tier filter. When set, the result is the intersection of
            the tier output and the role set. Useful for A/B tests
            ("only the proprietary peers, please").
        verify_with_lean: If True, every returned config has its
            ``verify_with_lean`` field set to True (via ``dataclasses.replace``,
            since :class:`ProposerConfig` is frozen). Pair this with
            ``race_mode="first_verified"`` in :func:`parallel_propose`
            so each leg is gated on a compile check before it can win.

    Returns:
        A non-empty list of :class:`ProposerConfig` legs in deterministic
        order. The list is suitable to pass directly to
        :func:`parallel_propose` or via
        ``run_minif2f(loop_proposer_configs=...)``.

    Raises:
        ValueError: If ``tier`` is not in ``{minimal, balanced, max}``,
            or if no eligible legs remain after filtering. The latter
            error message names which env vars would unlock a fanout,
            so the caller can fix their environment without grepping.
    """
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"smart_default_fanout: tier={tier!r} not in {_VALID_TIERS}"
        )

    eligible = _eligible_profiles()
    if not eligible:
        # Surface every env var we know about so the caller can pick one.
        candidates = sorted({p.required_env_var for p in _PROFILES})
        raise ValueError(
            "smart_default_fanout: no API keys present in environment; "
            f"set one of {candidates} (e.g. via backend/.env) and rerun"
        )

    after_tier = _filter_by_tier(eligible, tier)

    if include_roles is not None:
        wanted = set(include_roles)
        after_tier = [p for p in after_tier if p.role in wanted]

    if not after_tier:
        # The tier had eligible profiles but the role filter wiped them.
        # Distinct error so callers can tell "no keys" from "no overlap".
        raise ValueError(
            f"smart_default_fanout: no profiles remained after "
            f"include_roles={include_roles} (tier={tier}); eligible roles "
            f"were {sorted({p.role for p in eligible})}"
        )

    configs = [p.config for p in after_tier]
    if verify_with_lean:
        # ProposerConfig is frozen; replace() returns a new instance.
        configs = [replace(c, verify_with_lean=True) for c in configs]

    return configs

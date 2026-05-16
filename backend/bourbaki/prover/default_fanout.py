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
is for thinking models routed through ``ollama-native:``,
``reasoning-fast`` is for the same thinking models routed through the
streaming early-cancel adapter (``ollama-stream:``), ``coder`` for
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
            - ``reasoning-fast``: same thinking models via ollama-stream
              (early-cancel; only eligible under the ``max`` tier)
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
    # Ollama Stream — Kimi via streaming /api/chat with early-cancel.
    # Shorter llm_timeout because streaming is meant to be 3-5x faster than
    # the non-streaming reasoning path. ``reasoning-fast`` role is filtered
    # out by minimal/balanced tiers, so this only appears under ``max``.
    ModelProfile(
        ProposerConfig(
            model="ollama-stream:kimi-k2.6",
            llm_timeout=60.0,
            label="kimi-stream",
        ),
        required_env_var="OLLAMA_CLOUD_API_KEY",
        role="reasoning-fast",
    ),
    # Ollama Stream — DeepSeek via streaming /api/chat with early-cancel.
    ModelProfile(
        ProposerConfig(
            model="ollama-stream:deepseek-v4-pro",
            llm_timeout=60.0,
            label="deepseek-stream",
        ),
        required_env_var="OLLAMA_CLOUD_API_KEY",
        role="reasoning-fast",
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
    enable_mathlib_search: bool = False,
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
        enable_mathlib_search: If True, every returned config has its
            ``enable_mathlib_search`` field set to True so the proposer
            adapter registers the ``mathlib_search`` tool on the leg's
            agent. Currently only takes effect on ``ollama-native:`` legs
            (the native ``/api/chat`` adapter); pydantic_ai legs may
            also support it depending on model.

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
    return _apply_overrides(
        configs,
        verify_with_lean=verify_with_lean,
        enable_mathlib_search=enable_mathlib_search,
    )


# ---------------------------------------------------------------------------
# Per-problem-source routing
# ---------------------------------------------------------------------------

# Map source prefixes (lowercased, taken from the leading ``[a-z]+`` of the
# problem ID) to the role mix that empirically works best for that family.
#
# - aime / imo / putnam: hard olympiad-style problems where conceptual
#   reasoning beats pure pattern-matching. Lead with reasoning models
#   (ollama-native: kimi + deepseek) and Claude if available.
# - algebra / mathd_algebra / amc: computational manipulation where
#   Lean-tuned coders shine — qwen3-coder first, gpt-oss second.
# - numbertheory / induction: proof structure matters; balanced tier
#   (baseline + diversity) is the right starting point.
#
# Anything unrecognised falls back to balanced tier.
_SOURCE_ROLE_PREFERENCE: dict[str, tuple[str, ...]] = {
    # Reasoning-heavy
    "aime": ("reasoning", "claude", "frontier", "baseline"),
    "imo": ("reasoning", "claude", "frontier", "baseline"),
    "putnam": ("reasoning", "claude", "frontier", "baseline"),
    # Coder-heavy
    "algebra": ("coder", "frontier", "baseline"),
    "mathd": ("coder", "frontier", "baseline"),
    "amc": ("coder", "frontier", "baseline"),
    # Balanced
    "numbertheory": ("baseline", "frontier", "coder"),
    "induction": ("baseline", "frontier", "coder"),
}

# Hard cap on leg count for per-problem fanouts: 5 keeps wallclock
# reasonable on a 35-problem subset while still exercising 3-5 model
# families per problem.
_FANOUT_FOR_PROBLEM_MAX_LEGS: int = 5


def _source_from_problem_id(problem_id: str) -> str:
    """Extract the source prefix from a problem ID.

    Mirrors the regex used by the miniF2F loader (``^[a-z]+_``) but tolerates
    IDs without an underscore (returns the whole thing lowercased) so
    callers can pass arbitrary strings without crashing the auto-route.
    """
    head, _, _rest = problem_id.partition("_")
    return head.lower()


def _apply_overrides(
    configs: list[ProposerConfig],
    *,
    verify_with_lean: bool,
    enable_mathlib_search: bool,
) -> list[ProposerConfig]:
    """Apply the ``verify_with_lean`` / ``enable_mathlib_search`` overrides.

    ``ProposerConfig`` is frozen; we use :func:`dataclasses.replace` to
    return fresh instances rather than mutating the module-level profile
    templates (which would persist across calls).
    """
    if not verify_with_lean and not enable_mathlib_search:
        return configs
    out: list[ProposerConfig] = []
    for c in configs:
        updates: dict = {}
        if verify_with_lean:
            updates["verify_with_lean"] = True
        if enable_mathlib_search:
            updates["enable_mathlib_search"] = True
        out.append(replace(c, **updates))
    return out


def fanout_for_problem(
    problem_id: str,
    *,
    enable_mathlib_search: bool = False,
    verify_with_lean: bool = False,
) -> list[ProposerConfig]:
    """Smart fanout selection by problem source prefix.

    Routing (per leading ``[a-z]+`` token of ``problem_id``):

    - ``aime``, ``imo``, ``putnam`` → reasoning-heavy (kimi + deepseek +
      claude if available, plus baseline + frontier as fillers)
    - ``algebra``, ``mathd``, ``mathd_algebra``, ``amc`` → coder-heavy
      (qwen3-coder + gpt-oss + glm)
    - ``numbertheory``, ``induction`` → balanced (baseline + frontier + coder)
    - anything else → :func:`smart_default_fanout` with ``tier="balanced"``

    The function always returns at least one leg as long as at least one
    profile's env var is set. When the preferred roles for a given source
    are missing (e.g. no Ollama Cloud key for reasoning routing), the
    available legs are returned in role-preference order; if fewer than
    3 legs remain after preference filtering, the result is topped up
    with balanced-tier legs that were not already selected. The fanout
    is capped at :data:`_FANOUT_FOR_PROBLEM_MAX_LEGS` (5) total legs.

    Args:
        problem_id: Benchmark problem identifier (e.g. ``"aime_1988_p3"``,
            ``"mathd_algebra_116"``, ``"putnam_2020_a1"``). Only the
            prefix up to the first underscore is consulted.
        enable_mathlib_search: Forwarded to every returned ``ProposerConfig``.
        verify_with_lean: Forwarded to every returned ``ProposerConfig``.

    Returns:
        A non-empty list of :class:`ProposerConfig` legs. Capped at
        :data:`_FANOUT_FOR_PROBLEM_MAX_LEGS` (5) legs.

    Raises:
        ValueError: When no API keys are configured. The error is raised
            by :func:`smart_default_fanout` and names the env vars that
            would unlock a fanout.
    """
    eligible = _eligible_profiles()
    if not eligible:
        # Defer to smart_default_fanout's error message so the caller sees
        # a single canonical "no keys present" diagnostic.
        return smart_default_fanout(
            tier="balanced",
            verify_with_lean=verify_with_lean,
            enable_mathlib_search=enable_mathlib_search,
        )

    source = _source_from_problem_id(problem_id)
    preferred_roles = _SOURCE_ROLE_PREFERENCE.get(source)

    if preferred_roles is None:
        # Unknown source — fall back to balanced tier so the caller still
        # gets a sensible default rather than a single baseline leg.
        return smart_default_fanout(
            tier="balanced",
            verify_with_lean=verify_with_lean,
            enable_mathlib_search=enable_mathlib_search,
        )

    # Build the per-source fanout: preferred-role legs in profile order
    # within each role, then top up with balanced-tier legs that weren't
    # already selected. The balanced top-up handles the "preferred roles
    # are sparse" case (e.g. reasoning routing with only GLM available).
    selected: list[ModelProfile] = []
    seen: set[str] = set()  # model strings already in selected, for dedup

    for role in preferred_roles:
        for profile in eligible:
            if profile.role != role:
                continue
            if profile.config.model in seen:
                continue
            selected.append(profile)
            seen.add(profile.config.model)
            if len(selected) >= _FANOUT_FOR_PROBLEM_MAX_LEGS:
                break
        if len(selected) >= _FANOUT_FOR_PROBLEM_MAX_LEGS:
            break

    # Top up with balanced-tier legs if we're short. This ensures the
    # caller always gets at least the balanced default when keys are
    # available but the preferred roles aren't represented.
    if len(selected) < _FANOUT_FOR_PROBLEM_MAX_LEGS:
        balanced_profiles = _filter_by_tier(eligible, "balanced")
        for profile in balanced_profiles:
            if profile.config.model in seen:
                continue
            selected.append(profile)
            seen.add(profile.config.model)
            if len(selected) >= _FANOUT_FOR_PROBLEM_MAX_LEGS:
                break

    # If selected is somehow empty (shouldn't be — eligible is non-empty
    # and balanced tier returns at least one leg when keys are present),
    # fall back to smart_default_fanout for safety.
    if not selected:
        return smart_default_fanout(
            tier="balanced",
            verify_with_lean=verify_with_lean,
            enable_mathlib_search=enable_mathlib_search,
        )

    configs = [p.config for p in selected]
    return _apply_overrides(
        configs,
        verify_with_lean=verify_with_lean,
        enable_mathlib_search=enable_mathlib_search,
    )

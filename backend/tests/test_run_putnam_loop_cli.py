"""CLI parser tests for ``backend/scripts/run_putnam_loop.py``.

Mirrors the pattern of :mod:`tests.test_run_minif2f_subset_cli` so the two
runners can't drift apart on the wave-5 fanout flags (``--proposer-models``,
``--reviewer-models``, ``--smart-fanout``, ``--auto-route``, ``--race-mode``,
``--reviewer-review-mode``).

The script lives outside the ``bourbaki`` package, so we load it via
``importlib`` and exercise the helpers (``_parse_proposer_models``,
``_parse_reviewer_models``, ``_resolve_proposer_configs``,
``_resolve_reviewer_configs``) directly. The flags are the only mechanism
by which a benchmark run can opt into the multi-provider fanout and
auto-route paths, so the parser is on the hot path even though the rest
of the script touches the network.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_putnam_loop.py"
)


def _load_runner_module():
    """Load ``run_putnam_loop`` from ``backend/scripts`` as a module.

    The script imports ``_runner_common`` (a sibling). We add the scripts
    directory to ``sys.path`` first so that import resolves.
    """
    import sys

    scripts_dir = _SCRIPT_PATH.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "run_putnam_loop_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _parse_proposer_models — same grammar as the miniF2F runner
# ---------------------------------------------------------------------------


def test_parse_proposer_models_two_entries_with_timeouts():
    """Two comma-separated specs, each with an ``@<seconds>`` suffix,
    must yield two :class:`ProposerConfig` with the parsed timeouts.
    Model strings may themselves contain colons (``glm:glm-5.1``); the
    parser must use ``@`` for the model/timeout split.
    """
    mod = _load_runner_module()
    from bourbaki.prover.parallel_proposer import ProposerConfig

    cfgs = mod._parse_proposer_models("glm:glm-5.1@90,ollama-cloud:kimi-k2.6@600")

    assert cfgs is not None
    assert len(cfgs) == 2
    assert isinstance(cfgs[0], ProposerConfig)
    assert cfgs[0].model == "glm:glm-5.1"
    assert cfgs[0].llm_timeout == 90.0
    assert cfgs[1].model == "ollama-cloud:kimi-k2.6"
    assert cfgs[1].llm_timeout == 600.0


def test_parse_proposer_models_empty_string_returns_none():
    """Empty / whitespace-only / ``None`` input must return ``None`` —
    the runner uses that to take the legacy single-model path.
    """
    mod = _load_runner_module()

    assert mod._parse_proposer_models(None) is None
    assert mod._parse_proposer_models("") is None
    assert mod._parse_proposer_models("   ") is None


def test_parse_proposer_models_race_mode_first_verified_flips_verify_flag():
    """When ``race_mode="first_verified"`` is passed, every produced
    leg must have ``verify_with_lean=True``. Otherwise the CLI would
    silently produce an all-downgrade fanout where no leg can win the
    verified race.
    """
    mod = _load_runner_module()

    cfgs = mod._parse_proposer_models(
        "glm:glm-5.1@90,ollama-cloud:gpt-oss:120b@90",
        race_mode="first_verified",
    )
    assert cfgs is not None
    assert all(c.verify_with_lean for c in cfgs)

    # And the default ``first_completed`` mode keeps verify off.
    cfgs2 = mod._parse_proposer_models(
        "glm:glm-5.1@90", race_mode="first_completed",
    )
    assert cfgs2 is not None
    assert all(not c.verify_with_lean for c in cfgs2)


# ---------------------------------------------------------------------------
# _parse_reviewer_models — independent grammar, no verify flag
# ---------------------------------------------------------------------------


def test_parse_reviewer_models_two_entries():
    """Reviewer specs follow the same comma/pipe + ``@<timeout>`` grammar
    as proposer specs, but always produce :class:`ReviewerConfig` with
    no verify flag (the reviewer fanout never runs ``lean_prover`` per leg).
    """
    mod = _load_runner_module()
    from bourbaki.prover.reviewer_fanout import ReviewerConfig

    cfgs = mod._parse_reviewer_models(
        "glm:glm-5.1@60,anthropic:claude-sonnet-4-5@90"
    )

    assert cfgs is not None
    assert len(cfgs) == 2
    assert isinstance(cfgs[0], ReviewerConfig)
    assert cfgs[0].model == "glm:glm-5.1"
    assert cfgs[0].llm_timeout == 60.0
    assert cfgs[1].model == "anthropic:claude-sonnet-4-5"
    assert cfgs[1].llm_timeout == 90.0
    # ReviewerConfig has no verify_with_lean field — confirm.
    assert not hasattr(cfgs[0], "verify_with_lean")


def test_parse_reviewer_models_empty_returns_none():
    """Empty / ``None`` -> ``None`` so the loop falls back to the
    single-model reviewer.
    """
    mod = _load_runner_module()

    assert mod._parse_reviewer_models(None) is None
    assert mod._parse_reviewer_models("") is None
    assert mod._parse_reviewer_models("   ") is None


def test_parse_reviewer_models_bad_timeout_raises():
    """A non-float after ``@`` must raise rather than silently produce a
    ReviewerConfig with the default timeout.
    """
    mod = _load_runner_module()

    with pytest.raises(ValueError, match="bad timeout"):
        mod._parse_reviewer_models("glm:glm-5.1@not-a-number")


# ---------------------------------------------------------------------------
# _resolve_proposer_configs — precedence between --smart-fanout / --auto-route
# / --proposer-models
# ---------------------------------------------------------------------------


def test_resolve_proposer_configs_smart_fanout_balanced_builds_expected_fanout(
    monkeypatch,
):
    """``--smart-fanout balanced --race-mode first_verified`` must call
    ``smart_default_fanout(tier="balanced", verify_with_lean=True)`` and
    return the configs with verify flipped on.
    """
    mod = _load_runner_module()
    from bourbaki.prover import default_fanout as df_mod
    from bourbaki.prover.parallel_proposer import ProposerConfig

    fake_configs = [
        ProposerConfig(model="glm:glm-5.1", label="glm-5.1"),
        ProposerConfig(
            model="ollama-cloud:gpt-oss:120b", label="gpt-oss-120b",
        ),
    ]

    captured: dict = {}

    def fake_smart_default_fanout(*, tier, verify_with_lean=False, **kwargs):
        captured["tier"] = tier
        captured["verify_with_lean"] = verify_with_lean
        if verify_with_lean:
            from dataclasses import replace
            return [replace(c, verify_with_lean=True) for c in fake_configs]
        return list(fake_configs)

    monkeypatch.setattr(
        df_mod, "smart_default_fanout", fake_smart_default_fanout,
    )

    logger = logging.getLogger("test_resolve_proposer_configs_smart_fanout_balanced")
    args = argparse.Namespace(
        smart_fanout="balanced",
        proposer_models=None,
        auto_route=False,
        race_mode="first_verified",
    )

    resolved = mod._resolve_proposer_configs(args, logger)

    assert captured["tier"] == "balanced"
    assert captured["verify_with_lean"] is True
    assert resolved is not None
    assert len(resolved) == 2
    assert all(c.verify_with_lean for c in resolved)


def test_resolve_proposer_configs_auto_route_returns_none():
    """``--auto-route`` returns ``None`` from the resolver — per-problem
    dispatch happens inside ``run_putnam`` via ``fanout_for_problem``.
    The resolver short-circuits before consulting --smart-fanout or
    --proposer-models so the precedence is unambiguous.
    """
    mod = _load_runner_module()

    logger = logging.getLogger("test_resolve_proposer_configs_auto_route")
    args = argparse.Namespace(
        smart_fanout=None,
        proposer_models="glm:glm-5.1@90",  # would be parsed normally
        auto_route=True,
        race_mode="first_completed",
    )

    resolved = mod._resolve_proposer_configs(args, logger)
    assert resolved is None  # auto-route bypasses every other branch


def test_resolve_proposer_configs_legacy_namespace_without_auto_route():
    """Older callers (and the existing miniF2F pinning tests) build an
    ``argparse.Namespace`` without ``auto_route``. The resolver must
    tolerate that via ``getattr`` — otherwise this would crash with
    ``AttributeError`` when run alongside the rest of the suite.
    """
    mod = _load_runner_module()

    logger = logging.getLogger("test_legacy_namespace")
    # No auto_route attribute — should not crash.
    args = argparse.Namespace(
        smart_fanout=None,
        proposer_models=None,
        race_mode="first_completed",
    )

    resolved = mod._resolve_proposer_configs(args, logger)
    # Falls through to _parse_proposer_models(None) → None
    assert resolved is None


# ---------------------------------------------------------------------------
# _resolve_reviewer_configs — thin wrapper that logs
# ---------------------------------------------------------------------------


def test_resolve_reviewer_configs_with_reviewer_models():
    """When ``--reviewer-models`` is set, the resolver returns the
    parsed leg list and the runner logs the resolved fanout."""
    mod = _load_runner_module()
    from bourbaki.prover.reviewer_fanout import ReviewerConfig

    logger = logging.getLogger("test_resolve_reviewer_configs")
    args = argparse.Namespace(
        reviewer_models="glm:glm-5.1@60,anthropic:claude-sonnet-4-5@90",
        reviewer_review_mode="first_approved",
    )

    resolved = mod._resolve_reviewer_configs(args, logger)
    assert resolved is not None
    assert len(resolved) == 2
    assert all(isinstance(c, ReviewerConfig) for c in resolved)


def test_resolve_reviewer_configs_none_returns_none():
    """Empty / None reviewer spec -> None so the loop falls back to the
    single-model reviewer path."""
    mod = _load_runner_module()

    logger = logging.getLogger("test_resolve_reviewer_configs_none")
    args = argparse.Namespace(
        reviewer_models=None,
        reviewer_review_mode="first_approved",
    )

    resolved = mod._resolve_reviewer_configs(args, logger)
    assert resolved is None


# ---------------------------------------------------------------------------
# argparse-level mutex: --smart-fanout / --proposer-models / --auto-route
# ---------------------------------------------------------------------------


def test_parse_args_rejects_smart_fanout_with_proposer_models(monkeypatch):
    """``--smart-fanout`` and ``--proposer-models`` together must exit
    with an argparse error — no precedence rule, no silent override.
    """
    mod = _load_runner_module()
    import sys

    argv = [
        "run_putnam_loop.py",
        "--smart-fanout", "balanced",
        "--proposer-models", "glm:glm-5.1@90",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        mod.parse_args()


def test_parse_args_rejects_auto_route_with_proposer_models(monkeypatch):
    """``--auto-route`` and ``--proposer-models`` together must exit."""
    mod = _load_runner_module()
    import sys

    argv = [
        "run_putnam_loop.py",
        "--auto-route",
        "--proposer-models", "glm:glm-5.1@90",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        mod.parse_args()


def test_parse_args_rejects_auto_route_with_smart_fanout(monkeypatch):
    """``--auto-route`` and ``--smart-fanout`` together must exit."""
    mod = _load_runner_module()
    import sys

    argv = [
        "run_putnam_loop.py",
        "--auto-route",
        "--smart-fanout", "balanced",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        mod.parse_args()


# ---------------------------------------------------------------------------
# Composition checks: --pass-n + --smart-fanout / --race-mode propagation
# ---------------------------------------------------------------------------


def test_parse_args_pass_n_with_smart_fanout_balanced_composes(monkeypatch):
    """``--pass-n 3 --smart-fanout balanced`` must parse without an
    argparse clash — Pass@N and smart-fanout are orthogonal. The CLI
    layer is the only place this combination is enforced; the rest of
    the wiring lives in ``run_putnam``.
    """
    mod = _load_runner_module()
    import sys

    argv = [
        "run_putnam_loop.py",
        "--pass-n", "3",
        "--smart-fanout", "balanced",
        "--max-iter", "5",
        "--timeout", "120",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = mod.parse_args()
    assert args.pass_n == 3
    assert args.smart_fanout == "balanced"
    assert args.auto_route is False
    assert args.max_iter == 5
    assert args.timeout == 120


def test_parse_args_race_mode_first_verified_default_proposer(monkeypatch):
    """``--race-mode first_verified`` parses on its own (without any
    fanout flag). When propagated to ``run_putnam``, the race mode
    only takes effect under one of the fanout paths, but the CLI layer
    must accept it standalone — argparse must not reject the
    combination just because no proposer flag is set.
    """
    mod = _load_runner_module()
    import sys

    argv = [
        "run_putnam_loop.py",
        "--race-mode", "first_verified",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = mod.parse_args()
    assert args.race_mode == "first_verified"
    assert args.proposer_models is None
    assert args.smart_fanout is None
    assert args.auto_route is False


def test_parse_args_auto_route_with_race_mode_first_verified(monkeypatch):
    """``--auto-route --race-mode first_verified`` must compose — the
    runner forwards the race mode to ``fanout_for_problem`` as the
    ``verify_with_lean`` override on every produced leg, so the CLI
    must accept the combination.
    """
    mod = _load_runner_module()
    import sys

    argv = [
        "run_putnam_loop.py",
        "--auto-route",
        "--race-mode", "first_verified",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = mod.parse_args()
    assert args.auto_route is True
    assert args.race_mode == "first_verified"
    assert args.smart_fanout is None
    assert args.proposer_models is None

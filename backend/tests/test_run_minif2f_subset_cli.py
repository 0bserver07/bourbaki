"""CLI parser tests for ``backend/scripts/run_minif2f_subset.py``.

The script lives outside the ``bourbaki`` package, so we load it via
``importlib`` and exercise the ``_parse_proposer_models`` helper directly.
The flag is the only mechanism by which a benchmark run can opt into the
multi-provider parallel-fanout proposer, so the parser is on the
hot path even though the rest of the script touches the network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "run_minif2f_subset.py"
)


def _load_runner_module():
    """Load ``run_minif2f_subset`` from ``backend/scripts`` as a module.

    The script imports ``_runner_common`` (a sibling). We add the scripts
    directory to ``sys.path`` first so that import resolves.
    """
    import sys

    scripts_dir = _SCRIPT_PATH.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "run_minif2f_subset_under_test", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_proposer_models_two_entries_with_timeouts():
    """Two comma-separated specs, each with an ``@<seconds>`` suffix, must
    yield two :class:`ProposerConfig` with the parsed timeouts. Model
    strings may themselves contain colons (``glm:glm-5.1``); the parser
    must use ``@`` for the model/timeout split.
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


def test_parse_proposer_models_supports_pipe_separator():
    """Callers can use ``|`` as the entry separator when ``,`` is awkward
    for their shell. Mixed ``,`` / ``|`` works too.
    """
    mod = _load_runner_module()

    cfgs = mod._parse_proposer_models("glm:glm-5.1@90|ollama-cloud:kimi-k2.6@600")
    assert cfgs is not None
    assert len(cfgs) == 2
    assert cfgs[0].model == "glm:glm-5.1"
    assert cfgs[1].model == "ollama-cloud:kimi-k2.6"


def test_parse_proposer_models_bad_timeout_raises():
    """A non-float after ``@`` must raise rather than silently produce a
    ProposerConfig with the default timeout — silent fall-throughs are
    exactly the bug class issue #19 documents.
    """
    mod = _load_runner_module()

    with pytest.raises(ValueError, match="bad timeout"):
        mod._parse_proposer_models("glm:glm-5.1@not-a-number")


def test_parse_proposer_models_entry_without_timeout_uses_default():
    """Bare ``model`` (no ``@``) must default to ``ProposerConfig.llm_timeout``
    so callers who don't care about per-leg timing get the baseline 90s.
    """
    mod = _load_runner_module()
    from bourbaki.prover.parallel_proposer import ProposerConfig

    cfgs = mod._parse_proposer_models("glm:glm-5.1")
    assert cfgs is not None
    assert len(cfgs) == 1
    assert cfgs[0].model == "glm:glm-5.1"
    # Default value comes from ProposerConfig itself, not the parser.
    assert cfgs[0].llm_timeout == ProposerConfig.__dataclass_fields__["llm_timeout"].default

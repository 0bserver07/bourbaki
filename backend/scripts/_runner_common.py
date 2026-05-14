"""Shared helpers for standalone benchmark runner scripts.

This module is NOT part of bourbaki proper — it lives in ``backend/scripts/``
purely so the standalone runners under ``backend/scripts/run_*.py`` and
``backend/scripts/ab_*.py`` can share a small amount of bootstrap code
(logging, banners, env-var checks, path setup).

Keep this file self-contained: a runner script must work via
``python3 backend/scripts/<name>.py`` from the repo root.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Resolve repo root from this file (backend/scripts/_runner_common.py)
SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPTS_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
RESULTS_DIR = REPO_ROOT / ".bourbaki" / "benchmarks" / "results"
MINIF2F_DIR = REPO_ROOT / ".bourbaki" / "miniF2F-lean4"
PUTNAM_DIR = REPO_ROOT / ".bourbaki" / "putnam-bench"


def add_backend_to_path() -> None:
    """Ensure ``backend/`` is importable so ``from bourbaki...`` works.

    The runner scripts must work when invoked as ``python3 backend/scripts/X.py``
    from the repo root.  In that mode, only the script's own directory is on
    sys.path by default — not ``backend/``.  Insert it explicitly.
    """
    backend_str = str(BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def setup_logging(script_name: str) -> tuple[logging.Logger, Path]:
    """Wire logging to both stdout and a timestamped file under ``/tmp/bourbaki``.

    Returns
    -------
    (logger, log_path)
        The configured ``bourbaki.scripts.<script_name>`` logger and the path
        of the log file (printed in the banner so the user can ``tail -f`` it).
    """
    log_dir = Path("/tmp/bourbaki")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{script_name}-{stamp}.log"

    # Root logger configuration — apply only once.
    root = logging.getLogger()
    # Clear any pre-existing handlers (e.g. importing modules may have set them).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    stdout_handler.setLevel(logging.INFO)
    root.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "pydantic_ai", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(f"bourbaki.scripts.{script_name}"), log_path


def require_glm_api_key(logger: logging.Logger) -> str:
    """Read GLM_API_KEY from env or fail loudly.

    The proposer-builder-reviewer loop is fully GLM-driven, so a missing key
    will surface as a confusing pydantic_ai 401 deep in the loop.  Fail fast.
    """
    # The bourbaki config.export_api_keys() will copy .env values into env at
    # import time, but only if the variable isn't already set.  If the user
    # ran without sourcing the .env (as in CI-style invocation) and without
    # exporting the key, complain.
    key = os.environ.get("GLM_API_KEY")
    if not key:
        # Try to trigger bourbaki's auto-export from .env one more time.
        try:
            from bourbaki.config import export_api_keys
            export_api_keys()
            key = os.environ.get("GLM_API_KEY")
        except Exception:  # noqa: BLE001
            pass
    if not key:
        logger.error(
            "GLM_API_KEY is not set. Export it or add it to .env at the "
            "repo root before launching this script."
        )
        sys.exit(2)
    return key


def banner(
    logger: logging.Logger,
    script_name: str,
    config: dict[str, Any],
    log_path: Path,
    *,
    expected_wall_time: str | None = None,
    expected_result_glob: str | None = None,
) -> None:
    """Print a clean banner showing config, ETA, and the log/result paths."""
    bar = "=" * 72
    logger.info(bar)
    logger.info("Bourbaki standalone runner: %s", script_name)
    logger.info(bar)
    logger.info("Repo root:     %s", REPO_ROOT)
    logger.info("Log file:      %s", log_path)
    if expected_wall_time:
        logger.info("Expected wall: %s", expected_wall_time)
    if expected_result_glob:
        logger.info("Result file:   %s", expected_result_glob)
    logger.info("Config:")
    for k, v in config.items():
        logger.info("  %-26s %s", k + ":", v)
    logger.info(bar)


def print_per_source_breakdown(
    logger: logging.Logger,
    by_source: dict[str, dict[str, int]],
) -> None:
    """Pretty-print the per-source breakdown table."""
    if not by_source:
        return
    logger.info("Per-source breakdown:")
    width = max(len(s) for s in by_source) if by_source else 8
    for src, counts in sorted(by_source.items()):
        total = counts.get("total", 0)
        solved = counts.get("solved", 0)
        pct = 100 * solved / total if total else 0.0
        logger.info(
            "  %-*s %d/%d (%.1f%%)",
            width, src, solved, total, pct,
        )


def newest_result_json() -> Path | None:
    """Return the most recently mtime'd ``.bourbaki/benchmarks/results/*.json``.

    Used by view_result.py's ``--latest`` and by A/B scripts that need to
    locate the JSON the underlying runner just wrote (``run_minif2f`` and
    ``run_putnam`` save the file but don't return its path).
    """
    if not RESULTS_DIR.is_dir():
        return None
    candidates = sorted(
        RESULTS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# Default 35-problem stratified subset used by run_minif2f_subset.py
# (and as the basis for ab_pass_at_n.py).  Extracted from the
# 2026-03-19_1516 baseline run.
DEFAULT_SUBSET_BASELINE = (
    REPO_ROOT
    / ".bourbaki"
    / "benchmarks"
    / "results"
    / "2026-03-19_1516_minif2f_valid.json"
)


def load_default_subset_ids() -> list[str]:
    """Return the 35-problem stratified subset from the baseline result file.

    Raises FileNotFoundError if the baseline JSON is missing.
    """
    import json

    if not DEFAULT_SUBSET_BASELINE.is_file():
        raise FileNotFoundError(
            f"Baseline subset file not found at {DEFAULT_SUBSET_BASELINE}. "
            "Pass --ids or --from-file to specify problems explicitly."
        )
    data = json.loads(DEFAULT_SUBSET_BASELINE.read_text(encoding="utf-8"))
    return [r["problem_id"] for r in data["results"]]


# 8-problem hard subset for the mathlib_search A/B (issue #17): every
# problem here landed at attempts=0 on the May 9 35-problem run, so
# the loop never even produced a candidate proof under the baseline
# config.  These are the "hardest" of the 35.
AB_MATHLIB_DEFAULT_IDS = [
    "algebra_amgm_prod1toneq1_sum1tongeqn",
    "algebra_amgm_sqrtxymulxmyeqxpy_xpygeq4",
    "amc12a_2002_p21",
    "imo_1966_p5",
    "imo_1977_p5",
    "induction_divisibility_3div2tooddnp1",
    "mathd_algebra_510",
    "mathd_numbertheory_412",
]

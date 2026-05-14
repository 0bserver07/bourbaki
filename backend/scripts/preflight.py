#!/usr/bin/env python3
"""Pre-benchmark health check for Bourbaki long-running runs.

Run this BEFORE kicking off a 244-problem miniF2F or PutnamBench job —
silent slowdowns from a loaded box are the most common reason a recent
run regresses unexpectedly (see GitHub issue #19 for the worked example
that motivated this check).

Checks performed:

  1. **Load average.**  Warns if ``os.getloadavg()[0]`` > 4. Long
     proof runs are CPU-bound on lake + lean; a loaded box can stretch
     a healthy 30s ``lake env lean`` to several minutes, silently
     blowing through every per-call timeout in the chain.

  2. **Cold ``lake env lean`` compile.**  Writes a hello-world
     ``theorem hello : 1 = 1 := rfl`` (no Mathlib import) into the
     existing ``.bourbaki/lean-project`` and times its compile. If it
     takes > 120s for the bare theorem, the system is too loaded
     to run the loop reliably and we print "DO NOT START BENCHMARK
     NOW" with a non-zero exit code.

  3. **GLM_API_KEY + z.ai endpoint reachability.**  Confirms the
     Anthropic-compat endpoint is responding by sending a small
     "Hello" call through ``bourbaki.prover.proposer._resolve_model_object``.
     A 401/404 here would otherwise surface as a confusing pydantic_ai
     error several minutes into the benchmark.

  4. **``import Mathlib`` wall time in the REPL.**  The loop pays this
     once per process; if it exceeds the REPL's 300s ``ensure_initialized``
     timeout, the first problem will fail with a pipe-desync error.

Exit codes:
  0  — all checks passed; safe to start the benchmark.
  1  — at least one check failed; the reason is printed to stderr.
  2  — environment misconfiguration (e.g. ``GLM_API_KEY`` missing).

Usage:
  ``just preflight``           — default checks (about 30-60s + Mathlib init)
  ``python3 backend/scripts/preflight.py --skip-mathlib`` — skip the
       slow REPL Mathlib load when iterating on this script itself.
  ``python3 backend/scripts/preflight.py --help`` — list options.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make `bourbaki.*` importable when run as a script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _runner_common as common  # noqa: E402  (after sys.path edit)

common.add_backend_to_path()


# ---------------------------------------------------------------------------
# Tuning knobs — kept here so a future operator can rationalise the budgets.
# ---------------------------------------------------------------------------

LOAD_AVG_WARN_THRESHOLD = 4.0    # 1-min load average above this prints a warning
HELLO_WORLD_COMPILE_MAX_S = 120  # `lake env lean` of a no-import theorem above this is "system loaded"
GLM_HELLO_TIMEOUT_S = 30         # z.ai Anthropic-compat hello call
MATHLIB_IMPORT_MAX_S = 300       # matches lean_repl.ensure_initialized's own timeout


@dataclass
class CheckResult:
    """One preflight subcheck outcome."""
    name: str
    passed: bool
    detail: str
    is_warning: bool = False  # True = soft warning (still passes overall)
    duration_s: float = 0.0


@dataclass
class PreflightReport:
    """Aggregate outcome of all checks."""
    results: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> None:
        self.results.append(r)

    @property
    def failed(self) -> bool:
        return any(not r.passed and not r.is_warning for r in self.results)

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.is_warning]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("bourbaki.preflight")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Check 1 — system load
# ---------------------------------------------------------------------------

def check_load_average(logger: logging.Logger) -> CheckResult:
    start = time.monotonic()
    try:
        load1, load5, load15 = os.getloadavg()
    except (OSError, AttributeError):
        # getloadavg is POSIX-only; skip gracefully on platforms that
        # don't support it.
        return CheckResult(
            name="load-average",
            passed=True,
            detail="os.getloadavg() not available on this platform; skipped",
            is_warning=True,
            duration_s=time.monotonic() - start,
        )
    is_warning = load1 > LOAD_AVG_WARN_THRESHOLD
    detail = f"1-min={load1:.2f} 5-min={load5:.2f} 15-min={load15:.2f}"
    if is_warning:
        logger.warning(
            "load avg %.2f exceeds soft threshold %.1f — long runs may stretch.",
            load1, LOAD_AVG_WARN_THRESHOLD,
        )
    return CheckResult(
        name="load-average",
        passed=not is_warning,
        detail=detail,
        is_warning=is_warning,
        duration_s=time.monotonic() - start,
    )


# ---------------------------------------------------------------------------
# Check 2 — cold lake env lean hello-world
# ---------------------------------------------------------------------------

def check_hello_world_compile(logger: logging.Logger) -> CheckResult:
    """Time a ``lake env lean`` of a vanilla theorem (no Mathlib).

    Targets the existing ``.bourbaki/lean-project`` so the lake project
    layout is honoured. The file is written with a unique name and
    cleaned up afterwards. We measure the wall time of one ``lake env
    lean <file>`` invocation.
    """
    start = time.monotonic()

    # Resolve the lake project relative to repo root (the worktree).
    project_dir = common.REPO_ROOT / ".bourbaki" / "lean-project"
    if not project_dir.is_dir():
        return CheckResult(
            name="lake-env-lean-hello-world",
            passed=True,
            detail=(
                f"{project_dir} not present; skipped. "
                "Set up Lean before running the benchmark."
            ),
            is_warning=True,
            duration_s=time.monotonic() - start,
        )
    if shutil.which("lake") is None:
        return CheckResult(
            name="lake-env-lean-hello-world",
            passed=False,
            detail="`lake` not on PATH; install Lean toolchain via elan",
            duration_s=time.monotonic() - start,
        )

    test_name = f"_preflight_{secrets.token_hex(4)}.lean"
    test_path = project_dir / test_name
    try:
        test_path.write_text(
            "theorem hello : 1 = 1 := rfl\n", encoding="utf-8"
        )
    except OSError as exc:
        return CheckResult(
            name="lake-env-lean-hello-world",
            passed=False,
            detail=f"could not write probe file to {project_dir}: {exc}",
            duration_s=time.monotonic() - start,
        )

    try:
        compile_start = time.monotonic()
        proc = subprocess.run(
            ["lake", "env", "lean", str(test_path)],
            capture_output=True, text=True,
            cwd=str(project_dir),
            timeout=HELLO_WORLD_COMPILE_MAX_S + 30,
        )
        elapsed = time.monotonic() - compile_start
        if proc.returncode != 0:
            return CheckResult(
                name="lake-env-lean-hello-world",
                passed=False,
                detail=(
                    f"compile returned exit {proc.returncode}: "
                    f"{(proc.stderr or proc.stdout)[:200]}"
                ),
                duration_s=time.monotonic() - start,
            )
        if elapsed > HELLO_WORLD_COMPILE_MAX_S:
            logger.error(
                "lake env lean hello-world took %.1fs (> %ds cap). "
                "DO NOT START BENCHMARK NOW — system is too loaded.",
                elapsed, HELLO_WORLD_COMPILE_MAX_S,
            )
            return CheckResult(
                name="lake-env-lean-hello-world",
                passed=False,
                detail=(
                    f"compile took {elapsed:.1f}s (> {HELLO_WORLD_COMPILE_MAX_S}s cap). "
                    "DO NOT START BENCHMARK NOW."
                ),
                duration_s=time.monotonic() - start,
            )
        return CheckResult(
            name="lake-env-lean-hello-world",
            passed=True,
            detail=f"compiled in {elapsed:.1f}s",
            duration_s=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="lake-env-lean-hello-world",
            passed=False,
            detail=(
                f"compile killed after {HELLO_WORLD_COMPILE_MAX_S + 30}s. "
                "DO NOT START BENCHMARK NOW."
            ),
            duration_s=time.monotonic() - start,
        )
    finally:
        test_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Check 3 — GLM_API_KEY + z.ai reachability
# ---------------------------------------------------------------------------

async def check_glm_endpoint(logger: logging.Logger) -> CheckResult:
    """Send a one-token call through the z.ai Anthropic-compat endpoint.

    Reuses the prover's own model-resolver so we test the exact code
    path the loop will use. Catches missing keys, 401s, 404s, and DNS
    failures before the benchmark starts.
    """
    start = time.monotonic()

    # bourbaki.config.export_api_keys() copies GLM_API_KEY out of .env
    # into the environment on import; trigger it explicitly so this
    # check works under ``python3 backend/scripts/preflight.py`` even
    # without a sourced .env.
    try:
        from bourbaki.config import export_api_keys
        export_api_keys()
    except Exception:  # noqa: BLE001
        pass

    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        return CheckResult(
            name="glm-api-key",
            passed=False,
            detail=(
                "GLM_API_KEY not set; add it to .env at the repo root or "
                "export it before running."
            ),
            duration_s=time.monotonic() - start,
        )

    try:
        from pydantic_ai import Agent

        from bourbaki.prover.proposer import _resolve_model_object

        resolved = _resolve_model_object("glm:glm-5.1")
        agent: Agent[None, str] = Agent(resolved, output_type=str)
        await asyncio.wait_for(
            agent.run("Reply with the single word: OK"),
            timeout=GLM_HELLO_TIMEOUT_S,
        )
        return CheckResult(
            name="glm-endpoint",
            passed=True,
            detail=f"reachable (key starts {api_key[:4]}...)",
            duration_s=time.monotonic() - start,
        )
    except asyncio.TimeoutError:
        return CheckResult(
            name="glm-endpoint",
            passed=False,
            detail=(
                f"hello call exceeded {GLM_HELLO_TIMEOUT_S}s — z.ai endpoint "
                "is slow or unreachable"
            ),
            duration_s=time.monotonic() - start,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="glm-endpoint",
            passed=False,
            detail=f"hello call failed: {type(exc).__name__}: {exc}",
            duration_s=time.monotonic() - start,
        )


# ---------------------------------------------------------------------------
# Check 4 — Mathlib import in the REPL
# ---------------------------------------------------------------------------

async def check_mathlib_import(logger: logging.Logger) -> CheckResult:
    """Time the one-shot ``import Mathlib`` in a fresh REPL session.

    The loop pays this cost once per process, so the absolute wall time
    isn't itself a blocker — but if it's anywhere near the 300s budget
    in :func:`bourbaki.tools.lean_repl.LeanREPLSession.ensure_initialized`,
    the first benchmark problem will time out before the proposer even
    runs.
    """
    start = time.monotonic()

    try:
        from bourbaki.tools.lean_repl import LeanREPLSession, _find_repl_binary
    except ImportError as exc:
        return CheckResult(
            name="repl-mathlib-import",
            passed=False,
            detail=f"could not import LeanREPLSession: {exc}",
            duration_s=time.monotonic() - start,
        )

    if _find_repl_binary() is None:
        return CheckResult(
            name="repl-mathlib-import",
            passed=True,
            detail=(
                "lean4-repl binary not present; "
                "run scripts/setup-lean.sh before the benchmark"
            ),
            is_warning=True,
            duration_s=time.monotonic() - start,
        )

    session = LeanREPLSession(import_full_mathlib=True)
    try:
        try:
            await session.start()
        except RuntimeError as exc:
            return CheckResult(
                name="repl-mathlib-import",
                passed=False,
                detail=f"could not start REPL: {exc}",
                duration_s=time.monotonic() - start,
            )
        import_start = time.monotonic()
        await session.ensure_initialized()
        elapsed = time.monotonic() - import_start
        if not session._initialized:  # ensure_initialized warned but did not raise
            return CheckResult(
                name="repl-mathlib-import",
                passed=False,
                detail=(
                    f"REPL refused `import Mathlib` after {elapsed:.1f}s "
                    "(see stderr buffer in the REPL log)"
                ),
                duration_s=time.monotonic() - start,
            )
        is_warning = elapsed > (MATHLIB_IMPORT_MAX_S * 0.6)
        detail = f"loaded in {elapsed:.1f}s (cap {MATHLIB_IMPORT_MAX_S}s)"
        if is_warning:
            detail += " — close to the ensure_initialized cap"
        return CheckResult(
            name="repl-mathlib-import",
            passed=True,
            detail=detail,
            is_warning=is_warning,
            duration_s=time.monotonic() - start,
        )
    finally:
        await session.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preflight",
        description=(
            "Pre-benchmark health checks for Bourbaki. Runs four "
            "subchecks (load avg / lake env lean compile / GLM endpoint "
            "reachability / REPL Mathlib import) and exits non-zero if "
            "any of them suggest the system isn't ready for a long run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available checks:\n"
            "  load-average            (always run, cheap)\n"
            "  lake-env-lean-hello-world (skip with --skip-lake)\n"
            "  glm-endpoint            (skip with --skip-glm)\n"
            "  repl-mathlib-import     (skip with --skip-mathlib)\n\n"
            "Exits 0 if all checks pass, non-zero with details otherwise.\n"
        ),
    )
    parser.add_argument(
        "--skip-lake",
        action="store_true",
        help="Skip the lake env lean hello-world compile timing.",
    )
    parser.add_argument(
        "--skip-glm",
        action="store_true",
        help="Skip the GLM endpoint reachability check.",
    )
    parser.add_argument(
        "--skip-mathlib",
        action="store_true",
        help="Skip the REPL `import Mathlib` timing (saves ~20-90s).",
    )
    return parser.parse_args()


async def run_preflight(args: argparse.Namespace) -> PreflightReport:
    logger = _setup_logging()
    logger.info("Bourbaki preflight starting …")
    report = PreflightReport()

    # 1. Load average — always.
    report.add(check_load_average(logger))

    # 2. Lake env lean hello-world.
    if not args.skip_lake:
        report.add(check_hello_world_compile(logger))
    else:
        report.add(
            CheckResult(
                name="lake-env-lean-hello-world",
                passed=True,
                detail="skipped (--skip-lake)",
                is_warning=True,
            )
        )

    # 3. GLM endpoint.
    if not args.skip_glm:
        report.add(await check_glm_endpoint(logger))
    else:
        report.add(
            CheckResult(
                name="glm-endpoint",
                passed=True,
                detail="skipped (--skip-glm)",
                is_warning=True,
            )
        )

    # 4. Mathlib import.
    if not args.skip_mathlib:
        report.add(await check_mathlib_import(logger))
    else:
        report.add(
            CheckResult(
                name="repl-mathlib-import",
                passed=True,
                detail="skipped (--skip-mathlib)",
                is_warning=True,
            )
        )

    return report


def render_report(report: PreflightReport, logger: logging.Logger) -> None:
    logger.info("=" * 64)
    logger.info("Preflight results")
    logger.info("=" * 64)
    for r in report.results:
        if r.is_warning:
            # Soft warnings render as WARN regardless of `passed` —
            # `passed=False` on a warning means "below the soft
            # threshold but not a hard fail" (e.g. load average above
            # the soft cap but still runnable).
            tag = "WARN"
        elif not r.passed:
            tag = "FAIL"
        else:
            tag = "OK  "
        logger.info(
            "  [%s] %-30s %s (%.1fs)",
            tag, r.name, r.detail, r.duration_s,
        )
    logger.info("=" * 64)
    if report.failed:
        logger.error("Preflight FAILED — do not start the benchmark.")
    elif report.warnings:
        logger.warning("Preflight passed with warnings — review before running.")
    else:
        logger.info("Preflight passed — safe to start the benchmark.")


def main() -> int:
    args = parse_args()
    logger = _setup_logging()
    report = asyncio.run(run_preflight(args))
    render_report(report, logger)

    # Exit-code semantics:
    #   0 = clean
    #   1 = at least one hard failure
    #   2 = reserved for env-misconfig (currently mapped to 1; preserved
    #       for future structured failure routing)
    if report.failed:
        # Surface GLM-key-missing as the env-config exit code so
        # automation can react differently than to a perf failure.
        for r in report.results:
            if r.name == "glm-api-key" and not r.passed:
                return 2
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

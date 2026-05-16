#!/usr/bin/env python3
"""Single-iteration driver for the overnight pass-rate optimization loop.

This script performs ONE iteration of the meta-research loop:

1. Capture the current ``git rev-parse HEAD`` (the "prior" commit).
2. Run ``run_minif2f_subset.py`` with a fixed config to measure the
   working tree's current pass rate.
3. Parse the newest JSON in ``.bourbaki/benchmarks/results/`` for
   ``verified / total`` → ``pass_rate`` and ``total_time_seconds`` →
   ``duration_s``.
4. Compare to the best historical ``pass_rate`` in ``.bourbaki/results.tsv``.
5. Decide:
   - **First-ever run** (no ``results.tsv``): status=``baseline``.
   - **Improved or equal**: status=``kept``.  If ``prompts.py`` /
     ``strategies.py`` were modified in the working tree, stage them
     and commit (``feat(autoresearch): <description> [pass_rate -> NEW]``).
   - **Regressed**: status=``reverted``.  Restore any working-tree
     modifications to those two files via ``git checkout --``.
6. Append a tab-separated row to ``.bourbaki/results.tsv``.
7. Print a one-line summary.

The caller (an LLM coding agent) is expected to alternate between
editing ``backend/bourbaki/prover/prompts.py`` (or ``strategies.py``)
and invoking this script.

Tie-breaker decision (equal pass_rate): we treat ties as ``kept`` so
that the edit surface tends toward simpler / faster code over time.
If the agent shipped a prompt that's equally accurate but smaller,
keep it — that's the simplicity criterion baked into the loop.

Files this script is allowed to commit (when status=kept):
  ``backend/bourbaki/prover/prompts.py``
  ``backend/bourbaki/prover/strategies.py``

Files this script is allowed to revert (when status=reverted):
  same two paths.  No ``git reset --hard`` — the autoresearch
  infrastructure itself (this script, tests) must survive a revert.

Usage
-----
``python3 backend/scripts/autoresearch_step.py --description "..."``

A ``--dry-run`` flag does everything except commit / revert / append.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _runner_common as common  # noqa: E402

# Repo-relative paths the outer-loop agent is allowed to edit.  Anything
# outside this set is left untouched on revert (so the autoresearch
# script itself survives).
MUTABLE_PATHS: tuple[str, ...] = (
    "backend/bourbaki/prover/prompts.py",
    "backend/bourbaki/prover/strategies.py",
)

RESULTS_TSV = common.REPO_ROOT / ".bourbaki" / "results.tsv"
TSV_HEADER = (
    "timestamp\tcommit\tpass_rate\tn_problems\tduration_s\tdescription\tstatus\n"
)

# Default fanout when the env var isn't set.  Two-leg race that has
# historically been a good cost/quality tradeoff on the 35-problem
# subset.  See AUTORESEARCH.md §3 for the full discussion.
DEFAULT_PROPOSER_MODELS = "glm:glm-5.1@90,ollama-cloud:gpt-oss:120b@180"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one iteration of the outer-loop pass-rate optimization. "
            "Measures the working tree's verified pass rate on the 35-problem "
            "stratified subset, compares to history, and either commits the "
            "current edit (if it improved) or reverts mutable files (if it "
            "regressed)."
        ),
    )
    parser.add_argument(
        "--subset",
        default="minif2f-35",
        help=(
            "Subset identifier.  Currently only 'minif2f-35' is supported "
            "(the 35-problem stratified valid split)."
        ),
    )
    parser.add_argument(
        "--problems",
        default=None,
        help=(
            "Comma-separated problem IDs to override the subset default. "
            "Useful for fast smoke tests."
        ),
    )
    parser.add_argument(
        "--proposer-models",
        default=None,
        help=(
            "Comma- or pipe-separated proposer fanout spec, identical "
            "format to run_minif2f_subset.py's --proposer-models.  "
            "Falls back to $BOURBAKI_AUTORESEARCH_MODELS then to "
            f"{DEFAULT_PROPOSER_MODELS!r}."
        ),
    )
    parser.add_argument(
        "--race-mode",
        choices=["first_completed", "first_verified"],
        default="first_verified",
        help="Race semantics for the proposer fanout (default: first_verified).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Loop max iterations per problem (default: 5).",
    )
    parser.add_argument(
        "--per-problem-timeout",
        type=int,
        default=1500,
        help=(
            "Per-problem timeout passed to the inner runner (default: 1500s). "
            "Bumped from 300s — with race_mode=first_verified, each leg waits "
            "60-180s for lean_prover, so 300s is below the floor for most "
            "fanout configurations."
        ),
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=3600,
        help=(
            "Wall-clock cap on the benchmark subprocess (default: 3600s). "
            "On timeout the run is discarded — results.tsv is NOT appended."
        ),
    )
    parser.add_argument(
        "--description",
        required=True,
        help=(
            "Human-readable description of the hypothesis being tested.  "
            "Recorded verbatim in results.tsv and in the auto-commit message."
        ),
    )
    parser.add_argument(
        "--results-tsv",
        type=Path,
        default=None,
        help=(
            "Override the path to results.tsv.  Default: .bourbaki/results.tsv "
            "at repo root (gitignored).  Mainly for tests."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Do everything except commit / revert / append to results.tsv.  "
            "Print the one-line summary so callers can see what would happen."
        ),
    )
    return parser.parse_args(argv)


def _setup_logger() -> tuple[logging.Logger, Path]:
    """Wire stdout + ``/tmp/bourbaki-autoresearch/<stamp>.log``."""
    log_dir = Path("/tmp/bourbaki-autoresearch")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"autoresearch_step-{stamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("bourbaki.scripts.autoresearch_step")
    logger.setLevel(logging.INFO)
    # Idempotent: if invoked twice in one process, don't double-attach.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    logger.addHandler(out)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger, log_path


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    logger: logging.Logger,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``git`` with the given args, captured.  Raises on non-zero by default."""
    logger.debug("git %s", " ".join(args))
    cp = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and cp.returncode != 0:
        logger.error(
            "git %s failed (exit=%d): stdout=%r stderr=%r",
            " ".join(args), cp.returncode, cp.stdout, cp.stderr,
        )
        raise RuntimeError(f"git {args[0]} failed")
    return cp


def _head_sha(repo_root: Path, logger: logging.Logger) -> str:
    cp = _run_git(["rev-parse", "HEAD"], cwd=repo_root, logger=logger)
    return cp.stdout.strip()


def _modified_mutable_paths(repo_root: Path, logger: logging.Logger) -> list[str]:
    """Return the subset of ``MUTABLE_PATHS`` that have working-tree edits.

    Uses ``git status --porcelain=v1`` so the parse is stable across
    git versions.  Both unstaged and staged modifications are counted —
    if the user pre-staged a prompts.py edit, we still want to treat
    it as an experiment.
    """
    cp = _run_git(
        ["status", "--porcelain=v1", "--"] + list(MUTABLE_PATHS),
        cwd=repo_root,
        logger=logger,
    )
    modified: list[str] = []
    for line in cp.stdout.splitlines():
        if len(line) < 4:
            continue
        # Format: "XY path".  Either column 1 (staged) or column 2
        # (unstaged) being non-space means the path has an edit.
        x, y = line[0], line[1]
        path = line[3:]
        if x == "?" or y == "?":
            # Untracked file — only counts for strategies.py (which
            # doesn't exist yet in master).
            if path in MUTABLE_PATHS:
                modified.append(path)
            continue
        if x != " " or y != " ":
            if path in MUTABLE_PATHS:
                modified.append(path)
    return modified


def _read_history(tsv_path: Path) -> tuple[list[dict[str, str]], bool]:
    """Read existing TSV rows.  Returns (rows, header_present)."""
    if not tsv_path.is_file():
        return [], False
    text = tsv_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], False
    header_present = lines[0].split("\t")[0] == "timestamp"
    body = lines[1:] if header_present else lines
    rows: list[dict[str, str]] = []
    fields = ("timestamp", "commit", "pass_rate", "n_problems",
              "duration_s", "description", "status")
    for ln in body:
        parts = ln.split("\t")
        if len(parts) < len(fields):
            # Pad short rows so we don't crash on a malformed line —
            # the loop should be resilient to a hand-edited TSV.
            parts = parts + [""] * (len(fields) - len(parts))
        rows.append(dict(zip(fields, parts, strict=False)))
    return rows, header_present


def _best_pass_rate(rows: list[dict[str, str]]) -> float | None:
    """Highest historical pass_rate (over baseline + kept rows).

    Reverted rows are still kept in the TSV for auditability but we
    don't compare against them: by definition they're worse than the
    best-so-far at the time they were recorded.
    """
    best: float | None = None
    for row in rows:
        try:
            pr = float(row["pass_rate"])
        except (KeyError, ValueError):
            continue
        if best is None or pr > best:
            best = pr
    return best


def _resolve_problem_ids(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> list[str]:
    """Map ``--problems`` / ``--subset`` to a list of miniF2F IDs."""
    if args.problems:
        ids = [pid.strip() for pid in args.problems.split(",") if pid.strip()]
        if not ids:
            logger.error("--problems given but no IDs parsed")
            sys.exit(2)
        return ids
    if args.subset != "minif2f-35":
        logger.error("Unknown --subset %r (only 'minif2f-35' supported)", args.subset)
        sys.exit(2)
    try:
        return common.load_default_subset_ids()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(2)


def _resolve_proposer_models(args: argparse.Namespace) -> str:
    """``--proposer-models`` wins, then env, then the hard-coded default."""
    if args.proposer_models:
        return args.proposer_models
    env = os.environ.get("BOURBAKI_AUTORESEARCH_MODELS", "").strip()
    if env:
        return env
    return DEFAULT_PROPOSER_MODELS


def _run_benchmark(
    *,
    problem_ids: list[str],
    proposer_models: str,
    race_mode: str,
    max_iter: int,
    per_problem_timeout: int,
    benchmark_timeout: int,
    logger: logging.Logger,
    repo_root: Path,
) -> tuple[bool, Path | None, str]:
    """Spawn ``run_minif2f_subset.py``.  Capture stdout to a temp log.

    Returns ``(ok, log_path, message)``.  ``ok`` is True iff the
    subprocess returned 0 within the wall-clock budget.
    """
    runner = repo_root / "backend" / "scripts" / "run_minif2f_subset.py"
    if not runner.is_file():
        return False, None, f"runner not found at {runner}"

    ids_arg = ",".join(problem_ids)
    cmd = [
        sys.executable,
        str(runner),
        "--ids", ids_arg,
        "--proposer-models", proposer_models,
        "--race-mode", race_mode,
        "--max-iter", str(max_iter),
        "--timeout", str(per_problem_timeout),
    ]

    log_dir = Path("/tmp/bourbaki-autoresearch")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bench_log = log_dir / f"benchmark-{stamp}.log"

    logger.info("Spawning benchmark (timeout=%ds): %s", benchmark_timeout, " ".join(cmd))
    logger.info("Benchmark stdout/stderr -> %s", bench_log)

    try:
        with bench_log.open("w", encoding="utf-8") as fh:
            cp = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=benchmark_timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return False, bench_log, f"benchmark timed out after {benchmark_timeout}s"
    except OSError as e:
        return False, bench_log, f"benchmark spawn failed: {e}"

    if cp.returncode != 0:
        return False, bench_log, f"benchmark exited non-zero ({cp.returncode})"
    return True, bench_log, "ok"


def _parse_result_json(
    json_path: Path,
    *,
    started_before: float,
) -> dict[str, Any] | None:
    """Load + validate the newest result JSON.

    ``started_before`` is the wall-clock (``time.time()``) immediately
    before launching the subprocess.  We refuse to pick up a JSON
    older than that — a stale newest-mtime file from a prior run
    would silently poison the loop otherwise.
    """
    if not json_path.is_file():
        return None
    if json_path.stat().st_mtime < started_before:
        return None
    try:
        data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "verified" not in data or "total" not in data:
        return None
    return data


def _pass_rate(data: dict[str, Any]) -> float:
    total = int(data.get("total", 0))
    if total <= 0:
        return 0.0
    return float(data.get("verified", 0)) / float(total)


def _append_row(
    tsv_path: Path,
    *,
    header_present: bool,
    timestamp: str,
    commit: str,
    pass_rate: float,
    n_problems: int,
    duration_s: float,
    description: str,
    status: str,
) -> None:
    """Append one row + header (if absent) to ``tsv_path``."""
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not header_present and not tsv_path.exists()
    # Description must not contain tabs/newlines — sanitize.
    desc_clean = description.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    row = (
        f"{timestamp}\t{commit}\t{pass_rate:.6f}\t{n_problems}\t"
        f"{duration_s:.3f}\t{desc_clean}\t{status}\n"
    )
    with tsv_path.open("a", encoding="utf-8") as fh:
        if write_header:
            fh.write(TSV_HEADER)
        fh.write(row)


def _commit_kept(
    repo_root: Path,
    modified: list[str],
    description: str,
    pass_rate: float,
    logger: logging.Logger,
) -> str:
    """Stage + commit the experiment.  Returns the new commit SHA."""
    # Stage exactly the mutable paths that changed — don't ``git add .``
    # so the script-itself and any other in-flight work stay out of
    # the auto-commit.
    _run_git(["add", "--"] + modified, cwd=repo_root, logger=logger)
    desc_oneline = description.replace("\n", " ").replace("\r", " ").strip()
    pct = pass_rate * 100.0
    msg = f"feat(autoresearch): {desc_oneline} [pass_rate -> {pct:.2f}%]"
    _run_git(["commit", "-m", msg], cwd=repo_root, logger=logger)
    new_sha = _head_sha(repo_root, logger)
    return new_sha


def _revert(repo_root: Path, modified: list[str], logger: logging.Logger) -> None:
    """Restore mutable working-tree paths to HEAD.

    Tracked files: ``git checkout HEAD -- <path>`` resets the working
    copy + index (in case the agent pre-staged the file).
    Untracked files: ``git clean -f -- <path>`` removes the file.
    """
    if not modified:
        return
    # Probe each path: tracked vs untracked.  Tracked = ``git ls-files
    # --error-unmatch`` returns 0; untracked = non-zero.
    tracked: list[str] = []
    untracked: list[str] = []
    for path in modified:
        cp = _run_git(
            ["ls-files", "--error-unmatch", "--", path],
            cwd=repo_root, logger=logger, check=False,
        )
        if cp.returncode == 0:
            tracked.append(path)
        else:
            untracked.append(path)

    if tracked:
        _run_git(["checkout", "HEAD", "--"] + tracked, cwd=repo_root, logger=logger)
    if untracked:
        # ``-f`` is required by clean to actually delete; ``--`` separates
        # paths from flags.  We restrict to the named paths so nothing else
        # in the working tree gets wiped.
        _run_git(["clean", "-f", "--"] + untracked, cwd=repo_root, logger=logger)


def _emit_summary(
    *,
    iteration_num: int,
    pass_rate: float,
    best: float | None,
    status: str,
    commit: str,
    description: str,
    dry_run: bool,
) -> str:
    """Build the canonical one-line summary string."""
    best_str = f"{best * 100:.2f}%" if best is not None else "none"
    dry = " [DRY-RUN]" if dry_run else ""
    return (
        f"[autoresearch] iter {iteration_num}: "
        f"pass_rate={pass_rate * 100:.2f}% (best={best_str}) "
        f"status={status} commit={commit[:12]} "
        f"description={description!r}{dry}"
    )


def _resolve_tsv_path(args: argparse.Namespace) -> Path:
    """Honor ``--results-tsv`` (mostly for tests) else the default location."""
    if args.results_tsv is not None:
        return args.results_tsv.expanduser().resolve()
    return RESULTS_TSV


def main(argv: list[str] | None = None) -> int:  # noqa: C901 (top-level driver)
    args = parse_args(argv)
    logger, log_path = _setup_logger()
    logger.info("Autoresearch step log: %s", log_path)

    repo_root = common.REPO_ROOT
    tsv_path = _resolve_tsv_path(args)

    # 1. Read history + working-tree state.
    rows, header_present = _read_history(tsv_path)
    iteration_num = len(rows) + 1
    best = _best_pass_rate(rows)
    prior_sha = _head_sha(repo_root, logger)
    modified = _modified_mutable_paths(repo_root, logger)
    logger.info(
        "Prior HEAD=%s.  Modified mutable paths: %s",
        prior_sha[:12], modified or "<none>",
    )

    # 2. Resolve problem set + proposer fanout + spawn benchmark.
    problem_ids = _resolve_problem_ids(args, logger)
    proposer_models = _resolve_proposer_models(args)
    logger.info(
        "Running %d problems with proposer fanout %r race=%s",
        len(problem_ids), proposer_models, args.race_mode,
    )

    # Wall-clock floor for "newest result JSON belongs to this run."
    started_before = datetime.now(tz=timezone.utc).timestamp()

    ok, bench_log, msg = _run_benchmark(
        problem_ids=problem_ids,
        proposer_models=proposer_models,
        race_mode=args.race_mode,
        max_iter=args.max_iter,
        per_problem_timeout=args.per_problem_timeout,
        benchmark_timeout=args.benchmark_timeout,
        logger=logger,
        repo_root=repo_root,
    )
    if not ok:
        logger.error(
            "Benchmark did not complete (%s).  Refusing to append to %s "
            "or to commit/revert.  Log: %s",
            msg, tsv_path, bench_log,
        )
        return 1

    # 3. Locate + parse the newest result JSON.
    newest = common.newest_result_json()
    if newest is None:
        logger.error("No result JSONs found under %s after benchmark", common.RESULTS_DIR)
        return 1
    data = _parse_result_json(newest, started_before=started_before)
    if data is None:
        logger.error(
            "Newest result JSON %s is malformed, stale, or missing required keys; "
            "skipping commit/revert/append",
            newest,
        )
        return 1
    pass_rate = _pass_rate(data)
    n_problems = int(data.get("total", 0))
    duration_s = float(data.get("total_time_seconds", 0.0))
    logger.info(
        "Result JSON %s: verified=%d/%d pass_rate=%.4f duration=%.1fs",
        newest, data.get("verified", 0), n_problems, pass_rate, duration_s,
    )

    # 4. Decide status.  Three branches:
    #    - first row ever in TSV (no best): baseline
    #    - pass_rate >= best: kept  (ties go to "kept" to favor smaller code)
    #    - pass_rate < best: reverted
    if best is None:
        status = "baseline"
    elif pass_rate >= best:
        status = "kept"
    else:
        status = "reverted"

    # 5. Side-effects (commit / revert) — skipped on --dry-run.
    new_sha = prior_sha
    if args.dry_run:
        logger.info("DRY-RUN: skipping git side-effects and TSV append")
    else:
        if status == "kept" and modified:
            try:
                new_sha = _commit_kept(
                    repo_root, modified, args.description, pass_rate, logger,
                )
                logger.info("Committed experiment as %s", new_sha[:12])
            except RuntimeError:
                logger.exception("Failed to commit experiment")
                return 2
        elif status == "reverted":
            try:
                _revert(repo_root, modified, logger)
                logger.info("Reverted mutable files: %s", modified or "<none>")
            except RuntimeError:
                logger.exception("Failed to revert mutable files")
                return 2
        # baseline + kept-without-modifications: no git op, just append.

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _append_row(
            tsv_path,
            header_present=header_present,
            timestamp=timestamp,
            commit=new_sha,
            pass_rate=pass_rate,
            n_problems=n_problems,
            duration_s=duration_s,
            description=args.description,
            status=status,
        )
        logger.info("Appended row to %s", tsv_path)

    summary = _emit_summary(
        iteration_num=iteration_num,
        pass_rate=pass_rate,
        best=best,
        status=status,
        commit=new_sha,
        description=args.description,
        dry_run=args.dry_run,
    )
    print(summary)
    logger.info("%s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

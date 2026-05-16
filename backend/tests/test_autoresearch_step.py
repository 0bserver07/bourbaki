"""Tests for ``backend/scripts/autoresearch_step.py``.

These tests work by:

1. Building a self-contained git repo + ``results.tsv`` + result-JSON
   directory inside a ``tempfile.TemporaryDirectory``.
2. Monkey-patching ``_runner_common.REPO_ROOT`` and friends so the
   script's path constants point at the temp tree.
3. Stubbing ``subprocess.run`` so the "benchmark" call writes a
   synthetic result JSON and returns whatever exit code the test
   case wants.

This keeps every test sub-second and ensures we never spawn the real
benchmark (which would need GLM_API_KEY + Lean + the live network).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "autoresearch_step.py"
)


def _load_script() -> Any:
    """Import ``autoresearch_step`` as a module under a unique name.

    The script lives outside ``bourbaki`` so we go through importlib.
    We also drop ``backend/scripts`` on sys.path first because the
    script does ``import _runner_common`` (a sibling).
    """
    scripts_dir = _SCRIPT_PATH.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "autoresearch_step_under_test", _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    """Run git in a tmp repo with deterministic identity."""
    env_args = [
        "-c", "user.email=test@example.com",
        "-c", "user.name=Test",
        "-c", "commit.gpgsign=false",
    ]
    cp = subprocess.run(
        ["git", *env_args, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"git {args} failed in {cwd}: {cp.stdout!r} / {cp.stderr!r}"
        )
    return cp.stdout


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A fresh git repo with the mutable-paths layout pre-created.

    Mirrors enough of the real tree that the script's ``MUTABLE_PATHS``
    constants resolve.  Adds a baseline result JSON so
    ``load_default_subset_ids()`` doesn't need to be patched.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "master"], cwd=repo)

    # Pre-create the mutable file the script knows about.
    prompts = repo / "backend" / "bourbaki" / "prover" / "prompts.py"
    prompts.parent.mkdir(parents=True)
    prompts.write_text('PROPOSER_SYSTEM_PROMPT = "v1"\n', encoding="utf-8")

    # A throwaway runner so subprocess.run can resolve the path.
    runner = repo / "backend" / "scripts" / "run_minif2f_subset.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("# stub — patched in tests\n", encoding="utf-8")

    # Pre-make the results dir and a baseline JSON whose ``results[]``
    # carries problem IDs the loader can read.
    results_dir = repo / ".bourbaki" / "benchmarks" / "results"
    results_dir.mkdir(parents=True)
    baseline = results_dir / "2026-03-19_1516_minif2f_valid.json"
    baseline.write_text(
        json.dumps({
            "total": 2,
            "verified": 0,
            "total_time_seconds": 0.0,
            "results": [
                {"problem_id": "mathd_smoke_1"},
                {"problem_id": "mathd_smoke_2"},
            ],
        }),
        encoding="utf-8",
    )

    _git(["add", "."], cwd=repo)
    _git(["commit", "-q", "-m", "init"], cwd=repo)
    return repo


@pytest.fixture
def script(tmp_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the script and re-point its repo-root constants at ``tmp_repo``.

    The script imports ``_runner_common`` and caches absolute paths at
    import time.  We patch those module attributes so every helper
    (newest_result_json, load_default_subset_ids, RESULTS_TSV) sees
    the tmp tree instead of the real ``.bourbaki/``.
    """
    mod = _load_script()
    common = sys.modules["_runner_common"]

    monkeypatch.setattr(common, "REPO_ROOT", tmp_repo, raising=True)
    monkeypatch.setattr(
        common, "RESULTS_DIR",
        tmp_repo / ".bourbaki" / "benchmarks" / "results",
        raising=True,
    )
    monkeypatch.setattr(
        common, "DEFAULT_SUBSET_BASELINE",
        tmp_repo / ".bourbaki" / "benchmarks" / "results"
        / "2026-03-19_1516_minif2f_valid.json",
        raising=True,
    )
    monkeypatch.setattr(
        mod, "RESULTS_TSV", tmp_repo / ".bourbaki" / "results.tsv", raising=True,
    )
    return mod


def _stub_benchmark(
    *,
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    pass_rate: float,
    n_problems: int = 2,
    duration: float = 12.5,
    behavior: str = "ok",
) -> None:
    """Patch subprocess.run so the script's "benchmark" call is a no-op.

    The patch only short-circuits invocations that look like the
    runner script — git calls still go through to the real binary,
    which is what we want (the tmp_repo is a real git repo).

    Behaviors:
      * ``ok``      — write a result JSON, exit 0
      * ``timeout`` — raise TimeoutExpired
      * ``error``   — exit non-zero
      * ``bad_json``— write a malformed JSON file, exit 0
    """
    real_run = subprocess.run
    results_dir = repo_root / ".bourbaki" / "benchmarks" / "results"

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        # We only short-circuit the benchmark subprocess.  git invocations
        # are routed through subprocess.run too — let them through.
        is_runner = (
            isinstance(cmd, list) and len(cmd) >= 2
            and isinstance(cmd[1], str)
            and cmd[1].endswith("run_minif2f_subset.py")
        )
        if not is_runner:
            return real_run(cmd, **kwargs)

        if behavior == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

        if behavior == "ok" or behavior == "bad_json":
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            path = results_dir / f"{stamp}_minif2f_valid.json"
            if behavior == "bad_json":
                path.write_text("{not valid json", encoding="utf-8")
            else:
                verified = int(round(pass_rate * n_problems))
                payload: dict[str, Any] = {
                    "total": n_problems,
                    "verified": verified,
                    "total_time_seconds": duration,
                    "results": [],
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if behavior == "error":
            return subprocess.CompletedProcess(cmd, 7, "", "boom")

        raise AssertionError(f"unknown behavior {behavior!r}")

    monkeypatch.setattr(subprocess, "run", fake_run)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_run_writes_baseline_row(
    tmp_repo: Path,
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No ``results.tsv``: status=baseline, header + 1 row written."""
    _stub_benchmark(monkeypatch=monkeypatch, repo_root=tmp_repo, pass_rate=0.5)

    rc = script.main([
        "--description", "first run",
        "--problems", "mathd_smoke_1,mathd_smoke_2",
    ])
    assert rc == 0

    tsv = tmp_repo / ".bourbaki" / "results.tsv"
    assert tsv.is_file()
    lines = tsv.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp\tcommit\tpass_rate")
    assert len(lines) == 2  # header + 1 data row
    row = lines[1].split("\t")
    assert row[5] == "first run"
    assert row[6] == "baseline"
    out = capsys.readouterr().out
    assert "status=baseline" in out


def test_improvement_keeps_and_commits_modified_prompts(
    tmp_repo: Path,
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pass_rate > best, status=kept and the prompts edit is committed."""
    # Seed history with a baseline row at 50%.
    tsv = tmp_repo / ".bourbaki" / "results.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text(
        "timestamp\tcommit\tpass_rate\tn_problems\tduration_s\tdescription\tstatus\n"
        "2026-05-15T00:00:00Z\tdeadbeef\t0.500000\t2\t10.0\tseed\tbaseline\n",
        encoding="utf-8",
    )

    # Make a working-tree edit to prompts.py.
    prompts = tmp_repo / "backend" / "bourbaki" / "prover" / "prompts.py"
    prompts.write_text('PROPOSER_SYSTEM_PROMPT = "v2 (better)"\n', encoding="utf-8")

    head_before = _git(["rev-parse", "HEAD"], cwd=tmp_repo).strip()
    _stub_benchmark(monkeypatch=monkeypatch, repo_root=tmp_repo, pass_rate=1.0)

    rc = script.main([
        "--description", "tighter system prompt",
        "--problems", "mathd_smoke_1,mathd_smoke_2",
    ])
    assert rc == 0

    head_after = _git(["rev-parse", "HEAD"], cwd=tmp_repo).strip()
    assert head_after != head_before, "experiment should have been committed"

    # The new HEAD's commit message must contain the description + the
    # new pass-rate stamp.
    msg = _git(["log", "-1", "--format=%s"], cwd=tmp_repo).strip()
    assert "tighter system prompt" in msg
    assert "pass_rate" in msg

    rows = tsv.read_text(encoding="utf-8").splitlines()
    assert rows[-1].split("\t")[-1] == "kept"
    # The committed SHA in the TSV must match the new HEAD, not the old.
    assert rows[-1].split("\t")[1] == head_after


def test_regression_reverts_modified_prompts(
    tmp_repo: Path,
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pass_rate < best, status=reverted and prompts.py is restored."""
    tsv = tmp_repo / ".bourbaki" / "results.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text(
        "timestamp\tcommit\tpass_rate\tn_problems\tduration_s\tdescription\tstatus\n"
        "2026-05-15T00:00:00Z\tdeadbeef\t1.000000\t2\t10.0\tbest\tbaseline\n",
        encoding="utf-8",
    )

    prompts = tmp_repo / "backend" / "bourbaki" / "prover" / "prompts.py"
    original = prompts.read_text(encoding="utf-8")
    prompts.write_text("PROPOSER_SYSTEM_PROMPT = 'BAD'\n", encoding="utf-8")

    head_before = _git(["rev-parse", "HEAD"], cwd=tmp_repo).strip()
    _stub_benchmark(monkeypatch=monkeypatch, repo_root=tmp_repo, pass_rate=0.5)

    rc = script.main([
        "--description", "tried something dumb",
        "--problems", "mathd_smoke_1,mathd_smoke_2",
    ])
    assert rc == 0

    head_after = _git(["rev-parse", "HEAD"], cwd=tmp_repo).strip()
    assert head_after == head_before, "regression must not commit"
    assert prompts.read_text(encoding="utf-8") == original, "prompts.py restored"
    rows = tsv.read_text(encoding="utf-8").splitlines()
    assert rows[-1].split("\t")[-1] == "reverted"


def test_tie_is_kept_to_favor_simpler_edits(
    tmp_repo: Path,
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal pass_rate ties to ``kept``.

    Documented tie-breaker: when a refactor matches the prior best
    exactly, treat it as a win.  Rationale: the agent is encouraged
    to ship smaller / faster edits, and a tie at lower complexity is
    the explicit simplicity-criterion outcome.
    """
    tsv = tmp_repo / ".bourbaki" / "results.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text(
        "timestamp\tcommit\tpass_rate\tn_problems\tduration_s\tdescription\tstatus\n"
        "2026-05-15T00:00:00Z\tdeadbeef\t0.500000\t2\t10.0\tseed\tbaseline\n",
        encoding="utf-8",
    )

    prompts = tmp_repo / "backend" / "bourbaki" / "prover" / "prompts.py"
    prompts.write_text("PROPOSER_SYSTEM_PROMPT = 'shorter'\n", encoding="utf-8")

    _stub_benchmark(monkeypatch=monkeypatch, repo_root=tmp_repo, pass_rate=0.5)
    rc = script.main([
        "--description", "same accuracy, shorter prompt",
        "--problems", "mathd_smoke_1,mathd_smoke_2",
    ])
    assert rc == 0

    rows = tsv.read_text(encoding="utf-8").splitlines()
    assert rows[-1].split("\t")[-1] == "kept"


def test_benchmark_timeout_does_not_poison_tsv(
    tmp_repo: Path,
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TimeoutExpired must surface as a non-zero exit with no TSV append."""
    tsv = tmp_repo / ".bourbaki" / "results.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text(
        "timestamp\tcommit\tpass_rate\tn_problems\tduration_s\tdescription\tstatus\n"
        "2026-05-15T00:00:00Z\tdeadbeef\t0.500000\t2\t10.0\tseed\tbaseline\n",
        encoding="utf-8",
    )
    before = tsv.read_text(encoding="utf-8")

    _stub_benchmark(
        monkeypatch=monkeypatch, repo_root=tmp_repo,
        pass_rate=0.0, behavior="timeout",
    )

    rc = script.main([
        "--description", "would have hung",
        "--problems", "mathd_smoke_1,mathd_smoke_2",
    ])
    assert rc != 0
    assert tsv.read_text(encoding="utf-8") == before, "TSV unchanged on timeout"


def test_malformed_result_json_is_skipped(
    tmp_repo: Path,
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark exits 0 but JSON is malformed → non-zero exit, no TSV row."""
    tsv = tmp_repo / ".bourbaki" / "results.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_text(
        "timestamp\tcommit\tpass_rate\tn_problems\tduration_s\tdescription\tstatus\n"
        "2026-05-15T00:00:00Z\tdeadbeef\t0.500000\t2\t10.0\tseed\tbaseline\n",
        encoding="utf-8",
    )
    before = tsv.read_text(encoding="utf-8")

    _stub_benchmark(
        monkeypatch=monkeypatch, repo_root=tmp_repo,
        pass_rate=0.0, behavior="bad_json",
    )

    rc = script.main([
        "--description", "would have parsed badly",
        "--problems", "mathd_smoke_1,mathd_smoke_2",
    ])
    assert rc != 0
    assert tsv.read_text(encoding="utf-8") == before


def test_dry_run_skips_side_effects(
    tmp_repo: Path,
    script: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` must NOT commit, NOT revert, NOT append to TSV.

    But it MUST still print the one-line summary so the caller can
    eyeball what would have happened.
    """
    tsv = tmp_repo / ".bourbaki" / "results.tsv"

    prompts = tmp_repo / "backend" / "bourbaki" / "prover" / "prompts.py"
    prompts.write_text("PROPOSER_SYSTEM_PROMPT = 'v2'\n", encoding="utf-8")
    head_before = _git(["rev-parse", "HEAD"], cwd=tmp_repo).strip()

    _stub_benchmark(monkeypatch=monkeypatch, repo_root=tmp_repo, pass_rate=1.0)
    rc = script.main([
        "--description", "dry test",
        "--problems", "mathd_smoke_1,mathd_smoke_2",
        "--dry-run",
    ])
    assert rc == 0

    assert not tsv.exists(), "TSV must not be created in --dry-run"
    head_after = _git(["rev-parse", "HEAD"], cwd=tmp_repo).strip()
    assert head_after == head_before
    assert prompts.read_text(encoding="utf-8") == "PROPOSER_SYSTEM_PROMPT = 'v2'\n"
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "status=baseline" in out


def test_description_is_required() -> None:
    """``argparse`` should fail (exit 2) when ``--description`` is omitted."""
    mod = _load_script()
    with pytest.raises(SystemExit) as exc:
        mod.parse_args([])
    assert exc.value.code == 2

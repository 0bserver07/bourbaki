"""Read-only helpers for the benchmark results JSON archive.

This module reads files under ``.bourbaki/benchmarks/results/`` (or any
directory configured via ``RESULTS_DIR``). It never mutates them. The
public API is small on purpose:

- :func:`list_results` — summaries for every result file.
- :func:`load_result` — load a single result by path.
- :func:`latest_result` — most recent matching summary.
- :func:`diff_results` — per-problem set comparison between two results.
- :func:`by_source` — aggregate verified/total/pass-rate per source.

Each result file is a JSON object produced by ``run_minif2f`` or
``run_putnam`` (see :mod:`bourbaki.benchmarks.minif2f` /
:mod:`bourbaki.benchmarks.putnam`). Schemas have drifted over time;
the helpers below extract a common set of fields and tolerate missing
keys instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default results directory: <repo>/.bourbaki/benchmarks/results/
_THIS_FILE = Path(__file__).resolve()
# results_db.py lives at backend/bourbaki/benchmarks/results_db.py
# Repo root is three levels above backend/bourbaki/benchmarks/ -> ../../..
_REPO_ROOT = _THIS_FILE.parents[3]
RESULTS_DIR = _REPO_ROOT / ".bourbaki" / "benchmarks" / "results"


def _detect_benchmark(filename: str, data: dict[str, Any]) -> str:
    """Best-effort: 'minif2f' | 'putnam' | 'verified' | 'other'."""
    explicit = data.get("benchmark")
    if isinstance(explicit, str) and explicit:
        return explicit
    lower = filename.lower()
    if "putnam" in lower:
        return "putnam"
    if "verified_unknown" in lower:
        return "verified"
    if "minif2f" in lower:
        return "minif2f"
    return "other"


def _extract_verified(data: dict[str, Any]) -> int | None:
    """Return verified count, honoring both `verified` and `verified_count`."""
    for key in ("verified", "verified_count"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    return None


def _summarize(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Pull the standard summary fields out of a raw result dict."""
    filename = path.name
    benchmark = _detect_benchmark(filename, data)

    # Source filter / split fields live under config but vary by benchmark.
    config = data.get("config") or {}
    if not isinstance(config, dict):
        config = {}

    return {
        "path": str(path),
        "filename": filename,
        "benchmark": benchmark,
        "timestamp": data.get("timestamp"),
        "total": data.get("total"),
        "solved": data.get("solved"),
        "verified": _extract_verified(data),
        "repl_reported": data.get("repl_reported"),
        "false_positives": data.get("false_positives"),
        "pass_rate": data.get("pass_rate"),
        "total_time_seconds": data.get("total_time_seconds"),
        "split": config.get("split"),
        "source_filter": config.get("source_filter"),
        "use_loop": config.get("use_loop"),
        "use_search": config.get("use_search"),
        "use_decompose": config.get("use_decompose"),
        "use_repl": config.get("use_repl"),
        "pass_n": config.get("pass_n"),
        "loop_model": config.get("loop_model"),
        "loop_memory": config.get("loop_memory"),
        "loop_enable_mathlib_search": config.get("loop_enable_mathlib_search"),
    }


def _matches_filter(summary: dict[str, Any], filter: str | None) -> bool:
    """Case-insensitive substring match against benchmark name, filename, split."""
    if filter is None:
        return True
    needle = filter.lower()
    haystacks = (
        summary.get("filename") or "",
        summary.get("benchmark") or "",
        summary.get("split") or "",
        summary.get("source_filter") or "",
    )
    return any(needle in str(h).lower() for h in haystacks)


def _iter_result_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.json") if p.is_file())


def list_results(
    filter: str | None = None,
    results_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List all benchmark results with summary metadata.

    Args:
        filter: Optional case-insensitive substring matched against
            filename, benchmark, split, and source_filter.
        results_dir: Override the default results directory.

    Returns:
        Summaries sorted by timestamp descending (most recent first).
        Files that fail to parse are skipped silently.
    """
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    summaries: list[dict[str, Any]] = []
    for path in _iter_result_files(directory):
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        summary = _summarize(path, data)
        if _matches_filter(summary, filter):
            summaries.append(summary)

    def _sort_key(s: dict[str, Any]) -> tuple[int, str]:
        ts = s.get("timestamp") or ""
        # Sort by ISO timestamp string desc; missing timestamps sink to the bottom.
        return (0 if ts else 1, ts)

    summaries.sort(key=_sort_key)
    summaries.reverse()
    return summaries


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a single result JSON.

    Args:
        path: Path to a result JSON file.

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the parsed payload is not a JSON object.
    """
    p = Path(path)
    with p.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Result file {p} is not a JSON object")
    return data


def latest_result(
    filter: str | None = None,
    results_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return the most recent result matching the filter (or None).

    The return value is the full result JSON, not the summary. Use
    :func:`list_results` if you only need metadata.
    """
    summaries = list_results(filter=filter, results_dir=results_dir)
    if not summaries:
        return None
    return load_result(summaries[0]["path"])


def _per_problem_index(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map problem_id -> entry. Missing/duplicate ids fall back to position."""
    items = result.get("results")
    if not isinstance(items, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            continue
        pid = entry.get("problem_id")
        if not isinstance(pid, str) or not pid:
            pid = f"_anon_{i}"
        # If duplicates exist, keep the first; benchmark runs shouldn't dup.
        indexed.setdefault(pid, entry)
    return indexed


def _is_passing(entry: dict[str, Any]) -> bool:
    """A problem 'passes' if it's verified, or solved when verified is absent."""
    verified = entry.get("verified")
    if isinstance(verified, bool):
        return verified
    solved = entry.get("solved")
    if isinstance(solved, bool):
        return solved
    return False


def diff_results(a: dict[str, Any], b: dict[str, Any]) -> dict[str, list[str]]:
    """Per-problem comparison between two result payloads.

    Returns a dict with four sorted lists of ``problem_id``:

    - ``a_only`` — passes in *a* but not in *b*.
    - ``b_only`` — passes in *b* but not in *a*.
    - ``both_pass`` — passes in both.
    - ``both_fail`` — present in both, fails in both.

    Problems that only appear on one side count as fail on the missing
    side; they show up in ``a_only`` / ``b_only`` only if they pass on
    the side they appear on.
    """
    idx_a = _per_problem_index(a)
    idx_b = _per_problem_index(b)
    all_ids = set(idx_a) | set(idx_b)

    a_only: list[str] = []
    b_only: list[str] = []
    both_pass: list[str] = []
    both_fail: list[str] = []

    for pid in all_ids:
        entry_a = idx_a.get(pid)
        entry_b = idx_b.get(pid)
        pass_a = entry_a is not None and _is_passing(entry_a)
        pass_b = entry_b is not None and _is_passing(entry_b)
        if pass_a and pass_b:
            both_pass.append(pid)
        elif pass_a and not pass_b:
            a_only.append(pid)
        elif pass_b and not pass_a:
            b_only.append(pid)
        elif entry_a is not None and entry_b is not None:
            both_fail.append(pid)
        # If a problem appears on only one side and failed there, drop it
        # from the comparison: there is no symmetric outcome.

    return {
        "a_only": sorted(a_only),
        "b_only": sorted(b_only),
        "both_pass": sorted(both_pass),
        "both_fail": sorted(both_fail),
    }


def by_source(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Aggregate per-problem records by their ``source`` field.

    Always returns a dict (possibly empty) keyed by source name. Each
    value contains ``total``, ``solved``, ``verified``, and
    ``pass_rate`` (verified / total, rounded to 4 decimals). The
    ``by_source`` block already embedded in the result is used as a
    starting point when present so callers see the same numbers the
    runner wrote.

    Putnam-style results without per-problem source fields return ``{}``
    if no records carry a ``source`` and the file has no embedded
    ``by_source`` block.
    """
    embedded = result.get("by_source")
    aggregated: dict[str, dict[str, Any]] = {}

    if isinstance(embedded, dict):
        for name, block in embedded.items():
            if not isinstance(block, dict):
                continue
            total = int(block.get("total", 0) or 0)
            solved = int(block.get("solved", 0) or 0)
            aggregated[name] = {
                "total": total,
                "solved": solved,
                "verified": int(block.get("verified", solved) or 0),
                "pass_rate": round(solved / total, 4) if total else 0.0,
            }

    # Walk the per-problem records: lets us add verified counts when the
    # runner didn't embed them, and gives us a source breakdown when the
    # embedded block is missing entirely.
    records = result.get("results")
    if isinstance(records, list):
        from_records: dict[str, dict[str, int]] = {}
        for entry in records:
            if not isinstance(entry, dict):
                continue
            source = entry.get("source")
            if not isinstance(source, str) or not source:
                continue
            bucket = from_records.setdefault(
                source, {"total": 0, "solved": 0, "verified": 0}
            )
            bucket["total"] += 1
            if entry.get("solved"):
                bucket["solved"] += 1
            if entry.get("verified"):
                bucket["verified"] += 1
        for name, bucket in from_records.items():
            total = bucket["total"]
            current = aggregated.get(name)
            if current is None:
                aggregated[name] = {
                    "total": total,
                    "solved": bucket["solved"],
                    "verified": bucket["verified"],
                    "pass_rate": round(bucket["verified"] / total, 4) if total else 0.0,
                }
            else:
                # Backfill verified when missing.
                if not current.get("verified"):
                    current["verified"] = bucket["verified"]

    return aggregated


__all__ = [
    "RESULTS_DIR",
    "list_results",
    "load_result",
    "latest_result",
    "diff_results",
    "by_source",
]

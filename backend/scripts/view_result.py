#!/usr/bin/env python3
"""Pretty-print a benchmark result JSON.

Usage
-----
``python3 backend/scripts/view_result.py``                                 # newest .bourbaki/benchmarks/results/*.json
``python3 backend/scripts/view_result.py --latest``                        # same
``python3 backend/scripts/view_result.py path/to/file.json``               # specific file
``python3 backend/scripts/view_result.py file_a.json --diff file_b.json``  # A/B comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _runner_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print a Bourbaki benchmark result JSON.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a result JSON (omit to use --latest).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently written result JSON.",
    )
    parser.add_argument(
        "--diff",
        type=str,
        default=None,
        help="Path to a second result JSON; print A vs B comparison.",
    )
    return parser.parse_args()


def _resolve_path(args: argparse.Namespace) -> Path:
    if args.path and args.latest:
        print("error: pass either a path OR --latest, not both", file=sys.stderr)
        sys.exit(2)
    if args.path:
        p = Path(args.path).expanduser().resolve()
    elif args.latest:
        latest = common.newest_result_json()
        if latest is None:
            print(
                f"error: no result JSONs found in {common.RESULTS_DIR}",
                file=sys.stderr,
            )
            sys.exit(2)
        p = latest
    else:
        # No path AND no --latest: behave like --latest by default.
        latest = common.newest_result_json()
        if latest is None:
            print(
                "error: no path given and no result JSONs found in "
                f"{common.RESULTS_DIR}",
                file=sys.stderr,
            )
            sys.exit(2)
        p = latest
    if not p.is_file():
        print(f"error: file does not exist: {p}", file=sys.stderr)
        sys.exit(2)
    return p


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_header(title: str) -> None:
    bar = "=" * 72
    print(bar)
    print(title)
    print(bar)


def _print_aggregate(data: dict, path: Path) -> None:
    print(f"File:       {path}")
    print(f"Timestamp:  {data.get('timestamp', '?')}")
    total = data.get("total", 0)
    verified = data.get("verified", data.get("verified_count", 0))
    repl_reported = data.get("repl_reported", 0)
    fps = data.get("false_positives", 0)
    print(f"Total:      {total}")
    print(
        "Verified:   "
        f"{verified}/{total} "
        f"({100 * verified / total:.1f}%)" if total else f"Verified:   {verified}/0"
    )
    print(
        "REPL-rep:   "
        f"{repl_reported}/{total} "
        f"({100 * repl_reported / total:.1f}%)" if total else f"REPL-rep:   {repl_reported}/0"
    )
    print(f"FP count:   {fps}")
    total_time = data.get("total_time_seconds", 0.0)
    if isinstance(total_time, (int, float)):
        print(f"Total time: {total_time:.1f}s ({total_time / 60.0:.1f} min)")

    cfg = data.get("config", {}) or {}
    print("Config:")
    interesting = (
        "split", "timeout", "use_loop", "loop_max_iterations", "loop_model",
        "loop_memory", "loop_memory_k", "loop_enable_mathlib_search", "pass_n",
        "year_range", "exclude_answer", "attempt_answers",
    )
    for k in interesting:
        if k in cfg:
            print(f"  {k}: {cfg[k]}")


def _print_by_source(data: dict) -> None:
    for breakdown_key in ("by_source", "by_section", "by_decade"):
        bs = data.get(breakdown_key)
        if not bs:
            continue
        print()
        print(f"{breakdown_key}:")
        width = max(len(s) for s in bs) if bs else 8
        for src, counts in sorted(bs.items()):
            total = counts.get("total", 0)
            solved = counts.get("solved", 0)
            pct = 100 * solved / total if total else 0.0
            print(f"  {src:<{width}}  {solved:>3}/{total:<3} ({pct:5.1f}%)")


def _problem_table(data: dict) -> None:
    rows = data.get("results", []) or []
    if not rows:
        return

    # Sort by source then id, with putnam-style records using "putnam" + year/section.
    def sort_key(r: dict) -> tuple[str, str]:
        src = r.get("source") or f"putnam_{r.get('year', '?')}{r.get('section', '?')}"
        return (str(src), str(r.get("problem_id", "")))

    rows = sorted(rows, key=sort_key)
    print()
    print(f"Per-problem ({len(rows)} rows, sorted by source/id):")
    # Build the table
    header = f"  {'STATUS':<8} {'ATT':>4}  {'TIME':>7}  {'SRC':<10}  PROBLEM_ID"
    print(header)
    print(f"  {'-'*8} {'-'*4}  {'-'*7}  {'-'*10}  {'-'*40}")
    for r in rows:
        verified = bool(r.get("verified", False))
        repl_only = bool(r.get("repl_reported", False)) and not verified
        if verified:
            status = "PASS"
        elif repl_only:
            status = "FP"
        elif r.get("skipped"):
            status = "SKIP"
        else:
            status = "FAIL"
        attempts = r.get("attempts", 0)
        dur = r.get("duration_seconds", 0.0)
        src = r.get("source") or f"putnam_{r.get('section','?')}"
        pid = r.get("problem_id", "")
        print(f"  {status:<8} {attempts:>4}  {dur:>7.1f}  {str(src):<10}  {pid}")


def _print_false_positives(data: dict) -> None:
    rows = data.get("results", []) or []
    fps = [
        r for r in rows
        if r.get("repl_reported") and not r.get("verified")
        and not r.get("skipped")
    ]
    if not fps:
        return
    print()
    print(f"False positives ({len(fps)}):")
    for r in fps:
        pid = r.get("problem_id", "?")
        err = (r.get("error") or "")[:120]
        print(f"  - {pid}: {err}")


def _print_one(path: Path) -> None:
    data = _load(path)
    _print_header(f"Benchmark result")
    _print_aggregate(data, path)
    _print_by_source(data)
    _problem_table(data)
    _print_false_positives(data)


def _print_diff(path_a: Path, path_b: Path) -> None:
    a = _load(path_a)
    b = _load(path_b)
    a_rows = {r.get("problem_id"): r for r in (a.get("results", []) or [])}
    b_rows = {r.get("problem_id"): r for r in (b.get("results", []) or [])}

    a_total = a.get("total", 0)
    b_total = b.get("total", 0)
    a_verified = a.get("verified", a.get("verified_count", 0))
    b_verified = b.get("verified", b.get("verified_count", 0))

    _print_header("A vs B diff")
    print(f"A: {path_a}")
    print(f"   verified={a_verified}/{a_total}")
    print(f"B: {path_b}")
    print(f"   verified={b_verified}/{b_total}")
    delta = b_verified - a_verified
    sign = "+" if delta > 0 else ""
    print(f"Δ verified: {sign}{delta}")

    all_ids = sorted(set(a_rows) | set(b_rows))
    only_a_pass: list[str] = []
    only_b_pass: list[str] = []
    both_pass: list[str] = []
    both_fail: list[str] = []
    for pid in all_ids:
        ra = a_rows.get(pid) or {}
        rb = b_rows.get(pid) or {}
        a_pass = bool(ra.get("verified"))
        b_pass = bool(rb.get("verified"))
        if a_pass and b_pass:
            both_pass.append(pid)
        elif a_pass and not b_pass:
            only_a_pass.append(pid)
        elif b_pass and not a_pass:
            only_b_pass.append(pid)
        else:
            both_fail.append(pid)

    print()
    print(f"Both pass:    {len(both_pass)}")
    print(f"A only pass:  {len(only_a_pass)}  (regressions in B)")
    print(f"B only pass:  {len(only_b_pass)}  (gains in B)")
    print(f"Both fail:    {len(both_fail)}")

    def _list(label: str, ids: list[str]) -> None:
        if not ids:
            return
        print()
        print(f"{label}:")
        for pid in ids:
            print(f"  - {pid}")

    _list("Regressions in B (A pass, B fail)", only_a_pass)
    _list("Gains in B (A fail, B pass)", only_b_pass)


def main() -> None:
    args = parse_args()
    path_a = _resolve_path(args)
    if args.diff:
        path_b = Path(args.diff).expanduser().resolve()
        if not path_b.is_file():
            print(f"error: --diff file does not exist: {path_b}", file=sys.stderr)
            sys.exit(2)
        _print_diff(path_a, path_b)
    else:
        _print_one(path_a)


if __name__ == "__main__":
    main()

"""Tests for bourbaki.benchmarks.results_db.

The fixtures build a synthetic results directory so tests stay
independent of `.bourbaki/benchmarks/results/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bourbaki.benchmarks import results_db


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def synthetic_results(tmp_path: Path) -> Path:
    """Create a tmp directory with three result files spanning configs."""
    base = tmp_path / "results"
    base.mkdir()

    # Two miniF2F runs (newer wins as 'latest') and a putnam stub.
    _write(
        base / "2026-04-25_1835_minif2f_valid.json",
        {
            "total": 10,
            "solved": 9,
            "pass_rate": 0.9,
            "repl_reported": 9,
            "verified": 9,
            "false_positives": 0,
            "config": {
                "split": "valid",
                "use_loop": True,
                "use_search": False,
                "loop_model": "glm:glm-5.1",
            },
            "timestamp": "2026-04-25T22:35:28+00:00",
            "by_source": {
                "algebra": {"total": 5, "solved": 4},
                "mathd": {"total": 5, "solved": 5},
            },
            "results": [
                {"problem_id": "p1", "source": "algebra", "solved": True, "verified": True},
                {"problem_id": "p2", "source": "algebra", "solved": False, "verified": False},
                {"problem_id": "p3", "source": "mathd", "solved": True, "verified": True},
            ],
        },
    )

    _write(
        base / "2026-05-09_2241_minif2f_valid.json",
        {
            "total": 35,
            "solved": 22,
            "pass_rate": 0.6286,
            "repl_reported": 22,
            "verified": 22,
            "false_positives": 0,
            "config": {
                "split": "valid",
                "use_loop": True,
                "use_search": False,
                "loop_model": "glm:glm-5.1",
            },
            "timestamp": "2026-05-10T02:41:42+00:00",
            "by_source": {
                "algebra": {"total": 5, "solved": 3},
                "mathd": {"total": 15, "solved": 13},
            },
            "results": [
                {"problem_id": "p1", "source": "algebra", "solved": True, "verified": True},
                {"problem_id": "p2", "source": "algebra", "solved": True, "verified": True},
                {"problem_id": "p3", "source": "mathd", "solved": False, "verified": False},
                {"problem_id": "p4", "source": "mathd", "solved": True, "verified": True},
            ],
        },
    )

    _write(
        base / "2026-05-13_2206_putnam.json",
        {
            "benchmark": "putnam",
            "total": 0,
            "solved": 0,
            "pass_rate": 0.0,
            "verified_count": 0,
            "false_positives": 0,
            "config": {
                "use_loop": True,
                "loop_max_iterations": 50,
                "pass_n": 1,
            },
            "timestamp": "2026-05-14T02:06:06+00:00",
            "results": [],
        },
    )

    return base


# ---------------------------------------------------------------------------
# list_results
# ---------------------------------------------------------------------------


def test_list_results_returns_summaries_for_each_file(synthetic_results: Path) -> None:
    summaries = results_db.list_results(results_dir=synthetic_results)
    assert len(summaries) == 3
    filenames = {s["filename"] for s in summaries}
    assert filenames == {
        "2026-04-25_1835_minif2f_valid.json",
        "2026-05-09_2241_minif2f_valid.json",
        "2026-05-13_2206_putnam.json",
    }


def test_list_results_sorted_newest_first(synthetic_results: Path) -> None:
    summaries = results_db.list_results(results_dir=synthetic_results)
    timestamps = [s["timestamp"] for s in summaries]
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_results_filter_matches_substring(synthetic_results: Path) -> None:
    putnam = results_db.list_results(filter="putnam", results_dir=synthetic_results)
    assert len(putnam) == 1
    assert putnam[0]["benchmark"] == "putnam"


def test_list_results_filter_no_match_returns_empty(synthetic_results: Path) -> None:
    summaries = results_db.list_results(filter="nonexistent", results_dir=synthetic_results)
    assert summaries == []


def test_list_results_extracts_use_loop_and_pass_n(synthetic_results: Path) -> None:
    summaries = results_db.list_results(filter="putnam", results_dir=synthetic_results)
    assert summaries[0]["use_loop"] is True
    assert summaries[0]["pass_n"] == 1


def test_list_results_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert results_db.list_results(results_dir=tmp_path / "does_not_exist") == []


def test_list_results_skips_malformed_files(synthetic_results: Path) -> None:
    (synthetic_results / "broken.json").write_text("not valid json")
    summaries = results_db.list_results(results_dir=synthetic_results)
    assert len(summaries) == 3  # broken file silently dropped


def test_list_results_extracts_verified_count_field(synthetic_results: Path) -> None:
    summaries = results_db.list_results(filter="putnam", results_dir=synthetic_results)
    # Putnam stub uses `verified_count`, not `verified`.
    assert summaries[0]["verified"] == 0


# ---------------------------------------------------------------------------
# load_result
# ---------------------------------------------------------------------------


def test_load_result_returns_full_payload(synthetic_results: Path) -> None:
    path = synthetic_results / "2026-05-09_2241_minif2f_valid.json"
    data = results_db.load_result(path)
    assert data["total"] == 35
    assert data["verified"] == 22
    assert isinstance(data["results"], list)


def test_load_result_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        results_db.load_result(tmp_path / "does_not_exist.json")


def test_load_result_non_object_raises(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        results_db.load_result(path)


# ---------------------------------------------------------------------------
# latest_result
# ---------------------------------------------------------------------------


def test_latest_result_returns_newest(synthetic_results: Path) -> None:
    latest = results_db.latest_result(results_dir=synthetic_results)
    assert latest is not None
    assert latest["timestamp"] == "2026-05-14T02:06:06+00:00"


def test_latest_result_with_filter_returns_newest_match(synthetic_results: Path) -> None:
    latest = results_db.latest_result(filter="minif2f", results_dir=synthetic_results)
    assert latest is not None
    assert latest["total"] == 35


def test_latest_result_no_match_returns_none(synthetic_results: Path) -> None:
    assert results_db.latest_result(filter="nope", results_dir=synthetic_results) is None


def test_latest_result_empty_dir_returns_none(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert results_db.latest_result(results_dir=empty) is None


# ---------------------------------------------------------------------------
# diff_results
# ---------------------------------------------------------------------------


def test_diff_results_basic() -> None:
    a = {
        "results": [
            {"problem_id": "p1", "verified": True},
            {"problem_id": "p2", "verified": False},
            {"problem_id": "p3", "verified": True},
        ]
    }
    b = {
        "results": [
            {"problem_id": "p1", "verified": True},
            {"problem_id": "p2", "verified": True},
            {"problem_id": "p3", "verified": False},
        ]
    }
    diff = results_db.diff_results(a, b)
    assert diff["both_pass"] == ["p1"]
    assert diff["a_only"] == ["p3"]
    assert diff["b_only"] == ["p2"]
    assert diff["both_fail"] == []


def test_diff_results_solved_fallback_when_verified_absent() -> None:
    """Verified is absent — fall back to `solved` for the pass predicate."""
    a = {"results": [{"problem_id": "p1", "solved": True}]}
    b = {"results": [{"problem_id": "p1", "solved": False}]}
    diff = results_db.diff_results(a, b)
    assert diff["a_only"] == ["p1"]


def test_diff_results_both_fail_bucket() -> None:
    a = {"results": [{"problem_id": "p1", "verified": False}]}
    b = {"results": [{"problem_id": "p1", "verified": False}]}
    diff = results_db.diff_results(a, b)
    assert diff["both_fail"] == ["p1"]
    assert diff["both_pass"] == []


def test_diff_results_handles_missing_problems() -> None:
    """If a problem only appears on one side, it goes into that side's list only if it passed."""
    a = {"results": [{"problem_id": "p1", "verified": True}]}
    b = {"results": [{"problem_id": "p2", "verified": True}]}
    diff = results_db.diff_results(a, b)
    assert diff["a_only"] == ["p1"]
    assert diff["b_only"] == ["p2"]
    assert diff["both_pass"] == []
    assert diff["both_fail"] == []


def test_diff_results_empty_inputs() -> None:
    diff = results_db.diff_results({}, {})
    assert diff == {"a_only": [], "b_only": [], "both_pass": [], "both_fail": []}


# ---------------------------------------------------------------------------
# by_source
# ---------------------------------------------------------------------------


def test_by_source_uses_embedded_block() -> None:
    result = {
        "by_source": {
            "algebra": {"total": 5, "solved": 3},
            "mathd": {"total": 15, "solved": 13},
        },
        "results": [],
    }
    agg = results_db.by_source(result)
    assert agg["algebra"]["total"] == 5
    assert agg["algebra"]["solved"] == 3
    assert agg["algebra"]["pass_rate"] == 0.6
    assert agg["mathd"]["pass_rate"] == round(13 / 15, 4)


def test_by_source_aggregates_from_records_when_block_missing() -> None:
    """No embedded by_source — derive from the per-problem records."""
    result = {
        "results": [
            {"problem_id": "p1", "source": "algebra", "solved": True, "verified": True},
            {"problem_id": "p2", "source": "algebra", "solved": False, "verified": False},
            {"problem_id": "p3", "source": "mathd", "solved": True, "verified": True},
        ]
    }
    agg = results_db.by_source(result)
    assert agg["algebra"]["total"] == 2
    assert agg["algebra"]["solved"] == 1
    assert agg["algebra"]["verified"] == 1
    assert agg["algebra"]["pass_rate"] == 0.5
    assert agg["mathd"]["pass_rate"] == 1.0


def test_by_source_zero_total_safe() -> None:
    result = {"by_source": {"algebra": {"total": 0, "solved": 0}}}
    agg = results_db.by_source(result)
    assert agg["algebra"]["pass_rate"] == 0.0


def test_by_source_empty_returns_empty_dict() -> None:
    assert results_db.by_source({}) == {}
    assert results_db.by_source({"results": []}) == {}


def test_by_source_synthetic_file(synthetic_results: Path) -> None:
    """End-to-end: load a synthetic file then aggregate."""
    data = results_db.load_result(
        synthetic_results / "2026-05-09_2241_minif2f_valid.json"
    )
    agg = results_db.by_source(data)
    assert agg["algebra"]["total"] == 5
    assert agg["mathd"]["total"] == 15

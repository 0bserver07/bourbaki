"""Tests for persistent lemma library and shared lemma cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bourbaki.tools.lemma_library import (
    LemmaCache,
    LemmaEntry,
    LemmaLibrary,
    _extract_keywords,
    _normalize,
)


# ---------------------------------------------------------------------------
# LemmaEntry tests
# ---------------------------------------------------------------------------


class TestLemmaEntry:
    def test_creation_generates_id(self):
        entry = LemmaEntry(goal_pattern="a + b = b + a", tactics=["ring"])
        assert entry.id  # Should be auto-generated
        assert len(entry.id) == 12

    def test_creation_generates_timestamp(self):
        before = time.time()
        entry = LemmaEntry(goal_pattern="a + b = b + a", tactics=["ring"])
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_custom_id_preserved(self):
        entry = LemmaEntry(id="my-custom-id", goal_pattern="True", tactics=["trivial"])
        assert entry.id == "my-custom-id"

    def test_default_success_count(self):
        entry = LemmaEntry(goal_pattern="True", tactics=["trivial"])
        assert entry.success_count == 1

    def test_serialization_roundtrip(self):
        entry = LemmaEntry(
            id="test123",
            goal_pattern="n + 0 = n",
            tactics=["simp", "ring"],
            source="search_tree",
            theorem_context="theorem add_zero",
            timestamp=1000.0,
            success_count=5,
        )
        d = entry.to_dict()
        restored = LemmaEntry.from_dict(d)
        assert restored.id == "test123"
        assert restored.goal_pattern == "n + 0 = n"
        assert restored.tactics == ["simp", "ring"]
        assert restored.source == "search_tree"
        assert restored.theorem_context == "theorem add_zero"
        assert restored.timestamp == 1000.0
        assert restored.success_count == 5

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "id": "abc",
            "goal_pattern": "True",
            "tactics": ["trivial"],
            "source": "",
            "theorem_context": "",
            "timestamp": 1.0,
            "success_count": 1,
            "extra_field": "should be ignored",
        }
        entry = LemmaEntry.from_dict(d)
        assert entry.id == "abc"
        assert not hasattr(entry, "extra_field")


# ---------------------------------------------------------------------------
# Normalization / keyword extraction
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercase_and_strip(self):
        assert _normalize("  A + B  =  B + A  ") == "a + b = b + a"

    def test_collapses_whitespace(self):
        assert _normalize("a   +   b") == "a + b"


class TestExtractKeywords:
    def test_basic_keywords(self):
        kw = _extract_keywords("∀ n : ℕ, n + 0 = n")
        assert "n" not in kw  # Single char excluded
        assert "0" not in kw  # Single char excluded

    def test_strips_punctuation(self):
        kw = _extract_keywords("(Nat.add_comm)")
        assert "nat.add_comm" in kw

    def test_meaningful_tokens(self):
        kw = _extract_keywords("a * b = b * a")
        # Single chars excluded, but "*" and "=" also excluded (len < 2 after strip)
        # We mainly test it doesn't crash
        assert isinstance(kw, set)


# ---------------------------------------------------------------------------
# LemmaLibrary tests
# ---------------------------------------------------------------------------


class TestLemmaLibrary:
    def test_add_and_search(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(
            goal_pattern="a * b = b * a",
            tactics=["ring"],
            source="search_tree",
        ))
        results = lib.search("a * b = b * a")
        assert len(results) == 1
        assert results[0].tactics == ["ring"]

    def test_search_exact_match(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(goal_pattern="1 + 1 = 2", tactics=["norm_num"]))
        lib.add(LemmaEntry(goal_pattern="2 + 2 = 4", tactics=["norm_num"]))
        results = lib.search("1 + 1 = 2")
        assert len(results) >= 1
        assert results[0].goal_pattern == "1 + 1 = 2"

    def test_search_substring_match(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(
            goal_pattern="n + 0 = n",
            tactics=["simp"],
        ))
        # Searching with a broader goal that contains the pattern
        results = lib.search("∀ n : ℕ, n + 0 = n")
        assert len(results) >= 1

    def test_search_keyword_overlap(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(
            goal_pattern="Nat.add_comm m n",
            tactics=["exact Nat.add_comm m n"],
        ))
        results = lib.search("Nat.add_comm a b")
        assert len(results) >= 1

    def test_search_empty_goal(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(goal_pattern="True", tactics=["trivial"]))
        assert lib.search("") == []

    def test_search_empty_library(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        assert lib.search("anything") == []

    def test_search_max_results(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        for i in range(10):
            lib.add(LemmaEntry(
                goal_pattern=f"goal_{i} = result_{i}",
                tactics=[f"tactic_{i}"],
            ))
        results = lib.search("goal_0 = result_0", max_results=3)
        assert len(results) <= 3

    def test_search_sorted_by_success_count(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(
            id="low",
            goal_pattern="a + b = b + a",
            tactics=["ring"],
            success_count=1,
        ))
        lib.add(LemmaEntry(
            id="high",
            goal_pattern="a + b = b + a",
            tactics=["simp [add_comm]"],
            success_count=10,
        ))
        results = lib.search("a + b = b + a")
        # Both should match (exact), higher success_count first
        assert len(results) == 2
        assert results[0].success_count >= results[1].success_count

    def test_deduplication(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(goal_pattern="True", tactics=["trivial"]))
        lib.add(LemmaEntry(goal_pattern="True", tactics=["trivial"]))
        assert len(lib) == 1
        assert lib.entries[0].success_count == 2

    def test_deduplication_different_tactics(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(goal_pattern="True", tactics=["trivial"]))
        lib.add(LemmaEntry(goal_pattern="True", tactics=["simp"]))
        assert len(lib) == 2  # Different tactics = different entries

    def test_record_success(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        entry = LemmaEntry(id="test-id", goal_pattern="True", tactics=["trivial"])
        lib.add(entry)
        assert lib.entries[0].success_count == 1
        lib.record_success("test-id")
        assert lib.entries[0].success_count == 2
        lib.record_success("test-id")
        assert lib.entries[0].success_count == 3

    def test_record_success_nonexistent_id(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        lib.add(LemmaEntry(id="real-id", goal_pattern="True", tactics=["trivial"]))
        lib.record_success("nonexistent-id")  # Should not crash
        assert lib.entries[0].success_count == 1  # Unchanged

    def test_save_and_load(self, tmp_path: Path):
        path = tmp_path / "lemmas.json"

        # Save
        lib1 = LemmaLibrary(path)
        lib1.add(LemmaEntry(
            id="persist-1",
            goal_pattern="n + 0 = n",
            tactics=["simp"],
            source="search_tree",
            theorem_context="theorem add_zero",
        ))
        lib1.add(LemmaEntry(
            id="persist-2",
            goal_pattern="a * 1 = a",
            tactics=["ring"],
            source="decomposer",
        ))
        lib1.save()

        # Load in a new instance
        lib2 = LemmaLibrary(path)
        assert len(lib2) == 2
        assert lib2.entries[0].id == "persist-1"
        assert lib2.entries[0].goal_pattern == "n + 0 = n"
        assert lib2.entries[0].tactics == ["simp"]
        assert lib2.entries[1].id == "persist-2"

    def test_save_if_dirty(self, tmp_path: Path):
        path = tmp_path / "lemmas.json"
        lib = LemmaLibrary(path)

        # Not dirty initially
        lib.save_if_dirty()
        assert not path.exists()  # Nothing to save

        # Add an entry — now dirty
        lib.add(LemmaEntry(goal_pattern="True", tactics=["trivial"]))
        lib.save_if_dirty()
        assert path.exists()

    def test_save_creates_parent_directories(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "lemmas.json"
        lib = LemmaLibrary(path)
        lib.add(LemmaEntry(goal_pattern="True", tactics=["trivial"]))
        lib.save()
        assert path.exists()

    def test_load_nonexistent_file(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "nonexistent.json")
        assert len(lib) == 0  # Should not crash

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("this is not json{{{")
        lib = LemmaLibrary(path)
        assert len(lib) == 0  # Should gracefully degrade

    def test_len(self, tmp_path: Path):
        lib = LemmaLibrary(tmp_path / "lemmas.json")
        assert len(lib) == 0
        lib.add(LemmaEntry(goal_pattern="A", tactics=["a"]))
        assert len(lib) == 1
        lib.add(LemmaEntry(goal_pattern="B", tactics=["b"]))
        assert len(lib) == 2

    def test_json_structure(self, tmp_path: Path):
        path = tmp_path / "lemmas.json"
        lib = LemmaLibrary(path)
        lib.add(LemmaEntry(
            id="json-test",
            goal_pattern="1 + 1 = 2",
            tactics=["norm_num"],
            source="search_tree",
        ))
        lib.save()

        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "json-test"
        assert data[0]["goal_pattern"] == "1 + 1 = 2"
        assert data[0]["tactics"] == ["norm_num"]


# ---------------------------------------------------------------------------
# LemmaCache tests
# ---------------------------------------------------------------------------


class TestLemmaCache:
    def test_add_and_lookup_exact(self):
        cache = LemmaCache()
        cache.add("a + b = b + a", ["ring"])
        result = cache.lookup("a + b = b + a")
        assert result == ["ring"]

    def test_lookup_exact_normalized(self):
        cache = LemmaCache()
        cache.add("  A + B  =  B + A  ", ["ring"])
        result = cache.lookup("a + b = b + a")
        assert result == ["ring"]

    def test_lookup_substring_match(self):
        cache = LemmaCache()
        cache.add("n + 0 = n", ["simp"])
        # Broader goal containing the cached one
        result = cache.lookup("∀ n : ℕ, n + 0 = n")
        assert result == ["simp"]

    def test_lookup_substring_reverse(self):
        cache = LemmaCache()
        cache.add("∀ n : ℕ, n + 0 = n", ["omega"])
        # Narrower goal contained in the cached one
        result = cache.lookup("n + 0 = n")
        assert result == ["omega"]

    def test_lookup_miss(self):
        cache = LemmaCache()
        cache.add("a + b = b + a", ["ring"])
        result = cache.lookup("completely unrelated goal")
        assert result is None

    def test_lookup_empty_cache(self):
        cache = LemmaCache()
        assert cache.lookup("anything") is None

    def test_lookup_empty_goal(self):
        cache = LemmaCache()
        cache.add("True", ["trivial"])
        assert cache.lookup("") is None

    def test_add_empty_goal_ignored(self):
        cache = LemmaCache()
        cache.add("", ["ring"])
        assert len(cache) == 0

    def test_add_empty_tactics_ignored(self):
        cache = LemmaCache()
        cache.add("True", [])
        assert len(cache) == 0

    def test_len(self):
        cache = LemmaCache()
        assert len(cache) == 0
        cache.add("A", ["a"])
        assert len(cache) == 1
        cache.add("B", ["b"])
        assert len(cache) == 2

    def test_bool(self):
        cache = LemmaCache()
        assert not cache
        cache.add("A", ["a"])
        assert cache

    def test_multiple_entries_exact_preferred(self):
        cache = LemmaCache()
        cache.add("n * 0 = 0", ["simp"])
        cache.add("n + 0 = n", ["omega"])
        # Exact match should be returned
        result = cache.lookup("n + 0 = n")
        assert result == ["omega"]


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_search_tree_saving_pattern(self, tmp_path: Path):
        """Simulate what search_tree.py does when a proof is found."""
        lib = LemmaLibrary(tmp_path / "lemmas.json")

        # Simulate: search tree found a proof for a goal
        goal = "⊢ 1 + 1 = 2"
        tactics = ["norm_num"]
        theorem = "theorem one_plus_one : 1 + 1 = 2"

        lib.add(LemmaEntry(
            goal_pattern=goal,
            tactics=tactics,
            source="search_tree",
            theorem_context=theorem,
        ))
        lib.save()

        # Later, tactics.py searches the library
        lib2 = LemmaLibrary(tmp_path / "lemmas.json")
        hits = lib2.search("⊢ 1 + 1 = 2")
        assert len(hits) == 1
        assert hits[0].tactics == ["norm_num"]
        assert hits[0].source == "search_tree"

    def test_decomposer_cache_sharing_pattern(self):
        """Simulate how the decomposer shares solutions between siblings."""
        cache = LemmaCache()

        # First subgoal is solved
        cache.add("n + 0 = n", ["simp [Nat.add_zero]"])

        # Second subgoal has the same type — should hit the cache
        result = cache.lookup("n + 0 = n")
        assert result == ["simp [Nat.add_zero]"]

    def test_coordinator_saving_pattern(self, tmp_path: Path):
        """Simulate what coordinator.py does when a proof is verified."""
        lib = LemmaLibrary(tmp_path / "lemmas.json")

        # Coordinator produces a verified proof
        theorem = "theorem comm_add (a b : ℕ) : a + b = b + a"
        tactics = ["exact Nat.add_comm a b"]

        lib.add(LemmaEntry(
            goal_pattern=theorem,
            tactics=tactics,
            source="coordinator",
            theorem_context=theorem,
        ))
        lib.save()

        # Future run can find it
        lib2 = LemmaLibrary(tmp_path / "lemmas.json")
        hits = lib2.search("a + b = b + a")
        assert len(hits) >= 1

    def test_success_count_incrementing(self, tmp_path: Path):
        """Test that success_count tracks reuse correctly."""
        lib = LemmaLibrary(tmp_path / "lemmas.json")

        entry = LemmaEntry(
            id="reuse-test",
            goal_pattern="0 < 1",
            tactics=["norm_num"],
            source="search_tree",
        )
        lib.add(entry)
        assert lib.entries[0].success_count == 1

        # Record that this lemma was reused successfully
        lib.record_success("reuse-test")
        lib.record_success("reuse-test")
        assert lib.entries[0].success_count == 3

        # Save and reload — count should persist
        lib.save()
        lib2 = LemmaLibrary(tmp_path / "lemmas.json")
        assert lib2.entries[0].success_count == 3

    def test_full_add_search_save_load_cycle(self, tmp_path: Path):
        """Full lifecycle: add entries, search, save, load, search again."""
        path = tmp_path / "cycle.json"

        # Phase 1: Build library
        lib = LemmaLibrary(path)
        lib.add(LemmaEntry(
            goal_pattern="a * b = b * a",
            tactics=["ring"],
            source="search_tree",
            theorem_context="theorem mul_comm",
        ))
        lib.add(LemmaEntry(
            goal_pattern="0 + n = n",
            tactics=["simp"],
            source="decomposer",
            theorem_context="theorem zero_add",
        ))
        lib.add(LemmaEntry(
            goal_pattern="n < n + 1",
            tactics=["omega"],
            source="search_tree",
            theorem_context="theorem lt_succ",
        ))
        lib.save()

        # Phase 2: Load and search
        lib2 = LemmaLibrary(path)
        assert len(lib2) == 3

        # Search for multiplication commutativity
        hits = lib2.search("a * b = b * a")
        assert len(hits) >= 1
        assert hits[0].tactics == ["ring"]

        # Search for addition identity
        hits = lib2.search("0 + n = n")
        assert len(hits) >= 1
        assert hits[0].tactics == ["simp"]

        # Record a success
        lib2.record_success(hits[0].id)
        lib2.save()

        # Phase 3: Reload and verify success count
        lib3 = LemmaLibrary(path)
        addition_entries = lib3.search("0 + n = n")
        assert addition_entries[0].success_count == 2

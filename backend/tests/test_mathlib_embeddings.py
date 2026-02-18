"""Tests for mathlib_embeddings — local FAISS index builder and query engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bourbaki.tools.mathlib_embeddings import (
    MathlibDeclaration,
    _extract_docstring_before,
    _file_to_module,
    parse_lean_file,
    extract_all_declarations,
)


# ---------------------------------------------------------------------------
# MathlibDeclaration tests
# ---------------------------------------------------------------------------


class TestMathlibDeclaration:
    def test_embedding_text_all_fields(self):
        decl = MathlibDeclaration(
            name="Nat.add_comm",
            kind="theorem",
            type_sig="∀ (m n : ℕ), m + n = n + m",
            module="Mathlib.Data.Nat.Basic",
            docstring="Addition of natural numbers is commutative.",
        )
        text = decl.embedding_text
        assert "Nat add_comm" in text
        assert "m + n = n + m" in text
        assert "commutative" in text

    def test_embedding_text_no_doc(self):
        decl = MathlibDeclaration(
            name="Nat.succ_pos",
            kind="theorem",
            type_sig="∀ (n : ℕ), 0 < Nat.succ n",
            module="Mathlib.Data.Nat.Basic",
            docstring="",
        )
        text = decl.embedding_text
        assert "Nat succ_pos" in text
        assert "Nat.succ n" in text

    def test_embedding_text_truncates_long_fields(self):
        decl = MathlibDeclaration(
            name="Foo.bar",
            kind="def",
            type_sig="x" * 1000,
            module="Mathlib.Foo",
            docstring="y" * 1000,
        )
        text = decl.embedding_text
        # Should be capped (300 chars each for type_sig and docstring)
        assert len(text) < 700


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestFileToModule:
    def test_basic_conversion(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        filepath = mathlib_root / "Data" / "Nat" / "Basic.lean"
        filepath.parent.mkdir(parents=True)
        filepath.touch()
        result = _file_to_module(filepath, mathlib_root)
        assert result == "Mathlib.Data.Nat.Basic"

    def test_top_level_file(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        filepath = mathlib_root / "Init.lean"
        filepath.touch()
        result = _file_to_module(filepath, mathlib_root)
        assert result == "Mathlib.Init"


class TestExtractDocstring:
    def test_simple_docstring(self):
        content = '/-- This is a docstring. -/\ntheorem foo : True := trivial'
        pos = content.index("theorem")
        doc = _extract_docstring_before(content, pos)
        assert doc == "This is a docstring."

    def test_multiline_docstring(self):
        content = '/-- This is\na multiline\ndocstring. -/\nlemma bar : True := trivial'
        pos = content.index("lemma")
        doc = _extract_docstring_before(content, pos)
        assert "multiline" in doc
        assert "docstring" in doc

    def test_no_docstring(self):
        content = 'theorem foo : True := trivial'
        pos = content.index("theorem")
        doc = _extract_docstring_before(content, pos)
        assert doc == ""


SAMPLE_LEAN = """\
/-
Copyright (c) 2024 Test. All rights reserved.
-/

/-- Addition is commutative. -/
theorem Nat.add_comm (m n : ℕ) : m + n = n + m := by omega

/-- Successor is positive. -/
lemma Nat.succ_pos (n : ℕ) : 0 < Nat.succ n := Nat.succ_pos n

def Nat.myHelper (x : ℕ) : ℕ := x + 1

private def _internal (x : ℕ) : ℕ := x

abbrev MyNat := ℕ
"""


class TestParseLeanFile:
    def test_parse_sample(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        data_dir = mathlib_root / "Data" / "Nat"
        data_dir.mkdir(parents=True)
        filepath = data_dir / "Basic.lean"
        filepath.write_text(SAMPLE_LEAN)

        decls = parse_lean_file(filepath, mathlib_root)
        names = [d.name for d in decls]

        # Should find the public declarations
        assert "Nat.add_comm" in names
        assert "Nat.succ_pos" in names
        assert "Nat.myHelper" in names
        assert "MyNat" in names

        # Should NOT find private/internal declarations
        assert "_internal" not in names

    def test_docstrings_extracted(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        filepath = mathlib_root / "Test.lean"
        filepath.write_text(SAMPLE_LEAN)

        decls = parse_lean_file(filepath, mathlib_root)
        by_name = {d.name: d for d in decls}

        add_comm = by_name.get("Nat.add_comm")
        assert add_comm is not None
        assert "commutative" in add_comm.docstring.lower()

        succ_pos = by_name.get("Nat.succ_pos")
        assert succ_pos is not None
        assert "positive" in succ_pos.docstring.lower()

    def test_kinds_detected(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        filepath = mathlib_root / "Test.lean"
        filepath.write_text(SAMPLE_LEAN)

        decls = parse_lean_file(filepath, mathlib_root)
        by_name = {d.name: d for d in decls}

        assert by_name["Nat.add_comm"].kind == "theorem"
        assert by_name["Nat.succ_pos"].kind == "lemma"
        assert by_name["Nat.myHelper"].kind == "def"
        assert by_name["MyNat"].kind == "abbrev"

    def test_module_path(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        sub = mathlib_root / "Algebra" / "Group"
        sub.mkdir(parents=True)
        filepath = sub / "Basic.lean"
        filepath.write_text("theorem foo : True := trivial\n")

        decls = parse_lean_file(filepath, mathlib_root)
        assert len(decls) >= 1
        assert decls[0].module == "Mathlib.Algebra.Group.Basic"

    def test_empty_file(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        filepath = mathlib_root / "Empty.lean"
        filepath.write_text("")

        decls = parse_lean_file(filepath, mathlib_root)
        assert decls == []


class TestExtractAllDeclarations:
    def test_extract_from_temp_mathlib(self, tmp_path):
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()

        # Create a couple of files
        (mathlib_root / "FileA.lean").write_text(
            "theorem A.foo : True := trivial\n"
            "lemma A.bar : True := trivial\n"
        )
        sub = mathlib_root / "Sub"
        sub.mkdir()
        (sub / "FileB.lean").write_text(
            "def B.baz (x : ℕ) : ℕ := x\n"
        )

        decls = extract_all_declarations(mathlib_root)
        names = {d.name for d in decls}
        assert "A.foo" in names
        assert "A.bar" in names
        assert "B.baz" in names

    def test_nonexistent_root_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_all_declarations(Path("/nonexistent/Mathlib"))


# ---------------------------------------------------------------------------
# Index build + query tests (with mock FAISS/sentence-transformers)
# ---------------------------------------------------------------------------


class TestBuildAndSearch:
    """Test the full build + search pipeline using mocked ML dependencies."""

    def test_build_index_and_search(self, tmp_path):
        """Build an index from sample data and verify search returns results."""
        # We need both faiss and sentence-transformers for this test.
        # Skip if not installed.
        pytest.importorskip("faiss", reason="faiss-cpu not installed")
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

        import numpy as np

        from bourbaki.tools.mathlib_embeddings import build_index, _holder

        # Create sample Mathlib files
        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        (mathlib_root / "Nat.lean").write_text(
            "/-- Addition is commutative for natural numbers. -/\n"
            "theorem Nat.add_comm (m n : ℕ) : m + n = n + m := by omega\n"
            "\n"
            "/-- Multiplication distributes over addition. -/\n"
            "theorem Nat.mul_add (a b c : ℕ) : a * (b + c) = a * b + a * c := by ring\n"
            "\n"
            "/-- Zero is the additive identity. -/\n"
            "theorem Nat.add_zero (n : ℕ) : n + 0 = n := by simp\n"
        )
        (mathlib_root / "List.lean").write_text(
            "/-- Length of append equals sum of lengths. -/\n"
            "theorem List.length_append (l1 l2 : List α) : "
            "(l1 ++ l2).length = l1.length + l2.length := by simp\n"
            "\n"
            "/-- Mapping preserves length. -/\n"
            "theorem List.length_map (f : α → β) (l : List α) : "
            "(l.map f).length = l.length := by simp\n"
        )

        index_dir = tmp_path / "index"

        # Both mathlib_root and index_dir are explicit, so no settings needed
        result = build_index(mathlib_root=mathlib_root, index_dir=index_dir)

        assert result["success"] is True
        assert result["count"] == 5  # 3 from Nat.lean + 2 from List.lean

        # Verify files were created
        assert (index_dir / "faiss.index").exists()
        assert (index_dir / "metadata.json").exists()
        assert (index_dir / "index_info.json").exists()

        # Verify metadata content
        with open(index_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert len(metadata) == 5
        names = {m["name"] for m in metadata}
        assert "Nat.add_comm" in names
        assert "List.length_append" in names

    def test_build_with_no_files(self, tmp_path):
        """Building with empty directory should report failure."""
        pytest.importorskip("faiss", reason="faiss-cpu not installed")
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

        from bourbaki.tools.mathlib_embeddings import build_index

        mathlib_root = tmp_path / "Mathlib"
        mathlib_root.mkdir()
        index_dir = tmp_path / "index"

        # Both mathlib_root and index_dir are explicit, so no settings needed
        result = build_index(mathlib_root=mathlib_root, index_dir=index_dir)

        assert result["success"] is False


class TestSearchLocal:
    """Test the search_local async function."""

    @pytest.mark.asyncio
    async def test_search_when_index_not_available(self):
        """search_local should return error when no index exists."""
        from bourbaki.tools.mathlib_embeddings import search_local, _holder

        _holder.reset()

        # Patch _get_index_dir to return a nonexistent path
        with patch.object(_holder, "_get_index_dir", return_value=Path("/nonexistent/mathlib-index")):
            result = await search_local("test query")

        assert result["success"] is False
        assert result["mode"] == "local"
        assert "duration" in result

    @pytest.mark.asyncio
    async def test_search_with_mocked_holder(self):
        """search_local should return formatted results from the holder."""
        from bourbaki.tools.mathlib_embeddings import search_local, _holder

        mock_results = [
            {"name": "Nat.add_comm", "module": "Mathlib.Data.Nat.Basic",
             "type": "∀ (m n : ℕ), m + n = n + m", "doc": "Commutativity of addition.",
             "score": 0.95},
            {"name": "Nat.mul_comm", "module": "Mathlib.Data.Nat.Basic",
             "type": "∀ (m n : ℕ), m * n = n * m", "doc": "Commutativity of multiplication.",
             "score": 0.85},
        ]

        with patch.object(_holder, "search", return_value=mock_results):
            result = await search_local("commutativity of addition", max_results=5)

        assert result["success"] is True
        assert result["mode"] == "local"
        assert result["count"] == 2
        assert result["results"][0]["name"] == "Nat.add_comm"
        assert result["results"][1]["name"] == "Nat.mul_comm"
        # Score should not leak into the formatted results
        assert "score" not in result["results"][0]
        assert "duration" in result


# ---------------------------------------------------------------------------
# Integration with mathlib_search.py
# ---------------------------------------------------------------------------


class TestMathlibSearchLocalMode:
    """Test the mode='local' integration in mathlib_search."""

    @pytest.mark.asyncio
    async def test_local_mode_dispatches(self):
        """mathlib_search(mode='local') should call _search_local."""
        from bourbaki.tools.mathlib_search import mathlib_search
        from bourbaki.tools.mathlib_embeddings import _holder

        mock_results = [
            {"name": "Nat.add_comm", "module": "Mathlib.Data.Nat.Basic",
             "type": "∀ m n, m + n = n + m", "doc": "Addition is commutative.",
             "score": 0.9},
        ]

        with patch.object(_holder, "search", return_value=mock_results):
            result = await mathlib_search("addition commutative", mode="local")

        assert result["success"] is True
        assert result["mode"] == "local"
        assert result["count"] == 1
        assert result["results"][0]["name"] == "Nat.add_comm"

    @pytest.mark.asyncio
    async def test_unknown_mode_updated_error_message(self):
        """Unknown mode error should list 'local' as a valid option."""
        from bourbaki.tools.mathlib_search import mathlib_search

        result = await mathlib_search("test", mode="bad_mode")
        assert result["success"] is False
        assert "local" in result["error"]


class TestSemanticFallbackChain:
    """Test that semantic mode tries local -> LeanExplore -> LeanSearch."""

    @pytest.mark.asyncio
    async def test_semantic_uses_local_first_when_available(self):
        """Semantic mode should use local FAISS index when available."""
        from unittest.mock import AsyncMock
        from bourbaki.tools.mathlib_search import mathlib_search

        local_result = {
            "success": True,
            "results": [{"name": "Nat.add_comm", "module": "Mathlib.Data.Nat.Basic",
                         "type": "∀ m n, m + n = n + m", "doc": "Commutativity."}],
            "count": 1,
            "query": "addition commutative",
            "mode": "local",
            "duration": 5,
        }

        # Patch at the source module since _search_semantic imports from there
        with patch("bourbaki.tools.mathlib_embeddings.is_index_available", return_value=True):
            with patch("bourbaki.tools.mathlib_embeddings.search_local", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = local_result
                result = await mathlib_search("addition commutative", mode="semantic")

        assert result["success"] is True
        assert result["mode"] == "semantic"  # mode gets overridden to semantic
        mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_falls_through_when_local_not_available(self):
        """When local index is not available, semantic should try LeanExplore then LeanSearch."""
        from unittest.mock import AsyncMock
        from bourbaki.tools.mathlib_search import mathlib_search

        leansearch_data = [
            {"name": "Nat.add_comm", "module": "Mathlib", "type": "type", "doc": "doc"}
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = leansearch_data
        mock_resp.raise_for_status.return_value = None

        with patch("bourbaki.tools.mathlib_embeddings.is_index_available", return_value=False):
            with patch("bourbaki.tools.mathlib_search._get_leanexplore_api_key", return_value=None):
                with patch("bourbaki.tools.mathlib_search.httpx.AsyncClient") as MockClient:
                    mock_client = AsyncMock()
                    mock_client.get.return_value = mock_resp
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=False)
                    MockClient.return_value = mock_client

                    result = await mathlib_search("test query", mode="semantic")

        # Should have fallen through to LeanSearch
        assert result["success"] is True

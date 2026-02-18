"""Local FAISS embedding index for offline Mathlib lemma retrieval.

Builds and queries a FAISS index over Mathlib declarations (theorems, lemmas, defs)
extracted from Mathlib source files. Uses sentence-transformers for embedding.

Index is stored at .bourbaki/mathlib-index/ and lazy-loaded on first query.

Usage:
    # Build index (one-time, ~5-10 min depending on hardware)
    from bourbaki.tools.mathlib_embeddings import build_index
    await build_index()

    # Query (fast, <100ms after first load)
    from bourbaki.tools.mathlib_embeddings import search_local
    results = await search_local("product of positive numbers is positive")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR_NAME = "mathlib-index"
INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.json"
INFO_FILE = "index_info.json"

# Declaration patterns in Lean 4 source
_DECL_RE = re.compile(
    r"^(?:protected\s+|private\s+|noncomputable\s+|@\[.*?\]\s+)*"
    r"(theorem|lemma|def|abbrev|instance)\s+"
    r"([\w\.]+(?:\s*\{[^}]*\})?)"
    r"\s*(.*?)(?:\s*:=|\s*where|\s*\|)",
    re.MULTILINE | re.DOTALL,
)

# Simpler pattern that captures one-line declarations more reliably
_DECL_LINE_RE = re.compile(
    r"^(?:protected\s+|private\s+|noncomputable\s+|@\[.*?\]\s+)*"
    r"(theorem|lemma|def|abbrev)\s+"
    r"([\w\.]+)"
    r"\s*(.+?)$",
    re.MULTILINE,
)

# Doc comment pattern (Lean 4 uses /-- ... -/)
_DOC_COMMENT_RE = re.compile(r"/--\s*(.*?)\s*-/", re.DOTALL)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MathlibDeclaration:
    """A single Mathlib declaration with metadata."""
    name: str
    kind: str  # theorem, lemma, def, abbrev, instance
    type_sig: str  # type signature (may be partial)
    module: str  # Mathlib module path (e.g. Mathlib.Data.Nat.Basic)
    docstring: str  # docstring if available

    @property
    def embedding_text(self) -> str:
        """Text to embed for semantic search."""
        parts = [self.name.replace(".", " ")]
        if self.type_sig:
            parts.append(self.type_sig[:300])  # cap long signatures
        if self.docstring:
            parts.append(self.docstring[:300])
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Singleton index holder (lazy-loaded)
# ---------------------------------------------------------------------------

class _IndexHolder:
    """Lazy-loaded singleton for the FAISS index and metadata."""

    def __init__(self) -> None:
        self._index = None
        self._metadata: list[dict[str, str]] | None = None
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def reset(self) -> None:
        """Reset the index (for testing or rebuilding)."""
        self._index = None
        self._metadata = None
        self._model = None
        self._loaded = False

    def _get_index_dir(self) -> Path:
        """Get the index directory path."""
        from bourbaki.config import settings
        return Path(settings.bourbaki_dir) / INDEX_DIR_NAME

    def _ensure_model(self):
        """Load the sentence-transformers model (lazy)."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local Mathlib search. "
                "Install it with: pip install sentence-transformers"
            )
        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if self._loaded:
            return True

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for local Mathlib search. "
                "Install it with: pip install faiss-cpu"
            )

        index_dir = self._get_index_dir()
        index_path = index_dir / INDEX_FILE
        metadata_path = index_dir / METADATA_FILE

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(
                "Mathlib FAISS index not found at %s. Run build_index() first.", index_dir
            )
            return False

        logger.info("Loading Mathlib FAISS index from %s ...", index_dir)
        self._index = faiss.read_index(str(index_path))
        with open(metadata_path, "r") as f:
            self._metadata = json.load(f)

        # Sanity check
        if self._index.ntotal != len(self._metadata):
            logger.error(
                "Index/metadata mismatch: %d vectors vs %d entries",
                self._index.ntotal, len(self._metadata),
            )
            self.reset()
            return False

        self._loaded = True
        logger.info(
            "Loaded Mathlib index: %d declarations", self._index.ntotal
        )
        return True

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search the index for similar declarations."""
        import numpy as np

        if not self._loaded:
            if not self.load():
                return []

        model = self._ensure_model()

        # Embed query
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        # Search
        scores, indices = self._index.search(query_vec, min(max_results, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            entry = self._metadata[idx].copy()
            entry["score"] = float(score)
            results.append(entry)

        return results

    def save_index(self, index, metadata: list[dict], index_dir: Path) -> None:
        """Save FAISS index and metadata to disk."""
        import faiss

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / INDEX_FILE))
        with open(index_dir / METADATA_FILE, "w") as f:
            json.dump(metadata, f)
        logger.info("Saved index with %d entries to %s", len(metadata), index_dir)


# Global singleton
_holder = _IndexHolder()


# ---------------------------------------------------------------------------
# Mathlib source parser
# ---------------------------------------------------------------------------

def _file_to_module(filepath: Path, mathlib_root: Path) -> str:
    """Convert a file path to a Lean module name.

    E.g. Mathlib/Data/Nat/Basic.lean -> Mathlib.Data.Nat.Basic
    """
    rel = filepath.relative_to(mathlib_root.parent)
    return str(rel).replace("/", ".").removesuffix(".lean")


def _extract_docstring_before(content: str, pos: int) -> str:
    """Extract the doc comment immediately preceding position `pos`."""
    # Look backwards from pos for a /-- ... -/ block
    # We search in the 500 chars before the declaration
    search_region = content[max(0, pos - 500):pos]
    matches = list(_DOC_COMMENT_RE.finditer(search_region))
    if matches:
        # Take the last (closest) doc comment
        doc = matches[-1].group(1).strip()
        # Clean up multi-line doc comments
        doc = re.sub(r"\n\s*", " ", doc)
        return doc[:500]  # cap length
    return ""


def parse_lean_file(filepath: Path, mathlib_root: Path) -> list[MathlibDeclaration]:
    """Extract declarations from a single Lean source file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    module = _file_to_module(filepath, mathlib_root)
    declarations: list[MathlibDeclaration] = []
    seen_names: set[str] = set()

    for match in _DECL_LINE_RE.finditer(content):
        kind = match.group(1)
        name = match.group(2).strip()
        rest = match.group(3).strip()

        # Skip private/internal declarations
        if name.startswith("_") or "._" in name:
            continue

        # Avoid duplicates within the same file
        if name in seen_names:
            continue
        seen_names.add(name)

        # Extract type signature (everything after the name until := or where)
        # The rest from the regex is the remainder of the line
        type_sig = ""
        if ":" in rest:
            # Take everything after the first colon
            colon_idx = rest.index(":")
            type_sig = rest[colon_idx + 1:].strip()
            # Clean up — remove trailing := or where
            for terminator in [":=", " where", " by"]:
                if terminator in type_sig:
                    type_sig = type_sig[:type_sig.index(terminator)].strip()
            type_sig = type_sig[:500]  # cap length

        # Extract docstring
        docstring = _extract_docstring_before(content, match.start())

        declarations.append(MathlibDeclaration(
            name=name,
            kind=kind,
            type_sig=type_sig,
            module=module,
            docstring=docstring,
        ))

    return declarations


def _find_mathlib_root() -> Path | None:
    """Find the Mathlib source root in the lean project."""
    from bourbaki.config import settings

    candidates = [
        Path(settings.bourbaki_dir) / "lean-project" / ".lake" / "packages" / "mathlib" / "Mathlib",
        Path(".bourbaki") / "lean-project" / ".lake" / "packages" / "mathlib" / "Mathlib",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_all_declarations(
    mathlib_root: Path | None = None,
    progress_callback=None,
) -> list[MathlibDeclaration]:
    """Extract all declarations from Mathlib source files.

    Args:
        mathlib_root: Path to Mathlib/ directory. Auto-detected if None.
        progress_callback: Optional callable(current, total) for progress reporting.

    Returns:
        List of MathlibDeclaration objects.
    """
    if mathlib_root is None:
        mathlib_root = _find_mathlib_root()
        if mathlib_root is None:
            raise FileNotFoundError(
                "Cannot find Mathlib sources. Expected at "
                ".bourbaki/lean-project/.lake/packages/mathlib/Mathlib/"
            )

    if not mathlib_root.exists():
        raise FileNotFoundError(f"Mathlib root does not exist: {mathlib_root}")

    lean_files = sorted(mathlib_root.rglob("*.lean"))
    total = len(lean_files)
    logger.info("Extracting declarations from %d Lean files...", total)

    all_decls: list[MathlibDeclaration] = []
    for i, filepath in enumerate(lean_files):
        decls = parse_lean_file(filepath, mathlib_root)
        all_decls.extend(decls)
        if progress_callback and (i % 100 == 0 or i == total - 1):
            progress_callback(i + 1, total)

    logger.info("Extracted %d declarations from %d files", len(all_decls), total)
    return all_decls


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    mathlib_root: Path | None = None,
    index_dir: Path | None = None,
    batch_size: int = 256,
    progress_callback=None,
) -> dict[str, Any]:
    """Build the FAISS embedding index from Mathlib source files.

    This is a one-time operation that takes ~5-10 minutes.

    Args:
        mathlib_root: Path to Mathlib/ directory. Auto-detected if None.
        index_dir: Where to save the index. Defaults to .bourbaki/mathlib-index/.
        batch_size: Embedding batch size.
        progress_callback: Optional callable(stage, current, total) for progress.

    Returns:
        Dict with build statistics.
    """
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            f"Missing dependency for index building: {e}. "
            "Install with: pip install faiss-cpu sentence-transformers"
        )

    start = time.monotonic()

    # 1. Extract declarations
    logger.info("Step 1/3: Extracting Mathlib declarations...")
    decls = extract_all_declarations(
        mathlib_root,
        progress_callback=lambda cur, tot: (
            progress_callback("extract", cur, tot) if progress_callback else None
        ),
    )

    if not decls:
        return {"success": False, "error": "No declarations found", "count": 0}

    # 2. Embed
    logger.info("Step 2/3: Embedding %d declarations with %s...", len(decls), EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d.embedding_text for d in decls]

    # Encode in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
        if progress_callback:
            progress_callback("embed", min(i + batch_size, len(texts)), len(texts))

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    dim = embeddings_matrix.shape[1]

    # 3. Build FAISS index
    logger.info("Step 3/3: Building FAISS index (dim=%d, n=%d)...", dim, len(decls))
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
    index.add(embeddings_matrix)

    # Prepare metadata
    metadata = []
    for d in decls:
        metadata.append({
            "name": d.name,
            "kind": d.kind,
            "type": d.type_sig,
            "module": d.module,
            "doc": d.docstring,
        })

    # Save
    if index_dir is None:
        from bourbaki.config import settings
        index_dir = Path(settings.bourbaki_dir) / INDEX_DIR_NAME

    _holder.save_index(index, metadata, index_dir)

    # Save build info
    elapsed = time.monotonic() - start
    info = {
        "declarations": len(decls),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": dim,
        "build_time_seconds": round(elapsed, 1),
    }
    with open(index_dir / INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)

    # Reset holder so next query reloads the fresh index
    _holder.reset()

    logger.info("Index built in %.1fs: %d declarations", elapsed, len(decls))
    return {"success": True, "count": len(decls), "build_time": round(elapsed, 1), **info}


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

async def search_local(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search the local FAISS index for Mathlib lemmas.

    Args:
        query: Natural language query or type signature.
        max_results: Maximum number of results.

    Returns:
        Dict with success, results, count, query, mode, duration fields.
        Same format as other mathlib_search modes.
    """
    start = time.monotonic()

    try:
        results = _holder.search(query, max_results)
    except ImportError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "mode": "local",
            "duration": elapsed,
        }
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        logger.error("Local search error: %s", e)
        return {
            "success": False,
            "error": f"Local search failed: {e}",
            "query": query,
            "mode": "local",
            "duration": elapsed,
        }

    if not results:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": "Local index not available. Run build_index() first.",
            "query": query,
            "mode": "local",
            "duration": elapsed,
        }

    # Format results to match other modes
    formatted = []
    for r in results:
        formatted.append({
            "name": r.get("name", ""),
            "module": r.get("module", ""),
            "type": r.get("type", ""),
            "doc": r.get("doc", ""),
        })

    elapsed = int((time.monotonic() - start) * 1000)
    return {
        "success": True,
        "results": formatted,
        "count": len(formatted),
        "query": query,
        "mode": "local",
        "duration": elapsed,
    }


def is_index_available() -> bool:
    """Check if the local FAISS index exists on disk."""
    try:
        from bourbaki.config import settings
        index_dir = Path(settings.bourbaki_dir) / INDEX_DIR_NAME
        return (index_dir / INDEX_FILE).exists() and (index_dir / METADATA_FILE).exists()
    except Exception:
        return False


def get_index_info() -> dict[str, Any] | None:
    """Get info about the built index, if it exists."""
    try:
        from bourbaki.config import settings
        info_path = Path(settings.bourbaki_dir) / INDEX_DIR_NAME / INFO_FILE
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)
    except Exception:
        pass
    return None

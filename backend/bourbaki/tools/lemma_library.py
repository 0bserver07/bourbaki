"""Persistent lemma library and shared lemma cache.

Provides two complementary features for reusing discovered proof tactics:

1. **LemmaLibrary** (persistent): Stores lemma solutions across proof sessions
   in `.bourbaki/lemma-library.json`. When the search tree, decomposer, or
   coordinator discovers a tactic sequence that solves a goal, it is saved here
   for future proofs.

2. **LemmaCache** (in-memory, per session): Caches solved subgoals during a
   single proof search so sibling subgoals can reuse solutions immediately.
   Implements BFS-Prover-V2's shared cache pattern.

Both use simple keyword/substring matching — no embeddings needed (the FAISS
index in mathlib_embeddings.py handles semantic retrieval for Mathlib; this
module is for our own discovered lemmas).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location for the persistent lemma library
_DEFAULT_LIBRARY_PATH = Path(".bourbaki/lemma-library.json")


# ---------------------------------------------------------------------------
# Part 1: Persistent Lemma Library
# ---------------------------------------------------------------------------


@dataclass
class LemmaEntry:
    """A single discovered lemma (goal + tactic sequence that solved it)."""

    id: str = ""
    goal_pattern: str = ""           # The goal type that was solved
    tactics: list[str] = field(default_factory=list)  # Tactic sequence
    source: str = ""                 # "search_tree", "coordinator", "decomposer"
    theorem_context: str = ""        # Which theorem it was found while proving
    timestamp: float = 0.0
    success_count: int = 1           # How many times reused successfully

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LemmaEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _normalize(text: str) -> str:
    """Normalize a goal string for matching: lowercase, strip whitespace."""
    return " ".join(text.lower().split())


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a goal string for fuzzy matching."""
    normalized = _normalize(text)
    # Split on non-alphanumeric (keeping unicode math symbols as single tokens)
    tokens: set[str] = set()
    for token in normalized.split():
        token = token.strip("()[]{}.,;:")
        if len(token) >= 2:  # Skip single chars
            tokens.add(token)
    return tokens


class LemmaLibrary:
    """Persistent store for discovered lemma applications.

    Saves to `.bourbaki/lemma-library.json` by default. Supports:
    - add(): save a new lemma
    - search(): find lemmas matching a goal (keyword overlap)
    - record_success(): increment success_count when a lemma is reused
    - save()/load(): JSON persistence
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _DEFAULT_LIBRARY_PATH
        self._entries: list[LemmaEntry] = []
        self._dirty = False
        self.load()

    @property
    def entries(self) -> list[LemmaEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: LemmaEntry) -> None:
        """Save a discovered lemma.

        Deduplicates by checking for an existing entry with the same
        goal_pattern and tactics (exact match). If found, increments the
        existing entry's success_count instead of adding a duplicate.
        """
        # Check for duplicate
        for existing in self._entries:
            if (
                _normalize(existing.goal_pattern) == _normalize(entry.goal_pattern)
                and existing.tactics == entry.tactics
            ):
                existing.success_count += 1
                existing.timestamp = time.time()
                self._dirty = True
                logger.debug(
                    "Lemma deduplicated (success_count=%d): %s",
                    existing.success_count,
                    entry.goal_pattern[:60],
                )
                return

        self._entries.append(entry)
        self._dirty = True
        logger.debug("Lemma added: %s (%d tactics)", entry.goal_pattern[:60], len(entry.tactics))

    def search(self, goal: str, max_results: int = 5) -> list[LemmaEntry]:
        """Find lemmas relevant to a goal.

        Uses a combination of:
        1. Exact normalized match (highest priority)
        2. Substring match (goal contains or is contained in entry)
        3. Keyword overlap (Jaccard-like scoring)

        Returns sorted by (match quality desc, success_count desc).
        """
        if not goal or not self._entries:
            return []

        norm_goal = _normalize(goal)
        goal_keywords = _extract_keywords(goal)

        scored: list[tuple[float, LemmaEntry]] = []

        for entry in self._entries:
            norm_pattern = _normalize(entry.goal_pattern)

            # Exact match
            if norm_pattern == norm_goal:
                score = 100.0 + entry.success_count
                scored.append((score, entry))
                continue

            # Substring match
            if norm_pattern in norm_goal or norm_goal in norm_pattern:
                score = 50.0 + entry.success_count
                scored.append((score, entry))
                continue

            # Keyword overlap
            if goal_keywords:
                entry_keywords = _extract_keywords(entry.goal_pattern)
                if entry_keywords:
                    overlap = len(goal_keywords & entry_keywords)
                    union = len(goal_keywords | entry_keywords)
                    if overlap > 0 and union > 0:
                        jaccard = overlap / union
                        score = jaccard * 30.0 + entry.success_count * 0.5
                        if score > 1.0:
                            scored.append((score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def record_success(self, entry_id: str) -> None:
        """Increment success_count for a lemma that was reused successfully."""
        for entry in self._entries:
            if entry.id == entry_id:
                entry.success_count += 1
                entry.timestamp = time.time()
                self._dirty = True
                logger.debug(
                    "Lemma success recorded (count=%d): %s",
                    entry.success_count,
                    entry.goal_pattern[:60],
                )
                return

    def save(self) -> None:
        """Persist to disk (JSON)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [entry.to_dict() for entry in self._entries]
        self.path.write_text(json.dumps(data, indent=2))
        self._dirty = False
        logger.debug("Lemma library saved: %d entries -> %s", len(self._entries), self.path)

    def load(self) -> None:
        """Load from disk. No-op if file doesn't exist."""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self._entries = [LemmaEntry.from_dict(d) for d in data]
            self._dirty = False
            logger.debug("Lemma library loaded: %d entries from %s", len(self._entries), self.path)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load lemma library from %s: %s", self.path, e)
            self._entries = []

    def save_if_dirty(self) -> None:
        """Save only if there are unsaved changes."""
        if self._dirty:
            self.save()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_library: LemmaLibrary | None = None


def get_lemma_library(path: Path | None = None) -> LemmaLibrary:
    """Get the global lemma library singleton (lazy-initialized)."""
    global _global_library
    if _global_library is None:
        _global_library = LemmaLibrary(path)
    return _global_library


# ---------------------------------------------------------------------------
# Part 2: Shared Lemma Cache (in-memory, per proof session)
# ---------------------------------------------------------------------------


class LemmaCache:
    """In-memory cache of solved subgoals for sharing across sibling subgoals.

    Used during a single proof session (decomposer or coordinator run).
    When one subgoal is solved, its solution is cached so that sibling
    subgoals with the same or similar goal type can reuse it immediately.
    """

    def __init__(self) -> None:
        # Exact match: normalized goal -> tactics
        self._exact: dict[str, list[str]] = {}
        # All entries for substring matching
        self._entries: list[tuple[str, list[str]]] = []  # (normalized_goal, tactics)

    def add(self, goal_type: str, tactics: list[str]) -> None:
        """Cache a solved subgoal."""
        if not goal_type or not tactics:
            return
        norm = _normalize(goal_type)
        self._exact[norm] = tactics
        self._entries.append((norm, tactics))
        logger.debug("LemmaCache: cached %s (%d tactics)", goal_type[:60], len(tactics))

    def lookup(self, goal_type: str) -> list[str] | None:
        """Check if we've already solved a similar goal.

        Returns the tactic sequence if found, None otherwise.
        Priority: exact match first, then substring match.
        """
        if not goal_type:
            return None

        norm = _normalize(goal_type)

        # 1. Exact match
        if norm in self._exact:
            logger.debug("LemmaCache: exact hit for %s", goal_type[:60])
            return self._exact[norm]

        # 2. Substring match
        for cached_goal, tactics in self._entries:
            if cached_goal in norm or norm in cached_goal:
                logger.debug("LemmaCache: substring hit for %s", goal_type[:60])
                return tactics

        return None

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return len(self._entries) > 0

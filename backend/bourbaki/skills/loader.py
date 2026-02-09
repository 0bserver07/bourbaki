"""Skill loader — parses SKILL.md files using python-frontmatter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import frontmatter

SkillSource = Literal["builtin", "user", "project"]


@dataclass
class SkillMetadata:
    """Lightweight skill info (loaded at startup for system prompt)."""
    name: str
    description: str
    path: str
    source: SkillSource


@dataclass
class Skill(SkillMetadata):
    """Full skill with instructions (loaded on-demand)."""
    instructions: str = ""


def parse_skill_file(content: str, path: str, source: SkillSource) -> Skill:
    """Parse a SKILL.md file into a Skill object.

    Args:
        content: Raw file contents (YAML frontmatter + markdown body).
        path: Absolute path to the SKILL.md file.
        source: Where this skill was loaded from.

    Returns:
        Skill with all fields populated.

    Raises:
        ValueError: If required frontmatter fields are missing.
    """
    post = frontmatter.loads(content)

    name = post.metadata.get("name")
    description = post.metadata.get("description")

    if not name:
        raise ValueError(f"SKILL.md at {path} missing required 'name' field")
    if not description:
        raise ValueError(f"SKILL.md at {path} missing required 'description' field")

    return Skill(
        name=name,
        description=description,
        path=path,
        source=source,
        instructions=post.content,
    )


def load_skill_from_path(path: Path, source: SkillSource) -> Skill:
    """Load a skill from a SKILL.md file on disk."""
    content = path.read_text(encoding="utf-8")
    return parse_skill_file(content, str(path), source)


def extract_skill_metadata(path: Path, source: SkillSource) -> SkillMetadata | None:
    """Extract just the metadata from a SKILL.md file (no instructions body).

    Returns None if the file is invalid.
    """
    try:
        content = path.read_text(encoding="utf-8")
        post = frontmatter.loads(content)
        name = post.metadata.get("name")
        description = post.metadata.get("description")
        if not name or not description:
            return None
        return SkillMetadata(
            name=name,
            description=description,
            path=str(path),
            source=source,
        )
    except Exception:
        return None

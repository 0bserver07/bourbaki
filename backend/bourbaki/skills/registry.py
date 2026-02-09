"""Skill registry — discovers and caches skills from all directories."""

from __future__ import annotations

from pathlib import Path

from bourbaki.config import settings
from bourbaki.skills.loader import (
    Skill,
    SkillMetadata,
    SkillSource,
    extract_skill_metadata,
    load_skill_from_path,
)

# Metadata cache: name → SkillMetadata
_cache: dict[str, SkillMetadata] = {}
_discovered = False

# Source priority: later overrides earlier
_SOURCE_ORDER: list[tuple[SkillSource, int]] = [
    ("builtin", 0),
    ("user", 1),
    ("project", 2),
]


def _source_for_dir(dir_path: Path) -> SkillSource:
    """Determine the source type for a skills directory."""
    dir_str = str(dir_path)
    if "/.bourbaki/skills" in dir_str or "\\.bourbaki\\skills" in dir_str:
        # Project-level .bourbaki/skills
        return "project"
    home = Path.home()
    if str(home / ".bourbaki" / "skills") in dir_str:
        return "user"
    return "builtin"


def _scan_directory(directory: Path, source: SkillSource) -> dict[str, SkillMetadata]:
    """Scan a directory for SKILL.md files."""
    skills: dict[str, SkillMetadata] = {}
    if not directory.exists() or not directory.is_dir():
        return skills

    for skill_dir in sorted(directory.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        meta = extract_skill_metadata(skill_file, source)
        if meta:
            skills[meta.name] = meta

    return skills


def discover_skills() -> list[SkillMetadata]:
    """Discover all skills from configured directories.

    Scans builtin → user → project directories. Later sources override
    earlier ones by skill name. Results are cached.

    Returns:
        List of SkillMetadata for all discovered skills.
    """
    global _discovered
    if _discovered:
        return list(_cache.values())

    for dir_path in settings.resolved_skills_dirs:
        source = _source_for_dir(dir_path)
        found = _scan_directory(dir_path, source)
        _cache.update(found)  # Later sources override

    _discovered = True
    return list(_cache.values())


def get_skill(name: str) -> Skill | None:
    """Get a skill by name, loading full instructions on demand.

    Args:
        name: Skill name (e.g. 'proof-by-induction').

    Returns:
        Full Skill with instructions, or None if not found.
    """
    if not _discovered:
        discover_skills()

    meta = _cache.get(name)
    if not meta:
        return None

    path = Path(meta.path)
    if not path.exists():
        return None

    return load_skill_from_path(path, meta.source)


def build_skill_metadata_section() -> str:
    """Build a system prompt section listing all available skills.

    Returns:
        Formatted string for injection into the system prompt.
    """
    skills = discover_skills()
    if not skills:
        return ""

    lines = ["## Available Proof Techniques (Skills)", ""]
    for skill in sorted(skills, key=lambda s: s.name):
        lines.append(f"- **{skill.name}**: {skill.description}")
    lines.append("")
    lines.append(
        "To use a skill, state which technique you want to apply. "
        "The full instructions will be loaded."
    )
    return "\n".join(lines)


def clear_skill_cache() -> None:
    """Clear the skill cache (for testing)."""
    global _discovered
    _cache.clear()
    _discovered = False

"""Skill invocation tool — loads SKILL.md instructions for the agent."""

from __future__ import annotations

from bourbaki.skills.registry import discover_skills, get_skill


def skill_invoke(skill_name: str) -> dict:
    """Load a proof technique skill by name.

    The agent calls this when it wants to use a specific proof technique.
    Returns the full step-by-step instructions from the SKILL.md file.

    Args:
        skill_name: Name of the skill (e.g. 'proof-by-induction').

    Returns:
        Dict with success, name, instructions, or error.
    """
    skill = get_skill(skill_name)
    if skill is None:
        available = [s.name for s in discover_skills()]
        return {
            "success": False,
            "error": f"Skill '{skill_name}' not found",
            "available": available,
        }
    return {
        "success": True,
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
    }

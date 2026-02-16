"""Application configuration using Pydantic Settings."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Bourbaki backend configuration.

    Values are read from environment variables and .env file.
    """

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # LLM API keys
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_cloud_api_key: str | None = None
    glm_api_key: str | None = None
    google_api_key: str | None = None
    xai_api_key: str | None = None

    # Default model
    default_model: str = "openai:gpt-4o"

    # Optional tool API keys
    exasearch_api_key: str | None = None

    # Erdős Problems API (separate service)
    erdos_api_url: str = "http://localhost:3001"

    # Paths
    bourbaki_dir: str = ".bourbaki"
    skills_dirs: list[str] = [
        "src/skills",           # Builtin (project root)
        "../src/skills",        # Builtin (from backend/ dir)
        "~/.bourbaki/skills",   # User
        ".bourbaki/skills",     # Project (highest precedence)
    ]

    # Agent defaults
    max_iterations: int = 10
    tool_call_limit: int = 3  # Default max calls per tool per query

    # Per-tool call limit overrides (tool_name → limit)
    # Lean proof search is iterative — the model needs more attempts to refine proofs
    tool_call_limits: dict[str, int] = {
        "lean_prover": 15,
        "lean_tactic": 30,
        "mathlib_search": 10,
    }

    @property
    def bourbaki_path(self) -> Path:
        return Path(self.bourbaki_dir)

    @property
    def resolved_skills_dirs(self) -> list[Path]:
        return [Path(d).expanduser() for d in self.skills_dirs]

    model_config = {
        "env_file": [".env", "../.env"],  # backend/.env then project root
        "env_prefix": "",
        "extra": "ignore",
    }


settings = Settings()


def export_api_keys() -> None:
    """Export API keys to environment variables for LLM SDK discovery.

    Pydantic AI / OpenAI / Anthropic SDKs read keys from env vars.
    Our Settings loads from .env, so we bridge the gap here.
    """
    _key_map = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "OLLAMA_CLOUD_API_KEY": settings.ollama_cloud_api_key,
        "GLM_API_KEY": settings.glm_api_key,
        "GOOGLE_API_KEY": settings.google_api_key,
        "XAI_API_KEY": settings.xai_api_key,
    }
    for env_var, value in _key_map.items():
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value


# Auto-export on import
export_api_keys()

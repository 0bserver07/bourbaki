# Contributing to Bourbaki

Thanks for your interest in contributing to Bourbaki.

## Setup

### Prerequisites

- Python 3.11+
- [Bun](https://bun.sh/) (for the TUI)
- At least one LLM API key (see below)
- Optional: [Lean 4](https://leanprover.github.io/) + Mathlib for formal verification

### Environment

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

**API keys** (at least one LLM key required):

| Key | Service | Required |
|-----|---------|----------|
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/) — access many models with one key | Recommended |
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com/) — GPT-4, GPT-5, etc. | Or any other LLM key |
| `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com/) — Claude models | Or any other LLM key |
| `GOOGLE_API_KEY` | [Google AI](https://aistudio.google.com/) — Gemini models | Optional |
| `XAI_API_KEY` | [xAI](https://console.x.ai/) — Grok models | Optional |
| `OLLAMA_CLOUD_API_KEY` | [Ollama Cloud](https://ollama.com/) | Optional |
| `EXASEARCH_API_KEY` | [Exa](https://exa.ai/) — web/paper search | Optional |
| `TAVILY_API_KEY` | [Tavily](https://tavily.com/) — alternative search | Optional |

For local Ollama, no key needed — just have `ollama serve` running.

For Lean 4 verification, set `LEAN_PATH` to a Lean project with Mathlib.

### Install and run

```bash
# Backend
cd backend
pip install -e ".[dev]"
uvicorn bourbaki.main:app --reload --port 8000

# TUI (separate terminal)
bun install
bun start
```

### Run tests

```bash
# Python backend tests
cd backend && pytest

# TypeScript type checking
bun run typecheck
```

## Making changes

1. Fork the repo and create a branch from `master`
2. Make your changes
3. Run `pytest` and `bun run typecheck` to verify nothing is broken
4. Open a PR against `master`

### Code style

- Python: formatted with [Ruff](https://docs.astral.sh/ruff/) (`ruff check` / `ruff format`)
- TypeScript: standard TypeScript with strict mode

### Project structure

- `backend/bourbaki/` — Python backend (FastAPI + Pydantic AI)
- `src/` — TypeScript TUI (React + Ink)
- `src/skills/` — SKILL.md proof technique files

See `ARCHITECTURE.md` for detailed system design.

## Adding a proof technique skill

Skills are Markdown files in `src/skills/`. To add one:

1. Create `src/skills/your-technique/SKILL.md`
2. Add YAML frontmatter with `name`, `description`, `tags`
3. Write the proof strategy instructions in the body
4. The backend discovers it automatically — no code changes needed

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

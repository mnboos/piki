# piki — Claude Code guidance

## Python environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**Always use `uv run` instead of `python` / `python3` directly.**

```bash
# Run a script or command
uv run python src/manage.py runserver

# Run a one-liner
uv run python -c "from core.models import DetectionConfig; ..."

# Run tests
uv run pytest
```

Do not call `pip install`, `python -m venv`, or bare `python3` — uv handles the
virtual environment and lock file automatically.

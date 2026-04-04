# piki — Copilot instructions

## Python environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**Always use `uv run` instead of `python` / `python3` directly.**

```bash
# Run a script or command
uv run python src/manage.py runserver

# Run a one-liner
uv run python -c "..."

# Run tests
uv run pytest
```

Do not suggest `pip install`, bare `python3`, or manual venv setup — uv manages
the environment and lock file.

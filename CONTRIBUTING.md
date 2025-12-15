# Contributing

Thanks for your interest in contributing! This project aims to make RAG experiments easy to run and extend.

## Getting Started
- Use Python 3.10+ in a virtual environment.
- Install dependencies: `pip install -r requirements.txt`.
- Run a smoke test: `python test_setup.py`.

## Development Workflow
- Branch from `main` using a descriptive name (e.g., `feature/rag-topk-visuals`).
- Keep changes focused and small.
- Add/adjust config files under `config/` rather than hard-coding.
- Update docs if behavior or usage changes.

## Code Style
- Prefer small, composable functions.
- Avoid one-letter variable names.
- Keep public APIs stable; avoid breaking changes unless necessary.

## Testing
- Add or update minimal tests and run locally.
- For notebooks, ensure any heavy runs are optional or sample-sized.

## Pull Requests
- Describe the change clearly and link related issues.
- Include screenshots for UX/doc changes where helpful.
- Ensure CI passes.

## Issues
- Use clear titles, steps to reproduce, expected vs actual behavior.
- Include environment details (OS, Python version).

## Thank You
Your contributions help make this project more useful and reproducible!
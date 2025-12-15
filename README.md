# RAG Analysis

An end-to-end, reproducible analysis of Retrieval-Augmented Generation (RAG) on HotpotQA, including a modular pipeline, evaluation tooling, and methodology notes. This repo is structured for easy setup, fast experimentation, and clear results communication.

## Features
- Modular RAG pipeline in `src/`
- Config-driven runs via `config/config.yaml` and `config/config_fast.yaml`
- Reproducible environment with `requirements.txt` and `setup.sh`
- Evaluation reports and aggregated metrics
- Notebook walkthroughs for data exploration, experiments, and visualization

## Quickstart
1. Create a Python environment (recommended: Python 3.10+).
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run a fast test experiment:

```
python main.py --config config/config_fast.yaml
```

4. Full experiment (longer runtime):

```
python main.py --config config/config.yaml
```

Optional: see `QUICKSTART.md` for more details.

## Project Structure
- `src/`: Core modules (`data_loader.py`, `rag_pipeline.py`, `evaluator.py`, `utils.py`)
- `config/`: YAML configs for fast and full runs
- `data/`: Small sample datasets for local tests; see notes below
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `results/`: Lightweight sample outputs for documentation
- `experiments/`, `logs/`, `checkpoints/`: Generated during runs (excluded via `.gitignore`)

## Data Notes
- Small samples are included under `data/` to validate the pipeline.
- For full HotpotQA experiments, provide dataset paths via `config/config.yaml` or adjust `data_loader.py` accordingly.
- Large datasets are not committed; download instructions are in `METHODOLOGY.md`.

## Running Experiments
- Entry point: `main.py`
- Args: `--config` to select run configuration.
- Outputs: experiment folders under `experiments/` with `results/`, `logs/`, and `checkpoints/`.

## Development
- Use a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
- Style: keep functions small and modular; avoid one-letter variables.
- Tests: minimal smoke test can be run via `python test_setup.py`.

## Contributing
See `CONTRIBUTING.md` for guidelines on issues, PRs, and local setup.

## License
See `LICENSE` for terms.

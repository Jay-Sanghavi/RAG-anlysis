# RAG Analysis

An end-to-end, reproducible analysis of Retrieval-Augmented Generation (RAG) on HotpotQA, including a modular pipeline, evaluation tooling, and methodology notes. This repo is structured for easy setup, fast experimentation, and clear results communication.

[![CI](https://github.com/Jay-Sanghavi/RAG-anlysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Jay-Sanghavi/RAG-anlysis/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## Features
- Modular RAG pipeline in `src/`
- Config-driven runs via `config/config.yaml` and `config/config_fast.yaml`
- Reproducible environment with `requirements.txt` and `setup.sh`
- Evaluation reports and aggregated metrics
- Notebook walkthroughs for data exploration, experiments, and visualization

## Results Snapshot
- Fast subset (n=50) shows RAG improves correctness and reduces hallucinations versus no-RAG.
- Summary from [results/aggregated_metrics_fast.json](results/aggregated_metrics_fast.json):
	- no_rag: EM 0.00, F1 0.086, hallucination 1.00
	- rag_k1: EM 0.12, F1 0.190, hallucination 0.82, recall@k 0.23, MRR 0.46
	- rag_k3: EM 0.14, F1 0.212, hallucination 0.76, recall@k 0.25, MRR 0.46
	- rag_k5: EM 0.14, F1 0.215, hallucination 0.76, recall@k 0.25, MRR 0.46
- Detailed per-item metrics: [results/evaluation_results_fast.csv](results/evaluation_results_fast.csv)

## Why This Project Matters
- Shows practical gains from retrieval for small open-source LLMs on multi-hop QA.
- Connects retrieval quality (recall@k, MRR) to correctness and hallucination rate.
- Provides a clean, configurable pipeline interviewers can scan and run quickly.

## My Role and Key Contributions
- Designed experiment methodology and evaluation rubric; implemented `src/evaluator.py` metrics (EM, F1, hallucination categories).
- Built modular RAG components (`src/rag_pipeline.py`) integrating retrieval with generation.
- Implemented data preparation in `src/data_loader.py` and config-driven runs via `main.py`.
- Authored documentation (`README.md`, `METHODOLOGY.md`, `QUICKSTART.md`) and CI for reproducibility.

## Tech Stack
- Python, Jupyter Notebooks
- Retrieval: FAISS / SentenceTransformers (configurable)
- LLMs: TinyLlama/Mistral variants (configurable)
- CI: GitHub Actions minimal smoke test

## Architecture Overview
- `data_loader.py`: load subsets and prepare corpus
- `rag_pipeline.py`: build index, retrieve top-k, generate answers
- `evaluator.py`: compute EM, F1, hallucination labels, recall@k, MRR
- `main.py`: config-driven orchestration, writes `experiments/` outputs

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

## Reproducibility Notes
- Use `config/config_fast.yaml` for a quick run; `config/config.yaml` for full experiments.
- Large datasets and model caches are not committed; see `METHODOLOGY.md` for dataset info.
- Heavy outputs are excluded via `.gitignore`; only light sample results are included.

## Interviewer Guide
- Skim `README.md` and `METHODOLOGY.md` for scope and design.
- Review core modules in `src/` to see separation of concerns.
- Check [results/aggregated_metrics_fast.json](results/aggregated_metrics_fast.json) for the improvement from RAG.
- Run a smoke test locally: `python test_setup.py`.

## Development
- Use a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
- Style: keep functions small and modular; avoid one-letter variables.
- Tests: minimal smoke test can be run via `python test_setup.py`.

## Contributing
See `CONTRIBUTING.md` for guidelines on issues, PRs, and local setup.

## License
See `LICENSE` for terms.

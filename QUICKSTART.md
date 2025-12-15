# Quick Reference Guide

## Project Structure

```
RAG-anlysis/
├── config/
│   └── config.yaml              # Main configuration file
├── src/
│   ├── data_loader.py           # Load and preprocess HotpotQA
│   ├── rag_pipeline.py          # RAG implementation
│   ├── evaluator.py             # Evaluation metrics
│   └── utils.py                 # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_experiments.ipynb
│   └── 03_results_visualization.ipynb
├── data/                        # Generated data files
├── results/                     # Experiment results
├── main.py                      # Main execution script
└── test_setup.py               # Environment checker
```

## Quick Commands

### Setup
```bash
# Create virtual environment and install dependencies
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Test Setup
```bash
python test_setup.py
```

### Run Experiments

**Full pipeline:**
```bash
python main.py
```

**Interactive notebooks:**
```bash
jupyter notebook
```

### Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"  # Change model
  load_in_8bit: false  # Enable for less memory

dataset:
  subset_size: 100  # Number of examples (50-100)

retrieval:
  k_values: [1, 3, 5]  # Different k to test
```

## Workflow

### 1. Data Preparation
```python
from src.data_loader import HotpotQALoader

loader = HotpotQALoader(subset_size=100, random_seed=42)
loader.load_dataset(split='validation')
subset = loader.create_subset(strategy='random')
corpus = loader.prepare_corpus()
```

### 2. Build Index
```python
from src.rag_pipeline import VectorStore

vector_store = VectorStore()
vector_store.build_index(corpus)
```

### 3. Run Inference
```python
from src.rag_pipeline import LLMGenerator, RAGPipeline

generator = LLMGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.2")
pipeline = RAGPipeline(vector_store, generator)

# No RAG
result = pipeline.answer_without_rag(question)

# With RAG
result = pipeline.answer_with_rag(question, k=3)
```

### 4. Evaluate
```python
from src.evaluator import Evaluator

evaluator = Evaluator()
evaluator.evaluate_single(example, prediction, condition='rag_k3')
metrics = evaluator.aggregate_results()
```

## Memory Management

If you encounter OOM errors:

1. **Enable quantization:**
   ```yaml
   model:
     load_in_8bit: true
   ```

2. **Use smaller model:**
   ```yaml
   model:
     name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

3. **Reduce subset:**
   ```yaml
   dataset:
     subset_size: 50
   ```

## Expected Results

After running `main.py`:

- `data/hotpotqa_subset.json` - Dataset subset
- `data/corpus.json` - Text corpus
- `data/faiss_index.bin` - Vector index
- `results/evaluation_results.csv` - Per-item results
- `results/aggregated_metrics.json` - Summary metrics
- `experiments/[timestamp]/` - Full experiment output

## Evaluation Metrics

- **EM (Exact Match):** Binary correctness score
- **F1:** Token-level F1 score
- **Hallucination Rate:** Percentage of incorrect/unsupported answers
- **Recall@k:** Retrieval quality metric
- **MRR:** Mean reciprocal rank

## Statistical Tests

- **McNemar's Test:** Compare paired correctness between conditions
- **Spearman Correlation:** Retrieval quality vs correctness
- **Bootstrap CI:** Effect size with confidence intervals

## Troubleshooting

### ImportError
```bash
pip install -r requirements.txt --upgrade
```

### CUDA out of memory
Enable 8-bit quantization or use CPU:
```yaml
model:
  load_in_8bit: true
  device: "cpu"
```

### Slow inference
- Use GPU (CUDA or MPS)
- Reduce max_new_tokens
- Use smaller model
- Test on smaller subset first

### Retrieval errors
Check that corpus is prepared:
```python
corpus = loader.prepare_corpus()
```

## Citation

```bibtex
@misc{rag-analysis-2025,
  title={RAG Analysis: Evaluating Retrieval-Augmented Generation},
  year={2025}
}
```

## Support

- Documentation: README.md, METHODOLOGY.md
- Issues: Check test_setup.py output
- Configuration: config/config.yaml

---

**Last Updated:** December 2025

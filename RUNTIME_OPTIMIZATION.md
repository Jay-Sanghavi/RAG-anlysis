# Runtime Optimization Guide

## Expected Runtime Analysis

### Default Configuration (`config/config.yaml`)
- **Subset size:** 100 examples
- **Conditions:** 4 (no_rag, rag_k1, rag_k3, rag_k5)
- **Total inference calls:** 400

**Estimated Runtime:**
- **With GPU (CUDA/MPS):** 45-90 minutes
- **With CPU only:** 3-5 hours

### Breakdown by Step:
| Step | Task | GPU Time | CPU Time |
|------|------|----------|----------|
| 1 | Load dataset | 2-5 min | 2-5 min |
| 2 | Build FAISS index | 1-2 min | 1-2 min |
| 3 | Load LLM (first time) | 3-5 min | 3-5 min |
| 4 | Create pipeline | <1 min | <1 min |
| 5 | **Inference** | **35-65 min** | **2.5-4 hours** |
| 6-8 | Evaluation & reporting | 2-5 min | 2-5 min |
| **TOTAL** | | **45-90 min** | **3-5 hours** |

---

## 🚀 Speed-Up Strategies

### Strategy 1: Fast Test Configuration ⚡ (RECOMMENDED)
**Runtime: 5-10 minutes** | **Speedup: 80-90%**

Use the pre-configured fast test setup:

```bash
python main.py config/config_fast.yaml
```

**Changes:**
- ✓ Subset size: 100 → 20 (80% reduction in inference)
- ✓ K values: [1,3,5] → [3] (67% reduction in conditions)
- ✓ 8-bit quantization: enabled (15-25% faster per call)
- ✓ Max tokens: 100 → 50 (10-15% faster per call)
- ✓ Conditions: 4 → 2 (50% reduction)

**Total inference calls: 400 → 40** (90% reduction)

---

### Strategy 2: Moderate Optimization
**Runtime: 20-35 minutes** | **Speedup: 50-60%**

Edit `config/config.yaml`:

```yaml
dataset:
  subset_size: 50  # Half the examples, still statistically valid

model:
  load_in_8bit: true  # Enable quantization
  max_new_tokens: 50  # Shorter answers

retrieval:
  k_values: [3]  # Focus on most important k
```

**Total inference calls: 400 → 100** (75% reduction)

---

### Strategy 3: Minimal Changes (Easy)
**Runtime: 30-50 minutes** | **Speedup: 35-45%**

Just enable 8-bit quantization and reduce subset:

```yaml
dataset:
  subset_size: 50

model:
  load_in_8bit: true
```

**Total inference calls: 400 → 200** (50% reduction)

---

### Strategy 4: Ultra-Fast Development Testing
**Runtime: 2-3 minutes** | **Speedup: 95%+**

For development/debugging only (not for research):

```yaml
dataset:
  subset_size: 5  # Minimal test

retrieval:
  k_values: [3]

model:
  load_in_8bit: true
  max_new_tokens: 30
```

---

## Performance Comparison Table

| Configuration | Examples | Conditions | Inference Calls | GPU Time | CPU Time |
|--------------|----------|------------|-----------------|----------|----------|
| Default | 100 | 4 | 400 | 45-90 min | 3-5 hours |
| Fast Test | 20 | 2 | 40 | 5-10 min | 30-45 min |
| Moderate | 50 | 2 | 100 | 20-35 min | 1.5-2.5 hours |
| Minimal | 50 | 4 | 200 | 30-50 min | 2-3 hours |
| Ultra-Fast | 5 | 2 | 10 | 2-3 min | 8-12 min |

---

## Hardware-Specific Recommendations

### Apple Silicon (M1/M2/M3)
✓ Already optimized! The config uses `device: "auto"` which enables MPS.

**Recommended config:**
```yaml
model:
  device: "auto"  # Uses MPS automatically
  load_in_8bit: true  # Helps with memory
```

Expected: **20-40 minutes** for fast test config

### NVIDIA GPU (CUDA)
✓ Best performance. Should work out of the box.

**Expected:** 5-10 minutes for fast test config

### CPU Only (No GPU)
⚠️ Will be slow. Use aggressive optimizations.

**Strongly recommended:**
```yaml
dataset:
  subset_size: 20  # Keep small

model:
  load_in_8bit: true  # Essential
  max_new_tokens: 30

retrieval:
  k_values: [3]
```

Expected: **30-45 minutes** for fast test config

---

## Memory Usage

### GPU Memory Requirements

| Configuration | 8-bit Quantization | VRAM Needed |
|--------------|-------------------|-------------|
| Mistral-7B | OFF | ~14-16 GB |
| Mistral-7B | ON | ~7-8 GB |
| Llama-3-8B | OFF | ~16-18 GB |
| Llama-3-8B | ON | ~8-9 GB |

### RAM Requirements

- **Minimum:** 8 GB (with 8-bit quantization)
- **Recommended:** 16 GB
- **Comfortable:** 32 GB

---

## Running Fast Tests

### Quick Test Run
```bash
# Use fast configuration
python main.py config/config_fast.yaml
```

### Interactive Testing (Notebooks)
In `notebooks/02_rag_experiments.ipynb`, set:
```python
TEST_SIZE = 5  # Quick test with 5 examples
```

### Custom Configuration
1. Copy the config:
   ```bash
   cp config/config.yaml config/config_custom.yaml
   ```

2. Edit `config_custom.yaml` with your preferences

3. Run with custom config:
   ```bash
   python main.py config/config_custom.yaml
   ```

---

## Statistical Validity vs Speed

### Research Quality (Recommended: 50-100 examples)
- **100 examples:** Best statistical power, publishable results
- **50 examples:** Good balance, still statistically valid
- **20 examples:** Sufficient for initial testing, lower power

### K Values
- **[1, 3, 5]:** Complete analysis (as per proposal)
- **[3]:** Most important value, good for quick tests
- **[1, 5]:** Test extremes only

### McNemar's Test Power
- **100 examples:** Detects 10% difference with 80% power
- **50 examples:** Detects 15% difference with 80% power
- **20 examples:** Detects 25% difference with 80% power

---

## Bottleneck Identification

The **inference step (Step 5)** accounts for 80-90% of total runtime.

**Per-example time:**
- Model loading: ~3-5 seconds (one-time)
- Retrieval: ~0.1-0.3 seconds (very fast)
- Generation: **5-10 seconds (GPU)** or **45-60 seconds (CPU)** ← BOTTLENECK

**What doesn't help much:**
- ❌ Faster CPU/SSD (marginal)
- ❌ More RAM (if you already have enough)
- ❌ Better encoder model (retrieval is fast already)

**What helps a lot:**
- ✅ GPU acceleration (5-10× faster)
- ✅ 8-bit quantization (15-25% faster)
- ✅ Fewer examples (linear speedup)
- ✅ Fewer conditions (linear speedup)
- ✅ Shorter generation (10-20% faster)

---

## Recommendations by Use Case

### 🎓 Final Research Run (For Thesis/Paper)
**Config:** Default (`config/config.yaml`)
**Time:** 45-90 minutes (GPU)
**Justification:** Full statistical power, complete analysis

### 🔬 Initial Testing / Development
**Config:** Fast test (`config/config_fast.yaml`)
**Time:** 5-10 minutes (GPU)
**Justification:** Quick iteration, validate pipeline

### ⚖️ Balanced Approach
**Config:** Moderate (50 examples, k=3)
**Time:** 20-35 minutes (GPU)
**Justification:** Good results in reasonable time

### 🚀 Continuous Integration / Debugging
**Config:** Ultra-fast (5 examples)
**Time:** 2-3 minutes (GPU)
**Justification:** Catch bugs quickly

---

## Example: From 90 min → 10 min

**Original (`config/config.yaml`):**
```yaml
dataset:
  subset_size: 100
model:
  load_in_8bit: false
  max_new_tokens: 100
retrieval:
  k_values: [1, 3, 5]
```
**Runtime:** ~90 minutes (GPU)

**Optimized (`config/config_fast.yaml`):**
```yaml
dataset:
  subset_size: 20
model:
  load_in_8bit: true
  max_new_tokens: 50
retrieval:
  k_values: [3]
```
**Runtime:** ~10 minutes (GPU)

**Speedup:** 9× faster! ⚡

---

## Commands Summary

```bash
# Full research run (45-90 min)
python main.py

# Fast test run (5-10 min) - RECOMMENDED FOR TESTING
python main.py config/config_fast.yaml

# Custom configuration
python main.py config/config_custom.yaml

# Check your setup first
python test_setup.py
```

---

## Pro Tips

1. **Always test with fast config first** to ensure everything works
2. **Monitor GPU usage** with `nvidia-smi` or Activity Monitor
3. **Save intermediate results** - index and corpus are reusable
4. **Use notebooks for debugging** - easier to iterate
5. **Profile if needed:** Add timing to identify actual bottlenecks

---

**Updated:** December 2025

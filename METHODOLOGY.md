# Methodology

## Overview

This document describes the detailed methodology for evaluating Retrieval-Augmented Generation (RAG) as a method to reduce hallucinations in small open-source Large Language Models (LLMs).

## Research Questions and Hypotheses

### RQ1: RAG Impact on Hallucinations and Accuracy

**Question:** Does adding retrieval (RAG) reduce hallucinations and increase factual accuracy for small open-source LLMs on multi-hop QA?

**Hypothesis (H1):** RAG will reduce hallucination rate and increase accuracy by at least 10 percentage points versus no-RAG on a 50–100 item subset.

### RQ2: Retrieval Quality and Correctness

**Question:** How does retrieval quality relate to correctness?

**Hypothesis (H2):** Higher retrieval recall@k correlates positively with factual correctness and negatively with hallucinations.

### RQ3: Sensitivity to Retrieval Depth

**Question:** How sensitive are results to retrieval depth and context size?

**Hypothesis (H3):** Moderate k (e.g., 3–5 chunks) outperforms very small or very large k due to context dilution at large k.

## Dataset

### HotpotQA

- **Source:** Hugging Face Datasets (`hotpot_qa`, `fullwiki` subset)
- **License:** MIT License
- **Split:** Validation set
- **Subset Size:** 50–100 examples (configurable)

#### Dataset Selection Rationale

HotpotQA was chosen because:

1. **Multi-hop reasoning:** Questions require connecting information from multiple passages
2. **Ground truth:** Verified answers with supporting facts
3. **Factual questions:** Enables objective correctness evaluation
4. **Wikipedia-grounded:** Supporting passages from Wikipedia
5. **Widely used:** Established benchmark for QA systems

#### Subset Sampling Strategy

**Random Sampling (Default):**
- Select N examples uniformly at random from validation split
- Set random seed (42) for reproducibility
- Ensures diversity across question types

**Justification for Subset Size:**
- 50–100 examples balances statistical power with evaluation feasibility
- Enables manual hallucination annotation within reasonable time
- Sufficient for paired statistical tests (McNemar's test)
- Small enough for detailed error analysis

#### Handling Answer Variations

HotpotQA answers may vary in format. We address this by:

1. **Normalization:** 
   - Convert to lowercase
   - Remove articles (a, an, the)
   - Remove punctuation
   - Standardize whitespace

2. **Multiple Metrics:**
   - Exact Match (strict)
   - Token F1 (lenient, handles paraphrases)

3. **Manual Review:**
   - Spot-check 10-20 ambiguous cases
   - Document edge cases in error analysis

## Systems Compared

### Models

**Primary Model:** Mistral-7B-Instruct-v0.2
- Open-source, commercially usable
- Strong instruction-following capabilities
- Reasonable size for local inference

**Alternative Models (if needed):**
- Meta-Llama/Llama-3-8B-Instruct
- Smaller models if compute is limited

**Model Configuration:**
- Quantization: Optional 8-bit for memory efficiency
- Temperature: 0.7 (configurable)
- Max tokens: 100 (short-form answers)
- Deterministic seed: 42

### Experimental Conditions

1. **No-RAG Baseline:** Direct generation from question prompt
2. **RAG k=1:** Retrieve and use top 1 passage
3. **RAG k=3:** Retrieve and use top 3 passages
4. **RAG k=5:** Retrieve and use top 5 passages

## Retrieval Setup

### Text Preprocessing

**Chunking Strategy:**
- Chunk size: 200-400 tokens (approximated by words)
- Overlap: 20-30% (75-100 tokens)
- Rationale: Balance context completeness with retrieval precision

**Corpus Construction:**
- Extract all passages from HotpotQA context
- Each passage: Wikipedia article section
- Maintain metadata: title, source question ID

### Embedding and Indexing

**Encoder:** SentenceTransformers `all-MiniLM-L6-v2`
- Fast, efficient (384 dimensions)
- Good balance of speed and quality
- Widely used baseline

**Vector Store:** FAISS
- IndexFlatIP (inner product / cosine similarity)
- L2 normalization for cosine similarity
- Efficient for corpus size (~1000-5000 passages)

### Retrieval Process

1. Encode query (question) with same encoder
2. Retrieve top-k passages by cosine similarity
3. Concatenate passage texts as context
4. Format context in prompt template

## Inference Protocol

### Prompt Templates

**No-RAG Prompt:**
```
Answer the following question concisely and factually.

Question: {question}

Answer:
```

**RAG Prompt:**
```
Answer the following question based on the provided context. Be concise and factual.

Context:
{retrieved_passages}

Question: {question}

Answer:
```

### Generation Parameters

- **Temperature:** 0.7 (moderate randomness)
- **Max new tokens:** 100
- **Sampling:** Enabled (temperature > 0)
- **Seed:** Fixed per condition for reproducibility
- **Repetitions:** 1 per question per condition (may increase if time permits)

### Reproducibility Measures

- Fixed random seeds (data sampling, model inference)
- Deterministic tokenization
- Save exact model versions
- Log all hyperparameters
- Version control for code

## Evaluation Metrics

### Accuracy Metrics

#### Exact Match (EM)

**Definition:** Binary score (1 if normalized prediction exactly matches normalized ground truth, else 0)

**Calculation:**
```python
normalize(prediction) == normalize(ground_truth)
```

**Interpretation:** Strict correctness measure

#### Token F1

**Definition:** Token-level precision and recall between prediction and ground truth

**Calculation:**
```python
precision = common_tokens / prediction_tokens
recall = common_tokens / ground_truth_tokens
F1 = 2 * precision * recall / (precision + recall)
```

**Interpretation:** Lenient measure, captures partial correctness

### Hallucination Detection

#### Definition

**Hallucination:** Model output that is factually incorrect or unsupported by evidence

**Categories:**

1. **Factual Error:** Contains factually incorrect information
2. **Unsupported:** Not supported by provided context (RAG) or known facts (no-RAG)
3. **Non-Answer:** Model refuses to answer or provides no information
4. **Correct:** Factually accurate and supported

#### Detection Method

**Primary:** Rule-based classifier with manual validation

**Rules:**
- Non-answer: Contains phrases like "I don't know", "cannot answer"
- High F1 (>0.5): Likely correct
- Low context overlap (<0.3): Likely unsupported (for RAG)
- Low F1: Likely factual error

**Validation:**
- Manual annotation of all 50-100 examples
- Use rubric for consistency
- Optional: LLM-as-judge for initial labels
- Human review of disagreements (10-20 cases)

#### Hallucination Rate

**Calculation:**
```python
hallucination_rate = n_hallucinations / n_total_examples
```

### Retrieval Quality Metrics

#### Recall@k

**Definition:** Percentage of gold supporting facts retrieved in top-k passages

**Calculation:**
```python
recall@k = |retrieved_titles[:k] ∩ supporting_titles| / |supporting_titles|
```

**Interpretation:** Measures retrieval effectiveness

#### Mean Reciprocal Rank (MRR)

**Definition:** Reciprocal of rank of first relevant passage

**Calculation:**
```python
MRR = 1 / rank_of_first_relevant_passage
```

**Interpretation:** Measures ranking quality

## Statistical Testing

### McNemar's Test

**Purpose:** Compare paired binary outcomes (correct/incorrect) between conditions

**Null Hypothesis:** No difference in correctness between conditions

**Test Statistic:**
```
χ² = (|n_a_only - n_b_only| - 1)² / (n_a_only + n_b_only)
```

Where:
- n_a_only: Examples correct in condition A only
- n_b_only: Examples correct in condition B only

**Significance Level:** α = 0.05

**Application:** Compare no-RAG vs each RAG condition

### Correlation Analysis

**Purpose:** Assess relationship between retrieval quality and correctness

**Method:** Spearman rank correlation

**Variables:**
- X: Retrieval recall@k per example
- Y: Correctness (EM or F1) per example

**Interpretation:** 
- ρ > 0: Positive correlation (better retrieval → better accuracy)
- Statistical significance via p-value < 0.05

### Effect Size

**Bootstrap Confidence Intervals:**
- 95% CI for difference in accuracy between conditions
- 1000 bootstrap samples
- Percentile method

## Reporting

### Results Tables

**Metrics by Condition:**

| Condition | EM | F1 | Hallucination % | Recall@k | MRR |
|-----------|-----|-----|-----------------|----------|-----|
| No-RAG    | ... | ... | ...             | N/A      | N/A |
| RAG k=1   | ... | ... | ...             | ...      | ... |
| RAG k=3   | ... | ... | ...             | ...      | ... |
| RAG k=5   | ... | ... | ...             | ...      | ... |

**Statistical Comparisons:**

| Comparison | EM Δ | F1 Δ | Hall% Δ | p-value | Significant? |
|------------|------|------|---------|---------|--------------|
| No-RAG vs RAG k=3 | ... | ... | ... | ... | ... |

### Visualizations

1. **Accuracy vs k:** Line plot showing EM/F1 across k values
2. **Hallucination vs k:** Line plot showing hallucination rate vs k
3. **Retrieval quality distribution:** Histogram of recall@k scores
4. **Correlation scatter:** Recall@k vs EM with regression line

### Error Analysis

**Selection:** 5-10 representative failure cases per condition

**Documentation:**
- Question
- Gold answer
- Model prediction
- Retrieved passages (if applicable)
- Error type
- Analysis notes

**Categories:**
- RAG helped: Incorrect without RAG, correct with RAG
- RAG hurt: Correct without RAG, incorrect with RAG
- Both failed: Incorrect in both conditions
- Retrieval failure: Relevant passages not retrieved

## Reproducibility Checklist

- [ ] Fixed random seeds documented
- [ ] Exact model versions recorded
- [ ] Dataset subset committed to repository
- [ ] Configuration file version controlled
- [ ] Hyperparameters logged
- [ ] Environment requirements.txt provided
- [ ] Evaluation rubric documented
- [ ] Statistical methods specified
- [ ] Code publicly available
- [ ] Results data tables published

## Limitations

1. **Small scale:** 50-100 examples limits generalizability
2. **Single domain:** HotpotQA is Wikipedia-based
3. **Manual annotation:** Hallucination detection requires human effort
4. **Model selection:** Results specific to tested models
5. **Computational constraints:** May limit model size or repetitions

## Ethical Considerations

- **Dataset:** No personal data; MIT licensed
- **Attribution:** Acknowledge HotpotQA creators
- **Transparency:** Document all limitations
- **Reproducibility:** Provide complete methods for verification
- **No harm:** Academic research; no deployment to users

## References

1. Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
2. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
3. Ji et al. (2023). Survey of Hallucination in Natural Language Generation.
4. McNemar (1947). Note on the sampling error of the difference between correlated proportions.

---

**Version:** 1.0  
**Last Updated:** December 2025

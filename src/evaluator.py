"""
Evaluation metrics for RAG analysis.
Includes accuracy, hallucination detection, and retrieval quality metrics.
"""

import re
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from scipy import stats
import pandas as pd


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Args:
        text: Raw answer text

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(len(pred_tokens) == len(truth_tokens))

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_retrieval_recall(
    retrieved_passages: List[Dict], supporting_facts: Dict, k: int = 5
) -> float:
    """
    Compute recall@k for retrieved passages against supporting facts.

    Args:
        retrieved_passages: List of retrieved passage dictionaries
        supporting_facts: Dictionary with 'title' field containing supporting titles
        k: Number of top passages to consider

    Returns:
        Recall@k score
    """
    if not retrieved_passages or not supporting_facts:
        return 0.0

    # Get supporting titles
    supporting_titles = set(supporting_facts["title"])

    # Get retrieved titles (top-k)
    retrieved_titles = set([p.get("title", "") for p in retrieved_passages[:k]])

    # Calculate recall
    if len(supporting_titles) == 0:
        return 0.0

    overlap = len(supporting_titles & retrieved_titles)
    recall = overlap / len(supporting_titles)

    return recall


def compute_mrr(retrieved_passages: List[Dict], supporting_facts: Dict) -> float:
    """
    Compute Mean Reciprocal Rank for retrieved passages.

    Args:
        retrieved_passages: List of retrieved passage dictionaries
        supporting_facts: Dictionary with 'title' field

    Returns:
        MRR score
    """
    if not retrieved_passages or not supporting_facts:
        return 0.0

    supporting_titles = set(supporting_facts["title"])

    for rank, passage in enumerate(retrieved_passages, 1):
        if passage.get("title", "") in supporting_titles:
            return 1.0 / rank

    return 0.0


class HallucinationDetector:
    """Detect hallucinations in model outputs."""

    def __init__(self):
        """Initialize hallucination detector."""
        self.rubric = {
            "factual_error": "Answer contains factually incorrect information",
            "unsupported": "Answer is not supported by provided context (for RAG)",
            "non_answer": "Model refuses to answer or provides no information",
            "correct": "Answer is factually correct and supported",
        }

    def classify_hallucination(
        self, prediction: str, ground_truth: str, context: str = None
    ) -> Dict:
        """
        Classify whether prediction contains hallucination.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            context: Retrieved context (for RAG evaluation)

        Returns:
            Dictionary with classification and reasoning
        """
        # Simplified rule-based classification
        # In practice, this would use LLM-as-judge or human annotation

        pred_norm = normalize_answer(prediction)
        truth_norm = normalize_answer(ground_truth)

        # Heuristic thresholds (tunable)
        F1_CORRECT = 0.60  # Above -> correct
        F1_PARTIAL = 0.40  # Partial correctness band
        CONTEXT_SUPPORT_RATIO = 0.50  # Fraction of prediction tokens appearing in context to consider supported
        MIN_CONTEXT_OVERLAP = 0.20  # Below -> unsupported

        # Check for non-answer
        non_answer_patterns = [
            "i don't know",
            "cannot answer",
            "no information",
            "unable to",
            "not sure",
            "unclear",
        ]
        if any(pattern in pred_norm for pattern in non_answer_patterns):
            return {
                "is_hallucination": True,
                "category": "non_answer",
                "reasoning": "Model refused to answer or expressed uncertainty",
            }

        f1 = token_f1(prediction, ground_truth)

        if f1 >= F1_CORRECT:
            return {
                "is_hallucination": False,
                "category": "correct",
                "reasoning": f"F1 {f1:.2f} >= {F1_CORRECT:.2f}",
            }

        # Context grounding path for RAG answers
        if context:
            context_norm = normalize_answer(context)
            pred_tokens = pred_norm.split()
            context_tokens = set(context_norm.split())
            if pred_tokens:
                support_hits = sum(t in context_tokens for t in pred_tokens)
                support_ratio = support_hits / len(pred_tokens)
            else:
                support_ratio = 0.0

            # Overall overlap proxy
            overlap = support_ratio

            # If partial F1 and sufficiently grounded -> mark as correct (supported paraphrase)
            if f1 >= F1_PARTIAL and support_ratio >= CONTEXT_SUPPORT_RATIO:
                return {
                    "is_hallucination": False,
                    "category": "correct",
                    "reasoning": f"Partial F1 {f1:.2f} with grounded tokens ratio {support_ratio:.2f}",
                }

            # Unsupported if very low overlap
            if overlap < MIN_CONTEXT_OVERLAP:
                return {
                    "is_hallucination": True,
                    "category": "unsupported",
                    "reasoning": f"Grounding ratio {overlap:.2f} < {MIN_CONTEXT_OVERLAP:.2f}",
                }

        # Default factual error
        return {
            "is_hallucination": True,
            "category": "factual_error",
            "reasoning": f"F1 {f1:.2f} below partial threshold {F1_PARTIAL:.2f} or insufficient grounding",
        }


class Evaluator:
    """Complete evaluation suite for RAG analysis."""

    def __init__(self):
        """Initialize evaluator."""
        self.hallucination_detector = HallucinationDetector()
        self.results = []

    def evaluate_single(self, example: Dict, prediction: Dict, condition: str) -> Dict:
        """
        Evaluate a single example.

        Args:
            example: Example from dataset
            prediction: Model prediction dictionary
            condition: 'no_rag' or 'rag_k{k}'

        Returns:
            Evaluation metrics
        """
        # Accuracy metrics
        em = exact_match(prediction["answer"], example["answer"])
        f1 = token_f1(prediction["answer"], example["answer"])

        # Hallucination detection
        context = None
        if prediction.get("retrieved_passages"):
            context = "\n".join([p["text"] for p in prediction["retrieved_passages"]])

        hallucination = self.hallucination_detector.classify_hallucination(
            prediction["answer"], example["answer"], context
        )

        # Retrieval metrics (if applicable)
        retrieval_metrics = {}
        if prediction.get("retrieved_passages"):
            retrieval_metrics["recall@k"] = compute_retrieval_recall(
                prediction["retrieved_passages"],
                example["supporting_facts"],
                k=prediction["k"],
            )
            retrieval_metrics["mrr"] = compute_mrr(
                prediction["retrieved_passages"], example["supporting_facts"]
            )

        result = {
            "question_id": example["id"],
            "condition": condition,
            "exact_match": em,
            "token_f1": f1,
            "is_hallucination": hallucination["is_hallucination"],
            "hallucination_category": hallucination["category"],
            **retrieval_metrics,
        }

        self.results.append(result)
        return result

    def aggregate_results(self, condition: str = None) -> Dict:
        """
        Aggregate results across all examples.

        Args:
            condition: Optional filter by condition

        Returns:
            Aggregated metrics
        """
        df = pd.DataFrame(self.results)

        if condition:
            df = df[df["condition"] == condition]

        if len(df) == 0:
            return {}

        metrics = {
            "n_examples": len(df),
            "exact_match": df["exact_match"].mean(),
            "token_f1": df["token_f1"].mean(),
            "hallucination_rate": df["is_hallucination"].mean(),
        }

        # Add retrieval metrics if available
        if "recall@k" in df.columns:
            metrics["avg_recall@k"] = df["recall@k"].mean()
        if "mrr" in df.columns:
            metrics["avg_mrr"] = df["mrr"].mean()

        return metrics

    def compare_conditions(self, condition_a: str, condition_b: str) -> Dict:
        """
        Compare two conditions using statistical tests.

        Args:
            condition_a: First condition
            condition_b: Second condition

        Returns:
            Statistical comparison results
        """
        df = pd.DataFrame(self.results)

        results_a = df[df["condition"] == condition_a]
        results_b = df[df["condition"] == condition_b]

        # McNemar's test for paired binary outcomes
        # Create contingency table for exact match
        both_correct = (
            (results_a["exact_match"] == 1) & (results_b["exact_match"] == 1)
        ).sum()
        a_only = (
            (results_a["exact_match"] == 1) & (results_b["exact_match"] == 0)
        ).sum()
        b_only = (
            (results_a["exact_match"] == 0) & (results_b["exact_match"] == 1)
        ).sum()
        both_wrong = (
            (results_a["exact_match"] == 0) & (results_b["exact_match"] == 0)
        ).sum()

        # McNemar test
        n = a_only + b_only
        if n > 0:
            statistic = (abs(a_only - b_only) - 1) ** 2 / n
            p_value = 1 - stats.chi2.cdf(statistic, 1)
        else:
            p_value = 1.0

        return {
            "condition_a": condition_a,
            "condition_b": condition_b,
            "contingency_table": {
                "both_correct": both_correct,
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "both_wrong": both_wrong,
            },
            "mcnemar_p_value": p_value,
            "em_diff": results_b["exact_match"].mean()
            - results_a["exact_match"].mean(),
            "f1_diff": results_b["token_f1"].mean() - results_a["token_f1"].mean(),
            "hallucination_diff": results_a["is_hallucination"].mean()
            - results_b["is_hallucination"].mean(),
        }

    def get_results_df(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        return pd.DataFrame(self.results)

    def save_results(self, filepath: str) -> None:
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Saved results to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("This module should be imported, not run directly.")

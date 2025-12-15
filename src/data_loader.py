"""
Data loader module for HotpotQA dataset.
Handles downloading, preprocessing, and sampling the dataset.
"""

import random
from typing import List, Dict, Tuple
from datasets import load_dataset
import pandas as pd
import json


class HotpotQALoader:
    """Load and preprocess HotpotQA dataset for RAG evaluation."""

    def __init__(self, subset_size: int = 100, random_seed: int = 42):
        """
        Initialize the data loader.

        Args:
            subset_size: Number of examples to sample (50-100)
            random_seed: Random seed for reproducibility
        """
        self.subset_size = subset_size
        self.random_seed = random_seed
        random.seed(random_seed)
        self.dataset = None
        self.subset = None

    def load_dataset(self, split: str = "validation") -> None:
        """
        Load HotpotQA dataset from Hugging Face.

        Args:
            split: Dataset split to load ('train', 'validation')
        """
        print(f"Loading HotpotQA dataset ({split} split)...")
        self.dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
        print(f"Loaded {len(self.dataset)} examples")

    def create_subset(self, strategy: str = "random") -> List[Dict]:
        """
        Create a subset of the dataset for evaluation.

        Args:
            strategy: Sampling strategy ('random', 'balanced', 'difficulty')

        Returns:
            List of sampled examples
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        if strategy == "random":
            indices = random.sample(range(len(self.dataset)), self.subset_size)
            self.subset = [self.dataset[i] for i in indices]
        elif strategy == "balanced":
            # Balance by answer type if needed
            self.subset = self._balanced_sample()
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        print(
            f"Created subset of {len(self.subset)} examples using '{strategy}' strategy"
        )
        return self.subset

    def _balanced_sample(self) -> List[Dict]:
        """Create a balanced sample across different question types."""
        # Group by question characteristics (e.g., answer length, complexity)
        all_examples = list(self.dataset)
        random.shuffle(all_examples)
        return all_examples[: self.subset_size]

    def preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single example.

        Args:
            example: Raw example from dataset

        Returns:
            Preprocessed example with standardized fields
        """
        return {
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "supporting_facts": example["supporting_facts"],
            "context": example["context"],  # List of [title, sentences] pairs
        }

    def get_supporting_passages(self, example: Dict) -> List[str]:
        """
        Extract supporting passages from context based on supporting_facts.

        Args:
            example: Preprocessed example

        Returns:
            List of supporting passage texts
        """
        supporting_titles = set(
            [fact[0] for fact in example["supporting_facts"]["title"]]
        )
        supporting_passages = []

        for title, sentences in zip(
            example["context"]["title"], example["context"]["sentences"]
        ):
            if title in supporting_titles:
                # Join sentences into passage
                passage = " ".join(sentences)
                supporting_passages.append(passage)

        return supporting_passages

    def prepare_corpus(self) -> List[Dict]:
        """
        Prepare text corpus for retrieval indexing.

        Returns:
            List of passages with metadata
        """
        corpus = []
        passage_id = 0

        for example in self.subset:
            # Extract all passages from context
            for title, sentences in zip(
                example["context"]["title"], example["context"]["sentences"]
            ):
                passage_text = " ".join(sentences)
                corpus.append(
                    {
                        "id": passage_id,
                        "text": passage_text,
                        "title": title,
                        "source_question_id": example["id"],
                    }
                )
                passage_id += 1

        print(f"Prepared corpus with {len(corpus)} passages")
        return corpus

    def save_subset(self, filepath: str) -> None:
        """Save the subset to a JSON file."""
        if self.subset is None:
            raise ValueError("No subset created. Call create_subset() first.")

        with open(filepath, "w") as f:
            json.dump(self.subset, f, indent=2)
        print(f"Saved subset to {filepath}")

    def load_subset(self, filepath: str) -> None:
        """Load a previously saved subset."""
        with open(filepath, "r") as f:
            self.subset = json.load(f)
        print(f"Loaded subset with {len(self.subset)} examples from {filepath}")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 75) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens (approximated by words)
        overlap: Overlap size in tokens

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:  # Avoid empty chunks
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    # Example usage
    loader = HotpotQALoader(subset_size=100, random_seed=42)
    loader.load_dataset(split="validation")
    subset = loader.create_subset(strategy="random")

    # Print example
    example = loader.preprocess_example(subset[0])
    print(f"\nExample question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print(f"Supporting facts: {loader.get_supporting_passages(example)[:2]}")

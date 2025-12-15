"""
RAG Pipeline implementation.
Handles retrieval, embedding, and generation with/without RAG.
"""

import torch
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")


class VectorStore:
    """FAISS-based vector store for retrieval with optional BM25 hybrid."""

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_config: Optional[Dict] = None,
        hybrid_config: Optional[Dict] = None,
    ):
        """
        Initialize vector store.

        Args:
            encoder_model: SentenceTransformer model name
            reranker_config: Optional cross-encoder reranking config
            hybrid_config: Optional BM25 hybrid retrieval config
        """
        self.encoder = SentenceTransformer(encoder_model)
        self.index = None
        self.passages = []
        self.dimension = None
        # Reranker
        self.reranker_cfg = reranker_config or {}
        self._reranker = None
        # BM25 hybrid
        self.hybrid_cfg = hybrid_config or {}
        self._bm25 = None
        self._tokenized_corpus = None

    def build_index(self, passages: List[Dict]) -> None:
        """
        Build FAISS index from passages, optionally with BM25.

        Args:
            passages: List of passage dictionaries with 'text' field
        """
        self.passages = passages
        texts = [p["text"] for p in passages]

        print("Encoding passages...")
        embeddings = self.encoder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(
            self.dimension
        )  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        print(f"Built FAISS index with {len(passages)} passages (dim={self.dimension})")

        # Build BM25 index if hybrid enabled
        if self.hybrid_cfg.get("enabled", False) and BM25Okapi is not None:
            print("Building BM25 index for hybrid retrieval...")
            self._tokenized_corpus = [text.lower().split() for text in texts]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            print("BM25 index built successfully")

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve top-k passages for a query with optional BM25 hybrid.

        Args:
            query: Query text
            k: Number of passages to retrieve

        Returns:
            Tuple of (retrieved passages, similarity scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Check if hybrid retrieval enabled
        use_hybrid = (
            self.hybrid_cfg.get("enabled", False)
            and self._bm25 is not None
            and BM25Okapi is not None
        )

        # Determine initial pool size
        use_rerank = (
            bool(self.reranker_cfg.get("enabled", False)) and CrossEncoder is not None
        )
        pool_n = int(self.reranker_cfg.get("top_n", max(k, 20))) if use_rerank else k

        if use_hybrid:
            # Hybrid retrieval: combine BM25 + dense
            bm25_top_n = int(self.hybrid_cfg.get("bm25_top_n", 50))
            bm25_weight = float(self.hybrid_cfg.get("bm25_weight", 0.3))
            dense_weight = float(self.hybrid_cfg.get("dense_weight", 0.7))

            # BM25 scores
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)

            # Dense scores
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            dense_scores_raw, indices = self.index.search(
                query_embedding, len(self.passages)
            )
            dense_scores = dense_scores_raw[0]

            # Normalize scores to [0, 1] for combining
            bm25_norm = (
                (bm25_scores - bm25_scores.min())
                / (bm25_scores.max() - bm25_scores.min() + 1e-9)
                if bm25_scores.max() > bm25_scores.min()
                else bm25_scores
            )
            dense_norm = (
                (dense_scores - dense_scores.min())
                / (dense_scores.max() - dense_scores.min() + 1e-9)
                if dense_scores.max() > dense_scores.min()
                else dense_scores
            )

            # Combine scores
            combined_scores = bm25_weight * bm25_norm + dense_weight * dense_norm

            # Get top pool_n candidates
            top_indices = np.argsort(combined_scores)[::-1][:pool_n]
            candidates = [self.passages[idx] for idx in top_indices]
            scores_list = combined_scores[top_indices].tolist()

        else:
            # Dense-only retrieval (original FAISS path)
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, pool_n)
            candidates = [self.passages[idx] for idx in indices[0]]
            scores_list = scores[0].tolist()

        # Reranking if enabled
        if use_rerank and len(candidates) > 0:
            try:
                # Lazy load reranker
                if self._reranker is None:
                    model_name = self.reranker_cfg.get(
                        "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )
                    self._reranker = CrossEncoder(model_name)

                # Prepare pairs and score
                pairs = [(query, c.get("text", "")) for c in candidates]
                rerank_scores = self._reranker.predict(pairs).tolist()

                # Sort by reranker score desc and take top-k
                order = np.argsort(rerank_scores)[::-1][:k]
                reranked = [candidates[i] for i in order]
                reranked_scores = [rerank_scores[i] for i in order]
                return reranked, reranked_scores
            except Exception as e:
                print(f"Reranker unavailable or failed ({e}); using initial results.")
                # Fall back to initial top-k
                return candidates[:k], scores_list[:k]
        else:
            # No reranker; return initial top-k
            return candidates[:k], scores_list[:k]

    def save_index(self, filepath: str) -> None:
        """Save FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, filepath)
            print(f"Saved index to {filepath}")

    def load_index(self, filepath: str) -> None:
        """Load FAISS index from disk."""
        self.index = faiss.read_index(filepath)
        print(f"Loaded index from {filepath}")


class LLMGenerator:
    """LLM generator for answering questions."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        """
        Initialize LLM generator.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.model_name = model_name
        self.device = device

        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Resolve device
        use_cuda = torch.cuda.is_available()
        use_mps = (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        resolved_device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
        print(f"Resolved device: {resolved_device}")

        # Load model with optional quantization (fallback gracefully if not supported)
        self.model = None
        if load_in_8bit:
            try:
                # bitsandbytes may not be available on macOS/MPS; this will raise
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map=(device if device != "auto" else "auto"),
                )
                print("Loaded model in 8-bit quantization")
            except Exception as e:
                print(
                    f"8-bit load not available on this platform or missing bitsandbytes. "
                    f"Falling back to half/float precision. Error: {e}"
                )

        if self.model is None:
            dtype = torch.float16 if (use_cuda or use_mps) else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=(device if device != "auto" else "auto"),
            )
            print(
                f"Loaded model dtype: {dtype}, device_map: {device if device != 'auto' else 'auto'}"
            )

        self.model.eval()
        print("Model loaded successfully")

    def generate_answer(
        self,
        question: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 100,
        seed: int = 42,
    ) -> str:
        """
        Generate answer to a question.

        Args:
            question: Question text
            context: Optional context for RAG
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility

        Returns:
            Generated answer
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Build prompt
        if context:
            prompt = self._build_rag_prompt(question, context)
        else:
            prompt = self._build_no_rag_prompt(question)

        # Tokenize
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )

        # Move inputs to an appropriate device if needed
        if torch.cuda.is_available():
            target = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            target = "mps"
        else:
            target = "cpu"

        if target != "cpu":
            print(f"Sending inputs to device: {target}")
            inputs = {k: v.to(target) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature and temperature > 0 else False,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (remove prompt)
        answer = full_output[len(prompt) :].strip()

        return answer

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build prompt with retrieved context and citation requirement."""
        prompt = (
            "Answer the question using only the numbered context.\n"
            "Provide a single short answer (a few words).\n"
            "If the context is insufficient, answer 'Unknown'.\n"
            "Cite supporting evidence indices like [1], [2] in the answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return prompt

    def _build_no_rag_prompt(self, question: str) -> str:
        """Build prompt without context."""
        prompt = f"""Answer the following question concisely and factually.

Question: {question}

Answer:"""
        return prompt


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        vector_store: VectorStore,
        generator: LLMGenerator,
        enforce_citations: bool = False,
        context_max_sentences: int = 6,
        sentences_per_passage: int = 2,
    ):
        """
        Initialize RAG pipeline.

        Args:
            vector_store: VectorStore instance
            generator: LLMGenerator instance
        """
        self.vector_store = vector_store
        self.generator = generator
        self.enforce_citations = enforce_citations
        self.context_max_sentences = context_max_sentences
        self.sentences_per_passage = sentences_per_passage

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        import re

        # Simple sentence splitter
        parts = re.split(r"(?<=[.!?])\s+", text)
        # Clean and filter
        return [s.strip() for s in parts if s and len(s.strip().split()) > 3]

    @staticmethod
    def _score_sentence(question: str, sentence: str) -> float:
        # Token overlap score
        q = set(question.lower().split())
        s = set(sentence.lower().split())
        if not s:
            return 0.0
        return len(q & s) / len(s)

    def _select_relevant_sentences(
        self, question: str, text: str, per_passage: int
    ) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [text]
        scored = [(self._score_sentence(question, s), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:per_passage]]

    def answer_with_rag(self, question: str, k: int = 3, **generation_kwargs) -> Dict:
        """
        Answer question with RAG.

        Args:
            question: Question text
            k: Number of passages to retrieve
            **generation_kwargs: Additional arguments for generation

        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve passages
        retrieved_passages, scores = self.vector_store.retrieve(question, k=k)

        # Build numbered, trimmed context with titles and top sentences
        numbered = []
        total_sentences = 0
        for idx, p in enumerate(retrieved_passages, start=1):
            if total_sentences >= self.context_max_sentences:
                break
            title = p.get("title", "Untitled")
            text = p.get("text", "")
            selected = self._select_relevant_sentences(
                question, text, self.sentences_per_passage
            )
            # Truncate if exceeding global cap
            remaining = self.context_max_sentences - total_sentences
            selected = selected[: max(0, remaining)]
            total_sentences += len(selected)
            snippet = " ".join(selected) if selected else text
            numbered.append(f"[{idx}] {title}: {snippet}")
        context = "\n\n".join(numbered)

        # Generate answer
        answer = self.generator.generate_answer(
            question, context=context, **generation_kwargs
        )

        # Lightweight self-check: enforce citations if configured
        if self.enforce_citations:
            import re

            if not re.search(r"\[\d+\]", answer):
                answer = "Unknown"

        return {
            "answer": answer,
            "retrieved_passages": retrieved_passages,
            "retrieval_scores": scores,
            "k": k,
            "enforce_citations": self.enforce_citations,
        }

    def answer_without_rag(self, question: str, **generation_kwargs) -> Dict:
        """
        Answer question without RAG (baseline).

        Args:
            question: Question text
            **generation_kwargs: Additional arguments for generation

        Returns:
            Dictionary with answer
        """
        answer = self.generator.generate_answer(
            question, context=None, **generation_kwargs
        )

        return {
            "answer": answer,
            "retrieved_passages": None,
            "retrieval_scores": None,
            "k": 0,
        }


if __name__ == "__main__":
    # Example usage
    print("This module should be imported, not run directly.")
    print("See main.py for usage examples.")

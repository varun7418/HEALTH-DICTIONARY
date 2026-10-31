"""
HealthAura - Medical Retriever
Loads a local FAISS index + chunk metadata and retrieves the top-k most
relevant chunks for a query using cosine similarity.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Project root = two levels above src/rag/retriever.py
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = BASE_DIR / "data" / "index" / "index.faiss"
DEFAULT_METADATA_PATH = BASE_DIR / "data" / "index" / "metadata.json"

# MUST be the same model that was used in build_index.py
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    metadata: dict
    similarity_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class MedicalRetriever:
    def __init__(
        self,
        index_path: str = str(DEFAULT_INDEX_PATH),
        metadata_path: str = str(DEFAULT_METADATA_PATH),
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        """
        Initialize the Medical Retriever by loading the local FAISS index,
        the chunk metadata store, and the SentenceTransformer model.
        """
        self.index_path = str(index_path)
        self.metadata_path = str(metadata_path)
        self.model_name = model_name

        # Verify database files exist
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Index files not found. Please ensure you have run 'build_index.py' "
                f"to create {self.index_path} (index) and {self.metadata_path} (metadata) first."
            )

        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)

        logger.info("Loading chunks metadata store...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if self.index.ntotal != len(self.chunks):
            logger.warning(
                "Index has %d vectors but metadata has %d chunks. "
                "They should match - rebuild the index.",
                self.index.ntotal,
                len(self.chunks),
            )

        logger.info("Initializing local SentenceTransformer model...")
        self.model = SentenceTransformer(self.model_name)

        # Sanity check: model embedding size must match the index
        get_dim = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        model_dim = get_dim()
        if model_dim != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: model '{self.model_name}' outputs "
                f"{model_dim}-d vectors but the index expects {self.index.d}-d. "
                f"Use the same model that built the index."
            )

        self._embed_cached = lru_cache(maxsize=256)(self._embed)
        logger.info("Retriever loaded successfully and ready.")

    # ------------------------------------------------------------------ #
    def _embed(self, query: str) -> np.ndarray:
        """Embed + L2-normalize a query (so inner product == cosine similarity)."""
        vec = self.model.encode([query])
        vec = np.array(vec).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embed the input query, search the FAISS index, and return the top k
        matching chunks with their similarity scores.

        Args:
            query: user search text.
            k: number of results to return.
            min_score: if set, drop results with similarity below this value.
            source_filter: if set, keep only chunks whose 'source' contains this text.
        """
        if not query or not query.strip():
            return []
        if self.index.ntotal == 0:
            return []

        # Over-fetch when filtering so we still have up to k results afterwards
        fetch_k = self.index.ntotal if source_filter else k
        fetch_k = max(1, min(fetch_k, self.index.ntotal))

        # 1. Embed + normalize (copy so the cached array is never mutated by FAISS)
        query_vector = self._embed_cached(query.strip()).copy()

        # 2. Search. With IndexFlatIP and normalized vectors, scores are cosine similarity.
        similarities, indices = self.index.search(query_vector, fetch_k)

        results: list[RetrievedChunk] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 if there are not enough items in the index
                continue
            if min_score is not None and sim < min_score:
                continue

            chunk = self.chunks[idx]
            if source_filter and source_filter.lower() not in str(chunk.get("source", "")).lower():
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.get("chunk_id", str(idx)),
                    source=chunk.get("source", "unknown"),
                    text=chunk.get("text", ""),
                    metadata=chunk.get("metadata", {}),
                    similarity_score=float(sim),
                )
            )
            if len(results) >= k:
                break

        return [r.to_dict() for r in results]


# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=== HealthAura Local Retriever Test Console ===")

    try:
        retriever = MedicalRetriever()

        # Simple interactive loop
        while True:
            query = input("\nEnter search query (or type 'exit' to quit): ").strip()
            if not query:
                continue
            if query.lower() == "exit":
                print("Exiting test console.")
                break

            print(f"Searching for: '{query}'...")
            matches = retriever.retrieve(query, k=3)

            if not matches:
                print("No matches found.")
                continue

            print(f"\nTop {len(matches)} Results:")
            for i, match in enumerate(matches, 1):
                print("-" * 50)
                print(f"[{i}] Source: {match['source']} (Score: {match['similarity_score']:.4f})")
                print(f"Snippet: {match['text'].strip()}")
                print(f"Metadata: {match['metadata']}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nExiting test console.")
    except Exception:
        logger.exception("Error starting retriever test console")
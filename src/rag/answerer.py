"""
HealthAura - Extractive Answerer (no LLM required)

Pipeline:
  1. Retrieve top-k chunks with MedicalRetriever.
  2. Split those chunks into sentences.
  3. Rank sentences by similarity to the query (same embedding model).
  4. Return the best sentences, each with its source.

Every word in the answer comes directly from your documents, so nothing
can be invented. If nothing is relevant enough, it says so.
"""

import logging
import re
from dataclasses import dataclass

import faiss
import numpy as np

try:
    from retriever import MedicalRetriever
except ImportError:  # when imported as a package: from src.rag.answerer import ...
    from .retriever import MedicalRetriever

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This information is extracted from reference documents and is for "
    "educational purposes only. It is not medical advice. Please consult "
    "a qualified healthcare professional."
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")


@dataclass
class AnswerSentence:
    text: str
    source: str
    chunk_id: str
    score: float


@dataclass
class Answer:
    query: str
    sentences: list  # list[AnswerSentence]
    confident: bool
    disclaimer: str = DISCLAIMER

    def format(self) -> str:
        if not self.sentences:
            return (
                "I couldn't find relevant information in the documents for that question.\n\n"
                f"{self.disclaimer}"
            )
        lines = []
        if not self.confident:
            lines.append("Note: low confidence - these are the closest matches found.\n")
        for i, s in enumerate(self.sentences, 1):
            lines.append(f"{i}. {s.text}\n   [source: {s.source}, score: {s.score:.2f}]")
        lines.append(f"\n{self.disclaimer}")
        return "\n".join(lines)


class ExtractiveAnswerer:
    def __init__(
        self,
        retriever: MedicalRetriever = None,
        retrieve_k: int = 5,
        top_sentences: int = 4,
        min_chunk_score: float = 0.25,
        confident_score: float = 0.45,
        min_sentence_chars: int = 25,
    ):
        self.retriever = retriever or MedicalRetriever()
        self.retrieve_k = retrieve_k
        self.top_sentences = top_sentences
        self.min_chunk_score = min_chunk_score
        self.confident_score = confident_score
        self.min_sentence_chars = min_sentence_chars

    @staticmethod
    def _split_sentences(text: str) -> list:
        parts = _SENTENCE_SPLIT.split(text.strip())
        return [p.strip().replace("\n", " ") for p in parts if p and p.strip()]

    def answer(self, query: str) -> Answer:
        chunks = self.retriever.retrieve(
            query, k=self.retrieve_k, min_score=self.min_chunk_score
        )
        if not chunks:
            return Answer(query=query, sentences=[], confident=False)

        # Collect candidate sentences from retrieved chunks
        candidates = []  # (text, source, chunk_id)
        seen = set()
        for c in chunks:
            for sent in self._split_sentences(c["text"]):
                key = sent.lower()
                if len(sent) < self.min_sentence_chars or key in seen:
                    continue
                seen.add(key)
                candidates.append((sent, c["source"], c["chunk_id"]))

        if not candidates:
            return Answer(query=query, sentences=[], confident=False)

        # Score sentences against the query using the retriever's own model
        model = self.retriever.model
        sent_vecs = np.array(
            model.encode([c[0] for c in candidates])
        ).astype("float32")
        faiss.normalize_L2(sent_vecs)
        query_vec = self.retriever._embed_cached(query.strip())  # already normalized
        scores = sent_vecs @ query_vec[0]

        order = np.argsort(-scores)[: self.top_sentences]
        best = [
            AnswerSentence(
                text=candidates[i][0],
                source=candidates[i][1],
                chunk_id=candidates[i][2],
                score=float(scores[i]),
            )
            for i in order
            if scores[i] >= self.min_chunk_score
        ]

        confident = bool(best) and best[0].score >= self.confident_score
        return Answer(query=query, sentences=best, confident=confident)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("=== HealthAura Extractive Answer Console ===")
    try:
        answerer = ExtractiveAnswerer()
        while True:
            q = input("\nAsk a question (or type 'exit' to quit): ").strip()
            if not q:
                continue
            if q.lower() == "exit":
                print("Goodbye.")
                break
            print("\n" + answerer.answer(q).format())
    except KeyboardInterrupt:
        print("\nGoodbye.")
    except Exception:
        logger.exception("Error starting answer console")

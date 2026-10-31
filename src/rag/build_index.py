"""
HealthAura - Index Builder

Reads every PDF in data/raw/, splits the text into overlapping chunks,
embeds them, and saves:
    data/index/index.faiss    (vector index)
    data/index/metadata.json  (chunk text + source info)

Run from the project root:
    python src/rag/build_index.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
INDEX_DIR = BASE_DIR / "data" / "index"
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"

# MUST match DEFAULT_MODEL_NAME in retriever.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 200      # words per chunk
OVERLAP_WORDS = 40     # words shared between neighbouring chunks
MIN_CHUNK_WORDS = 30   # skip tiny fragments


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)   # re-join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_words(text: str, size: int, overlap: int) -> list:
    words = text.split()
    if not words:
        return []
    chunks, step = [], max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= MIN_CHUNK_WORDS or (start == 0 and piece):
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def load_pdf_chunks(pdf_path: Path) -> list:
    chunks = []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        logger.warning("Could not open %s: %s", pdf_path.name, e)
        return chunks

    for page_num, page in enumerate(reader.pages, 1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception as e:
            logger.warning("Failed on %s page %d: %s", pdf_path.name, page_num, e)
            continue
        for piece in chunk_words(text, CHUNK_WORDS, OVERLAP_WORDS):
            chunks.append({
                "source": pdf_path.name,
                "text": piece,
                "metadata": {"page": page_num},
            })
    return chunks


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s\nPut your PDFs there and run again.", RAW_DIR)
        sys.exit(1)

    logger.info("Found %d PDF(s). Reading...", len(pdfs))
    all_chunks = []
    for pdf in pdfs:
        pdf_chunks = load_pdf_chunks(pdf)
        logger.info("  %s -> %d chunks", pdf.name, len(pdf_chunks))
        all_chunks.extend(pdf_chunks)

    if not all_chunks:
        logger.error(
            "No text could be extracted. If your PDFs are scanned images, "
            "they need OCR first."
        )
        sys.exit(1)

    for i, c in enumerate(all_chunks):
        c["chunk_id"] = f"chunk_{i:06d}"

    logger.info("Loading embedding model (first run downloads ~90 MB)...")
    model = SentenceTransformer(MODEL_NAME)

    logger.info("Embedding %d chunks...", len(all_chunks))
    vectors = model.encode(
        [c["text"] for c in all_chunks],
        batch_size=64,
        show_progress_bar=True,
    )
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info("\nDone.")
    logger.info("  Index:    %s (%d vectors)", INDEX_PATH, index.ntotal)
    logger.info("  Metadata: %s", METADATA_PATH)


if __name__ == "__main__":
    main()

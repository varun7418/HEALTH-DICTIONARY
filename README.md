# Health Dictionary - Local Medical RAG Retriever

A local question-answering system for medical PDFs. It searches by **meaning** (not just keywords) and returns the most relevant sentences from your documents with their sources. Answers are **extractive**: every sentence is quoted from the documents, so the system cannot invent medical facts. Everything runs locally; no API key is needed.

## Features
- Semantic search with sentence embeddings and a FAISS vector index
- Extractive answers with source file and relevance score
- NLP helpers: abbreviation expansion (BP -> blood pressure), keyword extraction, keyword highlighting
- Confidence warning and "not found" response for unrelated questions
- Streamlit web interface with live tuning sliders
- Medical disclaimer on every answer

## Tech stack
Python, NLP (SentenceTransformers `all-MiniLM-L6-v2`), RAG retrieval (FAISS), pypdf, NumPy, Streamlit

## How it works
1. **Index (once):** PDFs in `data/raw` are cleaned, split into 200-word chunks (40-word overlap), embedded, and stored in a FAISS index.
2. **Ask:** the question is expanded, embedded, and matched against the index. Retrieved chunks are split into sentences, and the sentences closest to the question are returned.

## Project structure
```
health-dictionary/
├── app.py                 # Streamlit interface
├── requirements.txt
├── data/
│   └── raw/               # put your PDFs here
└── src/rag/
    ├── build_index.py     # PDFs -> FAISS index
    ├── retriever.py       # finds the best chunks
    ├── answerer.py        # picks the best sentences
    └── nlp_utils.py       # NLP helpers
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install -r requirements.txt
```

## Run
```bash
python src/rag/build_index.py   # build the index (repeat when PDFs change)
python -m streamlit run app.py  # start the web app
```
Terminal version: `python src/rag/answerer.py`

## Example questions
- What are the symptoms of diabetes?
- What is a normal blood pressure?
- What is HTN?
- When should I see a doctor for the flu?

## Future work
Reranker, hybrid keyword + vector search, evaluation set (recall@k), medical entity recognition, optional LLM generation.

## Disclaimer
For educational purposes only. Not medical advice.

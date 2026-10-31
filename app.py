"""
HealthAura - Streamlit UI

Run from the project root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "rag"))

from answerer import ExtractiveAnswerer, DISCLAIMER  # noqa: E402
from nlp_utils import (  # noqa: E402
    expand_query,
    extract_keywords,
    highlight,
    keyword_overlap,
)

st.set_page_config(page_title="HealthAura", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner="Loading index and model...")
def load_answerer():
    return ExtractiveAnswerer()


st.title("🩺 HealthAura")
st.caption("Ask questions about your documents. Answers are extracted word-for-word "
           "from your PDFs, with sources.")

# ---- Load the pipeline ----
try:
    answerer = load_answerer()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not start the retriever: {e}")
    st.stop()

# ---- Sidebar settings ----
with st.sidebar:
    st.header("Settings")
    use_expansion = st.toggle("Expand medical abbreviations", value=True,
                              help="BP -> blood pressure, MRI -> magnetic resonance imaging, ...")
    answerer.retrieve_k = st.slider("Chunks to retrieve", 1, 10, 5)
    answerer.top_sentences = st.slider("Sentences in answer", 1, 10, 4)
    answerer.min_chunk_score = st.slider("Minimum relevance", 0.0, 0.8, 0.25, 0.05,
                                         help="Lower = more results but more noise.")
    answerer.confident_score = st.slider("Confidence threshold", 0.0, 0.9, 0.45, 0.05)
    show_chunks = st.toggle("Show full retrieved chunks", value=False)
    st.divider()
    st.caption(f"Index size: {answerer.retriever.index.ntotal} chunks")

# ---- Query box ----
query = st.text_input("Your question", placeholder="e.g. What are the symptoms of diabetes?")

if query and query.strip():
    expanded, expansions = expand_query(query) if use_expansion else (query, [])
    keywords = extract_keywords(expanded)

    if expansions:
        st.caption("Expanded: " + ", ".join(f"**{a.upper()}** → {f}" for a, f in expansions))
    if keywords:
        st.caption("Keywords: " + " · ".join(f"`{k}`" for k in keywords))

    with st.spinner("Searching..."):
        answer = answerer.answer(expanded)

    if not answer.sentences:
        st.info("I couldn't find relevant information in the documents for that question. "
                "Try rewording it, or lower the minimum relevance in the sidebar.")
    else:
        if answer.confident:
            st.success("Confident match")
        else:
            st.warning("Low confidence - these are the closest matches found.")

        for i, s in enumerate(answer.sentences, 1):
            matched, total = keyword_overlap(s.text, keywords)
            with st.container(border=True):
                st.markdown(highlight(s.text, keywords), unsafe_allow_html=True)
                kw = f" · keywords {matched}/{total}" if total else ""
                st.caption(f"📄 {s.source} · relevance {s.score:.2f}{kw}")

        if show_chunks:
            with st.expander("Retrieved chunks"):
                chunks = answerer.retriever.retrieve(
                    expanded,
                    k=answerer.retrieve_k,
                    min_score=answerer.min_chunk_score,
                )
                for c in chunks:
                    page = c.get("metadata", {}).get("page", "?")
                    st.markdown(f"**{c['source']}** (page {page}, score {c['similarity_score']:.2f})")
                    st.markdown(highlight(c["text"], keywords), unsafe_allow_html=True)
                    st.divider()

    st.caption(DISCLAIMER)

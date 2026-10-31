"""
HealthAura - Lightweight NLP helpers (pure Python, no extra downloads)

- expand_query:      expands medical abbreviations (BP -> blood pressure)
- extract_keywords:  removes stopwords and returns the meaningful words
- keyword_overlap:   how many query keywords appear in a piece of text
- highlight:         wraps matching words in <mark> tags (HTML-safe)
"""

import html
import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "at", "by", "for", "with", "about", "as", "into", "from", "up", "out",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "doing", "have", "has", "had", "having", "can", "could", "should", "would",
    "may", "might", "will", "shall", "must", "what", "which", "who", "whom",
    "whose", "when", "where", "why", "how", "this", "that", "these", "those",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "them", "their", "there", "here", "tell", "give", "explain", "describe",
    "mean", "means", "meaning", "define", "definition", "please", "some", "any",
    "not", "no", "than", "too", "very", "just", "also", "more", "most", "other",
}

# Appended (not replaced) to the query so both forms are searched.
ABBREVIATIONS = {
    "bp": "blood pressure",
    "hr": "heart rate",
    "bmi": "body mass index",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "copd": "chronic obstructive pulmonary disease",
    "uti": "urinary tract infection",
    "gerd": "gastroesophageal reflux disease",
    "hiv": "human immunodeficiency virus",
    "tb": "tuberculosis",
    "ibs": "irritable bowel syndrome",
    "ckd": "chronic kidney disease",
    "mi": "myocardial infarction heart attack",
    "cvd": "cardiovascular disease",
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "htn": "hypertension high blood pressure",
    "hba1c": "glycated hemoglobin",
    "ldl": "low density lipoprotein cholesterol",
    "hdl": "high density lipoprotein cholesterol",
    "ibd": "inflammatory bowel disease",
    "adhd": "attention deficit hyperactivity disorder",
    "ptsd": "post traumatic stress disorder",
    "nsaid": "nonsteroidal anti-inflammatory drug",
    "nsaids": "nonsteroidal anti-inflammatory drugs",
    "otc": "over the counter",
}

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_SUFFIXES = ("ies", "ing", "ed", "es", "ly", "s")


def stem(word: str) -> str:
    """Very simple suffix stripper used as a prefix for matching
    (diabetes/diabetic -> 'diabet', symptoms -> 'symptom')."""
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def expand_query(query: str):
    """Return (expanded_query, [(abbreviation, expansion), ...])."""
    found = []
    extra = []
    for tok in _TOKEN.findall(query):
        full = ABBREVIATIONS.get(tok.lower())
        if full and (tok.lower(), full) not in found:
            found.append((tok.lower(), full))
            extra.append(full)
    expanded = query.strip()
    if extra:
        expanded = f"{expanded} ({'; '.join(extra)})"
    return expanded, found


def extract_keywords(query: str) -> list:
    """Meaningful words of a query: no stopwords, no 1-2 letter words
    (unless the word contains a digit, e.g. 'hba1c')."""
    seen, out = set(), []
    for tok in _TOKEN.findall(query):
        t = tok.lower()
        if t in STOPWORDS or t in seen:
            continue
        if len(t) < 3 and not any(ch.isdigit() for ch in t):
            continue
        seen.add(t)
        out.append(t)
    return out


def keyword_overlap(text: str, keywords: list):
    """Return (matched, total) keyword counts for a piece of text."""
    if not keywords:
        return 0, 0
    stems = {stem(k) for k in keywords}
    matched = sum(
        1 for s in stems if re.search(r"\b" + re.escape(s), text, re.IGNORECASE)
    )
    return matched, len(stems)


def highlight(text: str, keywords: list) -> str:
    """HTML-escape text and wrap words that match any keyword in <mark>."""
    if not keywords:
        return html.escape(text)
    stems = sorted({stem(k) for k in keywords}, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in stems) + r")\w*", re.IGNORECASE
    )
    out, last = [], 0
    for m in pattern.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        out.append(
            '<mark style="background:#ffe066;color:#000;padding:0 2px;'
            f'border-radius:3px">{html.escape(m.group(0))}</mark>'
        )
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)

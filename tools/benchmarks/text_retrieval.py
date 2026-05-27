#!/usr/bin/env python3
"""
Text-similarity retrieval for problems keyed on natural language, not
examples. Used by GSM8K's VPoT path to inline top-K similar past
solve() functions as few-shot context.

The scheme is deliberately simple:
  - Tokenize each question into lowercase word bigrams (and unigrams
    for short questions).
  - Compute TF-IDF vectors across the cache.
  - Cosine similarity → top-K matches.

This is good enough for "find a similar word problem in my cache" —
our benchmarks suggest sim≥0.25 already correlates with useful
few-shot examples. For richer similarity (sentence-transformers,
etc.), the API is `text_lookup(query, k, min_sim) → list of hits`;
swap the backend when you're willing to ship a model dependency.

Cache format: reuses the main TSV schema. The stored `examples` field is
unused for text queries; we read the question text from the optional
`question` column. Backward-compatible — old 5/6-col rows simply aren't
queryable by text.
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_solution_cache import _cache_path, _load_all, record  # noqa: E402


_WORD_RE = re.compile(r"[a-z][a-z0-9]*", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens + word bigrams. Bigrams help catch
    distinctive phrases like 'round trip' vs 'one way'."""
    if not text:
        return []
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < 4:
        return words
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _tf(tokens: List[str]) -> Dict[str, float]:
    if not tokens:
        return {}
    c = Counter(tokens)
    total = float(len(tokens))
    return {t: n / total for t, n in c.items()}


def _idf(docs_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(docs_tokens)
    df: Counter = Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    return {t: math.log((N + 1) / (n + 1)) + 1.0 for t, n in df.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def record_with_question(fp: str, model: str, code: str,
                          question: str) -> None:
    """Persist a cache row with an attached question text for future
    text-similarity retrieval."""
    record(fp, model, code, question=question)


def _load_all_with_questions(path: Path) -> Dict[str, dict]:
    """Load the full cache preserving the optional question field."""
    if str(path) != str(_cache_path()):
        old = os.environ.get("NSYNTH_LLM_CACHE_PATH")
        os.environ["NSYNTH_LLM_CACHE_PATH"] = str(path)
        try:
            return _load_all()
        finally:
            if old is None:
                del os.environ["NSYNTH_LLM_CACHE_PATH"]
            else:
                os.environ["NSYNTH_LLM_CACHE_PATH"] = old
    if not path.exists():
        return {}
    return _load_all()


_EMBED_MODEL = None
_EMBED_CACHE: Dict[str, List[float]] = {}


def _get_embedder():
    """Lazy-load a sentence-transformer model for semantic retrieval.
    Gated by env var NSYNTH_TEXT_EMBEDDER — opt-in to avoid pulling
    ~80MB of weights on first call by default. Returns None if the
    model is unavailable; callers should fall back to TF-IDF."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    model_name = os.environ.get("NSYNTH_TEXT_EMBEDDER", "")
    if not model_name:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _EMBED_MODEL = SentenceTransformer(model_name)
        return _EMBED_MODEL
    except Exception:
        return None


def _embed(text: str, model) -> Optional[List[float]]:
    if text in _EMBED_CACHE:
        return _EMBED_CACHE[text]
    try:
        vec = model.encode(text, convert_to_numpy=False).tolist()
    except Exception:
        return None
    _EMBED_CACHE[text] = vec
    return vec


def _cos_list(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def text_lookup(
    query_text: str, k: int = 3, min_similarity: float = 0.25,
    include_self: bool = False,
) -> List[Dict]:
    """Find the K most-similar cached entries by TF-IDF cosine on the
    `question` field. Returns `{fingerprint, model, code, question,
    success_count, similarity}` dicts. Empty list if fewer than K
    cached rows have a question text, or if no entry exceeds
    `min_similarity`.

    The similarity scale is TF-IDF cosine: 0..1. Empirically, GSM8K
    questions covering the same problem pattern score ≥0.3; identical
    questions score ~1.0 (after self-exclusion)."""
    rows = _load_all_with_questions(_cache_path())
    # Only rows with a question are candidates.
    entries = [(fp, r) for fp, r in rows.items() if r.get("question")]
    if not entries:
        return []

    # Prefer a real sentence embedding when one is configured — TF-IDF
    # is noisy on small corpora because rare-word weighting dominates.
    embedder = _get_embedder()
    if embedder is not None:
        q_vec = _embed(query_text, embedder)
        if q_vec is None:
            embedder = None  # fall through to TF-IDF

    if embedder is not None:
        hits: List[Dict] = []
        for fp, r in entries:
            if r["question"] == query_text and not include_self:
                continue
            cand = _embed(r["question"], embedder)
            if cand is None:
                continue
            sim = _cos_list(q_vec, cand)
            if sim < min_similarity:
                continue
            hits.append({
                "fingerprint": fp,
                "model": r["model"],
                "code": r["code"],
                "question": r["question"],
                "success_count": r["success_count"],
                "similarity": round(sim, 4),
            })
        hits.sort(key=lambda h: -h["similarity"])
        return hits[:k]

    # TF-IDF fallback (no model configured).
    all_tokens = [_tokenize(r["question"]) for _, r in entries]
    all_tokens.append(_tokenize(query_text))
    idf = _idf(all_tokens)

    def _tfidf(tokens: List[str]) -> Dict[str, float]:
        tf_ = _tf(tokens)
        return {t: tf_[t] * idf.get(t, 0.0) for t in tf_}

    query_vec = _tfidf(_tokenize(query_text))
    if not query_vec:
        return []

    hits = []
    for (fp, r), toks in zip(entries, all_tokens[:-1]):
        if r["question"] == query_text and not include_self:
            continue
        cand_vec = _tfidf(toks)
        sim = _cosine(query_vec, cand_vec)
        if sim < min_similarity:
            continue
        hits.append({
            "fingerprint": fp,
            "model": r["model"],
            "code": r["code"],
            "question": r["question"],
            "success_count": r["success_count"],
            "similarity": round(sim, 4),
        })
    hits.sort(key=lambda h: -h["similarity"])
    return hits[:k]


def build_text_retrieval_prefix(
    query_text: str, k: int = 3, min_similarity: float = 0.25,
    max_code_chars: int = 400, max_question_chars: int = 200,
) -> str:
    """Few-shot prefix block built from top-K text-similar cached rows."""
    hits = text_lookup(query_text, k=k, min_similarity=min_similarity)
    if not hits:
        return ""
    lines = ["# Similar verified solutions retrieved from cache:", ""]
    for i, h in enumerate(hits, 1):
        q = h["question"]
        if len(q) > max_question_chars:
            q = q[:max_question_chars] + "..."
        c = h["code"]
        if len(c) > max_code_chars:
            c = c[:max_code_chars] + "\n# ...(truncated)"
        lines.append(f"# --- Example {i} (sim={h['similarity']:.2f}, "
                     f"wins={h['success_count']}) ---")
        lines.append(f"# Problem: {q}")
        lines.append(c)
        lines.append("")
    lines.append(
        "# Your task — use the above as reference where relevant, "
        "but write a fresh solution:")
    lines.append("")
    return "\n".join(lines)

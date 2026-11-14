"""Small lexical retrievers for the live harness, run over eligible passages only.

A passage is one section of one source version, keyed ``source_id#section_id``
(:attr:`Ref.label`), the unit the labels cite. For each case the pool is every
section whose source passes :func:`~rageval.legal_demo.model.ineligibility`, so
corpus statistics, scores and returned ids never come from text the requester
cannot use. Both retrievers are lexical:

- ``bm25``: Okapi BM25 (``k1 = 1.5``, ``b = 0.75``) with the non-negative idf
  ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
- ``tfidf``: cosine between L2-normalised TF-IDF vectors with the smoothed idf
  ``ln((1 + N) / (1 + df)) + 1``. It is a lexical vector space, **not** a
  learned semantic embedding; no model is loaded.

Words are lowercase ASCII letters and digits minus the shared stopwords.
Only passages that share a word with the query score above zero, and only
those are returned. Scores are rounded to :data:`SCORE_DECIMALS` decimals and
ties break on the passage id, so the order does not depend on dict order or on
the last bit of ``math.log`` across platforms.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Final

from .model import STOPWORDS, Case, Dataset, ineligibility

SCORE_DECIMALS: Final = 9
BM25_K1: Final = 1.5
BM25_B: Final = 0.75
RETRIEVERS: Final = ("bm25", "tfidf")

Ranking = tuple[tuple[str, float], ...]


def tokens(text: str) -> list[str]:
    """Lowercase ASCII words of ``text`` minus STOPWORDS, in order, repeats kept."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def eligible_passages(dataset: Dataset, case: Case) -> dict[str, str]:
    """``{source_id#section_id: text}`` for every section ``case`` may use, sorted by id."""
    pool = {}
    for source_id in sorted(dataset.sources):
        source = dataset.sources[source_id]
        if ineligibility(source, case):
            continue
        for section_id in sorted(source.sections):
            pool[f"{source_id}#{section_id}"] = source.sections[section_id]
    return pool


def _ranked(scores: Mapping[str, float], depth: int) -> Ranking:
    rounded = ((pid, round(score, SCORE_DECIMALS)) for pid, score in scores.items())
    kept = [(pid, score) for pid, score in rounded if score > 0]
    kept.sort(key=lambda pair: (-pair[1], pair[0]))
    return tuple(kept[:depth])


def _document_frequency(docs: Mapping[str, list[str]]) -> Counter[str]:
    return Counter(word for words in docs.values() for word in set(words))


def bm25(passages: Mapping[str, str], query: str, depth: int) -> Ranking:
    """Up to ``depth`` passages by BM25 score, best first, scores above zero only."""
    docs = {pid: tokens(text) for pid, text in passages.items()}
    if not docs:
        return ()
    n = len(docs)
    df = _document_frequency(docs)
    avg_len = sum(len(words) for words in docs.values()) / n or 1.0
    terms = Counter(word for word in tokens(query) if word in df)
    scores = {}
    for pid, words in docs.items():
        tf = Counter(words)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * len(words) / avg_len)
        scores[pid] = sum(
            qtf
            * math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            * tf[term]
            * (BM25_K1 + 1)
            / (tf[term] + norm)
            for term, qtf in terms.items()
        )
    return _ranked(scores, depth)


def _unit_vector(words: list[str], idf: Mapping[str, float]) -> dict[str, float]:
    weights = {word: count * idf[word] for word, count in Counter(words).items() if word in idf}
    length = math.sqrt(sum(w * w for w in weights.values()))
    return {word: w / length for word, w in weights.items()} if length else {}


def tfidf(passages: Mapping[str, str], query: str, depth: int) -> Ranking:
    """Up to ``depth`` passages by TF-IDF cosine, best first, scores above zero only."""
    docs = {pid: tokens(text) for pid, text in passages.items()}
    if not docs:
        return ()
    n = len(docs)
    idf = {
        word: math.log((1 + n) / (1 + count)) + 1
        for word, count in _document_frequency(docs).items()
    }
    q = _unit_vector(tokens(query), idf)
    scores = {}
    for pid, words in docs.items():
        vector = _unit_vector(words, idf)
        scores[pid] = sum(weight * vector.get(word, 0.0) for word, weight in q.items())
    return _ranked(scores, depth)


RANKERS: Final = {"bm25": bm25, "tfidf": tfidf}

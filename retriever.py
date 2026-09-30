"""Local hybrid retrieval over page-addressable PDF text."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def normalize_text(value: str) -> str:
    """Create one retrieval form for Thai text, spacing, punctuation, and zero-width chars.

    This is deliberately used for both corpus and queries.  For example
    ``แรง ขาย ?``, ``แรง-ขาย`` and ``แรงขาย!!!`` become the same search form.
    """
    text = unicodedata.normalize("NFKC", str(value or "").lower())
    text = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", text)
    return re.sub(r"[^\w\u0e00-\u0e7f]", "", text)


def contains_term(text: str, term: str) -> bool:
    """Match an English acronym/word as a whole, while allowing Thai adjacency."""
    term = term.strip()
    if not term:
        return False
    pattern = re.escape(term)
    if term[0].isascii() and term[0].isalnum():
        pattern = r"(?<![a-zA-Z0-9])" + pattern
    if term[-1].isascii() and term[-1].isalnum():
        pattern += r"(?![a-zA-Z0-9])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def char_tokens(value: str, width: int = 3) -> list[str]:
    """Character tokens work for Thai without requiring a word-segmentation service."""
    text = re.sub(r"\s+", "", normalize_text(value))
    if len(text) <= width:
        return [text] if text else []
    return [text[i : i + width] for i in range(len(text) - width + 1)]


@dataclass(frozen=True)
class Record:
    record_id: str
    question: str
    answer: str
    keywords: str
    source: str
    page: str
    excerpt: str
    search_text: str
    source_page: str = ""


@dataclass(frozen=True)
class Hit:
    record: Record
    dense_score: float
    bm25_score: float
    rrf_score: float


class HybridRetriever:
    def __init__(self, records: list[Record]) -> None:
        if not records:
            raise ValueError("ไม่พบข้อความที่ค้นหาได้จากคู่มือ PDF")
        self.records = records
        texts = [normalize_text(r.search_text) for r in records]
        # Character n-grams handle Thai without relying on exact word spacing.
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True, lowercase=False)
        self.matrix = normalize(self.vectorizer.fit_transform(texts))
        self.docs = [char_tokens(t) for t in texts]
        self.doc_lengths = [len(d) or 1 for d in self.docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.df: dict[str, int] = {}
        for doc in self.docs:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    def _bm25(self, question: str) -> list[float]:
        query = char_tokens(question)
        n, k1, b = len(self.docs), 1.5, 0.75
        scores = []
        for doc, dl in zip(self.docs, self.doc_lengths):
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for token in query:
                if token not in counts:
                    continue
                idf = math.log(1 + (n - self.df.get(token, 0) + 0.5) / (self.df.get(token, 0) + 0.5))
                tf = counts[token]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / self.avgdl))
            scores.append(score)
        return scores

    def search(self, question: str, top_k: int = 24) -> list[Hit]:
        q = self.vectorizer.transform([normalize_text(question)])
        dense = (self.matrix @ q.T).toarray().ravel()
        bm25 = self._bm25(question)
        dense_rank = sorted(range(len(self.records)), key=lambda i: dense[i], reverse=True)
        bm25_rank = sorted(range(len(self.records)), key=lambda i: bm25[i], reverse=True)
        rrf: dict[int, float] = {}
        for rank, index in enumerate(dense_rank[:30], 1):
            rrf[index] = rrf.get(index, 0) + 1 / (60 + rank)
        for rank, index in enumerate(bm25_rank[:30], 1):
            rrf[index] = rrf.get(index, 0) + 1 / (60 + rank)
        # First collect a wider set of passages. The LLM reranker needs the
        # right page to be present even when an adjacent chunk scores higher.
        pool_size = max(80, top_k * 4)
        chosen = set(dense_rank[:pool_size]) | set(sorted(rrf, key=rrf.get, reverse=True)[:pool_size])

        # The source PDFs are paginated books. Rank at page level so several
        # useful passages on one page reinforce each other instead of returning
        # multiple near-duplicate chunks as separate candidates.
        pages: dict[tuple[str, str], list[int]] = {}
        for index in chosen:
            record = self.records[index]
            pages.setdefault((record.source, record.page), []).append(index)

        page_hits: list[tuple[float, int]] = []
        for indexes in pages.values():
            indexes.sort(key=lambda i: rrf.get(i, 0.0), reverse=True)
            weights = (1.0, 0.18, 0.08)
            page_score = sum(weight * rrf.get(i, 0.0) for weight, i in zip(weights, indexes))
            # Return the most relevant passage from the page, while its score
            # reflects corroborating matches elsewhere on that same page.
            page_hits.append((page_score, indexes[0]))
        page_hits.sort(key=lambda item: item[0], reverse=True)
        return [
            Hit(self.records[index], float(dense[index]), float(bm25[index]), float(page_score))
            for page_score, index in page_hits[:top_k]
        ]

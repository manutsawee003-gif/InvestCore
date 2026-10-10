"""Local hybrid retrieval for Thai/English text without a word segmenter.

Three signals are combined for every chunk:

* exact phrase match: the user's key words appear verbatim (after normalising
  spaces, punctuation and case). This is what makes "any word from the book"
  findable.
* BM25 over character 2/3-grams: robust keyword ranking for Thai, which has no
  spaces between words.
* TF-IDF cosine over character 2-4-grams: tolerant of small spelling or OCR
  differences.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


def normalize_text(value: str) -> str:
    """One search form for corpus and queries: lower case, no spaces/punctuation."""
    text = unicodedata.normalize("NFKC", str(value or "").lower())
    text = text.replace("ํา", "ำ")  # NIKHAHIT + SARA AA == SARA AM
    text = re.sub(r"[\s​‌‍﻿]+", "", text)
    return re.sub(r"[^\w฀-๿]", "", text)


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


# Question words and polite particles carry no topic information. Removing them
# leaves the words the user is actually looking for.
_THAI_FILLER = [
    "ช่วยอธิบาย", "อธิบาย", "ช่วยบอก", "บอกหน่อย", "ขอทราบ", "อยากรู้", "อยากทราบ", "ช่วย", "แนะนำ",
    "คืออะไร", "หมายถึงอะไร", "หมายถึง", "แปลว่าอะไร", "แปลว่า", "ความหมายของ", "ความหมาย",
    "อย่างไรบ้าง", "อย่างไร", "ยังไงบ้าง", "ยังไง", "อะไรบ้าง", "อะไร", "มีกี่แบบ", "มีกี่ประเภท", "กี่แบบ", "กี่ประเภท",
    "มีอะไรบ้าง", "มีอะไร", "ทำไม", "เพราะอะไร", "ได้ไหม", "ไหม", "มั้ย", "หรือไม่", "หรือเปล่า", "เท่าไร", "เท่าไหร่",
    "ครับ", "ค่ะ", "คะ", "นะคะ", "นะครับ", "จ้า", "หน่อย", "บ้าง", "เกี่ยวกับ", "คือ",
]
_ENGLISH_FILLER = ["what is", "what are", "what", "how to", "how", "explain", "meaning of", "is", "are", "the", "of", "a", "an"]
_FILLER_RE = re.compile(
    "|".join(sorted((re.escape(w) for w in _THAI_FILLER), key=len, reverse=True))
    + "|" + "|".join(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])" for w in sorted(_ENGLISH_FILLER, key=len, reverse=True)),
    re.IGNORECASE,
)
# Thai connectives used as extra split points (the unsplit phrase is kept too).
_CONNECTIVE_RE = re.compile(r"ของ|และ|กับ|หรือ|ระหว่าง|ต่างจาก|เทียบ|ใน|ที่")


def key_phrases(query: str) -> list[str]:
    """Return normalised phrases from the query worth matching verbatim."""
    cleaned = _FILLER_RE.sub(" ", query)
    phrases: list[str] = []

    def add(value: str) -> None:
        value = normalize_text(value)
        if len(value) >= 2 and value not in phrases:
            phrases.append(value)

    add(cleaned)
    for part in re.split(r"[\s,/?!()\"'“”]+", cleaned):
        add(part)
        for piece in _CONNECTIVE_RE.split(part):
            if len(normalize_text(piece)) >= 3:
                add(piece)
    return phrases


@dataclass(frozen=True)
class Record:
    record_id: str
    page: str
    heading: str
    text: str           # shown to the user and the LLM
    search_text: str    # indexed for retrieval
    source: str = ""


@dataclass(frozen=True)
class Hit:
    record: Record
    score: float
    phrase_score: float = 0.0


class HybridRetriever:
    def __init__(self, records: list[Record]) -> None:
        if not records:
            raise ValueError("ไม่พบข้อความที่ค้นหาได้จากคู่มือ")
        self.records = records
        self.norm = [normalize_text(r.search_text) for r in records]
        self.tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, lowercase=False)
        self.tfidf_matrix = normalize(self.tfidf.fit_transform(self.norm))

        # BM25 on character 2/3-grams, computed with sparse matrices.
        self.counter = CountVectorizer(analyzer="char", ngram_range=(2, 3), lowercase=False)
        tf = self.counter.fit_transform(self.norm).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        k1, b = 1.4, 0.75
        denom_per_doc = k1 * (1 - b + b * lengths / lengths.mean())
        tf = tf.tocoo()
        weights = tf.data * (k1 + 1) / (tf.data + denom_per_doc[tf.row])
        from scipy.sparse import csr_matrix
        self.bm25_matrix = csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape)

    def _phrase_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.records))
        for phrase in key_phrases(query):
            # Longer phrases are more specific; a 2-char hit is weak evidence.
            weight = min(len(phrase), 12) / 12
            for i, text in enumerate(self.norm):
                count = text.count(phrase)
                if count:
                    scores[i] += weight * (1 + 0.15 * min(count - 1, 4))
        return scores

    @staticmethod
    def _scale(values: np.ndarray) -> np.ndarray:
        top = values.max()
        return values / top if top > 0 else values

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        normalized = normalize_text(query)
        if not normalized:
            return []
        dense = (self.tfidf_matrix @ self.tfidf.transform([normalized]).T).toarray().ravel()
        q_counts = self.counter.transform([normalized]).toarray().ravel()
        bm25 = self.bm25_matrix @ (self.idf * (q_counts > 0))
        phrase = self._phrase_scores(query)
        combined = 0.35 * self._scale(dense) + 0.35 * self._scale(bm25) + 0.6 * self._scale(phrase)
        order = np.argsort(-combined)[:top_k]
        return [Hit(self.records[i], float(combined[i]), float(phrase[i])) for i in order if combined[i] > 0]

"""Local hybrid RAG retrieval. The Excel workbook is the only knowledge source."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TEXT_COLUMNS = ("Question", "Search_Aliases", "Answer", "Source_Excerpt", "Search_Text")


def normalize_text(value: str) -> str:
    """Create one retrieval form for Thai text, spacing, punctuation, and zero-width chars.

    This is deliberately used for both corpus and queries.  For example
    ``แรง ขาย ?``, ``แรง-ขาย`` and ``แรงขาย!!!`` become the same search form.
    """
    text = unicodedata.normalize("NFKC", str(value or "").lower())
    text = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", text)
    return re.sub(r"[^\w\u0e00-\u0e7f]", "", text)


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


@dataclass(frozen=True)
class Hit:
    record: Record
    dense_score: float
    bm25_score: float
    rrf_score: float


class HybridRetriever:
    def __init__(self, records: list[Record]) -> None:
        if not records:
            raise ValueError("ไม่พบรายการ Q&A ในแผ่นงาน Dataset")
        self.records = records
        texts = [normalize_text(r.search_text) for r in records]
        # Vector search (the dense arm). Local vectors keep dataset content on this machine.
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True, lowercase=False)
        self.matrix = normalize(self.vectorizer.fit_transform(texts))
        self.docs = [char_tokens(t) for t in texts]
        self.doc_lengths = [len(d) or 1 for d in self.docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.df: dict[str, int] = {}
        for doc in self.docs:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    @classmethod
    def from_excel(cls, path: str | Path) -> "HybridRetriever":
        workbook = pd.ExcelFile(path)
        sheet = next((s for s in workbook.sheet_names if "Dataset" in s), None)
        if sheet is None:
            raise ValueError("ไม่พบแผ่นงาน Dataset ในไฟล์ Excel")
        frame = pd.read_excel(path, sheet_name=sheet).fillna("")
        required = {"ID", "Question", "Answer", "Source"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"คอลัมน์ที่จำเป็นหายไป: {', '.join(sorted(missing))}")
        records = []
        for _, row in frame.iterrows():
            content = " | ".join(str(row.get(c, "")) for c in TEXT_COLUMNS)
            records.append(Record(str(row["ID"]), str(row["Question"]), str(row["Answer"]),
                                  str(row.get("Keywords", "")), str(row["Source"]), str(row.get("Page", "")),
                                  str(row.get("Source_Excerpt", "")), content))
        return cls(records)

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

    def search(self, question: str, top_k: int = 4) -> list[Hit]:
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
        # Preserve the two strongest vector matches so RRF cannot bury a semantic result.
        chosen = set(dense_rank[:2]) | set(sorted(rrf, key=rrf.get, reverse=True)[:top_k])
        ranking = sorted(chosen, key=lambda i: rrf.get(i, 0), reverse=True)[:max(top_k, 2)]
        # A user-supplied dataset record ID is an explicit source selection.
        requested_ids = set(re.findall(r"(?:\bID|ข้อ)\s*[:#-]?\s*(\d{1,4})\b", question, flags=re.IGNORECASE))
        exact_indexes = [i for i, record in enumerate(self.records) if record.record_id.upper() in requested_ids]
        # A complete query equal to an approved dataset keyword is in scope even
        # when it is very short (for example, "หุ้น").  Rank its records first.
        normalized_question = normalize_text(question)
        keyword_indexes = [i for i, record in enumerate(self.records) if normalize_text(record.keywords) == normalized_question]
        keyword_indexes.sort(key=lambda i: ("คืออะไร" not in self.records[i].question, -dense[i]))
        priority = exact_indexes + [i for i in keyword_indexes if i not in exact_indexes]
        ranking = (priority + [i for i in ranking if i not in priority])[:max(top_k, 2)]
        return [Hit(self.records[i], float(dense[i]), float(bm25[i]), rrf.get(i, 0.0)) for i in ranking]

    def exact_id_exists(self, question: str) -> bool:
        ids = re.findall(r"(?:\bID|ข้อ)\s*[:#-]?\s*(\d{1,4})\b", question, flags=re.IGNORECASE)
        known = {r.record_id.upper() for r in self.records}
        return any(i.upper() in known for i in ids)

    def exact_keyword_exists(self, question: str) -> bool:
        normalized_question = normalize_text(question)
        return bool(normalized_question) and any(normalize_text(r.keywords) == normalized_question for r in self.records)

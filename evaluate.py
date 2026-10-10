"""Measure retrieval: pick random phrases from the handbook and check they are found.

    py evaluate.py            # 300 random phrases, offline (no API calls)

A phrase counts as found when its page is among the top 5 retrieved chunks,
which is what the answer model receives as context.
"""
from __future__ import annotations

import random
import re
import sys

from pdf_knowledge import PDFKnowledgeBase
from retriever import normalize_text


def sample_phrases(kb: PDFKnowledgeBase, count: int, seed: int = 7) -> list[tuple[str, set[str]]]:
    """Random 1-3 word phrases that appear on at most 3 pages (so the page is identifiable)."""
    rng = random.Random(seed)
    page_text: dict[str, str] = {}
    for r in kb.records:
        page_text[r.page] = page_text.get(r.page, "") + normalize_text(r.text)
    samples = []
    ocr = [r for r in kb.records if "-ocr-" in r.record_id]
    while len(samples) < count:
        record = rng.choice(ocr)
        words = [w for w in re.split(r"\s+", re.sub(r"[#*>|\-]+", " ", record.text)) if re.search(r"[ก-๙A-Za-z]{2}", w)]
        if len(words) < 3:
            continue
        start = rng.randrange(len(words))
        phrase = " ".join(words[start:start + rng.choice((1, 2, 3))])[:40]
        key = normalize_text(phrase)
        pages = {p for p, text in page_text.items() if key in text}
        if len(key) >= 4 and 1 <= len(pages) <= 3:
            samples.append((phrase, pages))
    return samples


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    kb = PDFKnowledgeBase()
    found1 = found5 = 0
    misses = []
    for phrase, expected in sample_phrases(kb, count):
        pages = [hit.record.page for hit in kb.search(phrase, top_k=5)]
        found1 += bool(pages) and pages[0] in expected
        found5 += bool(expected & set(pages))
        if not expected & set(pages):
            misses.append((phrase, sorted(expected), pages))
    print(f"knowledge file: {kb.path.name}, chunks: {len(kb.records)}")
    print(f"random phrases: {count}  found@1: {found1 / count:.1%}  found@5: {found5 / count:.1%}")
    for phrase, page, pages in misses[:15]:
        print(f"  miss: {phrase!r} expected pages {page}, got {pages}")


if __name__ == "__main__":
    main()

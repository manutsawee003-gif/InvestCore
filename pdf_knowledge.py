"""Load data/knowledge.md (built by build_knowledge.py) into searchable chunks."""
from __future__ import annotations

import re
from pathlib import Path

from retriever import HybridRetriever, Hit, Record

DATA = Path(__file__).resolve().parent / "data"
KNOWLEDGE_FILES = ("knowledge.md", "knowledge_ocr.md")  # newest first
NATIVE_HEADING = "### ข้อความจากชั้นข้อความ PDF"
SOURCE_NAME = "Datasetหุ้น.pdf"


def page_reference(record: Record) -> str:
    return f"หน้า PDF {record.page}"


def _clean(markdown: str) -> str:
    text = markdown.replace("&amp;", "&")
    text = re.sub(r"</?figure>", "", text)
    text = re.sub(r"<[^>]+>", " ", text)          # stray HTML from OCR tables
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_sections(text: str, limit: int = 900, overlap: int = 150) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) chunks of roughly `limit` characters."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+)", line)
        if heading:
            sections.append((heading.group(1).strip(), [line]))
        else:
            sections[-1][1].append(line)

    chunks: list[tuple[str, str]] = []
    current_heading, current = "", ""
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        if current and len(current) + len(body) > limit:
            chunks.append((current_heading, current))
            current_heading, current = heading, ""
        current_heading = current_heading or heading
        current = f"{current}\n\n{body}".strip()
        while len(current) > limit * 1.6:   # one very long section
            cut = current.rfind("\n", 0, limit) if current.rfind("\n", 0, limit) > limit // 2 else limit
            chunks.append((current_heading, current[:cut]))
            current = current[max(cut - overlap, 0):]
    if current:
        chunks.append((current_heading, current))
    return chunks


def load_records(path: Path) -> list[Record]:
    content = path.read_text(encoding="utf-8")
    pages = re.findall(r"(?ms)^## หน้า PDF (\d+)\s*\n(.*?)(?=^## หน้า PDF \d+\s*$|\Z)", content)
    if not pages:
        raise ValueError(f"No '## หน้า PDF N' sections found in {path}")

    records: list[Record] = []
    for number, body in pages:
        ocr, _, native = body.partition(NATIVE_HEADING)
        if "### OCR text" in ocr:                   # legacy knowledge_ocr.md layout
            native, _, ocr = ocr.partition("### OCR text")
        ocr, native = _clean(ocr), _clean(native)
        titles = [h for h in re.findall(r"(?m)^\s*#{1,2}\s+(.+)$", ocr) if len(h) < 80][:3]
        page_title = " / ".join(titles)
        for kind, text in (("ocr", ocr), ("pdf", native)):
            for heading, chunk in split_sections(text):
                if len(re.sub(r"\W", "", chunk)) < 15:
                    continue
                records.append(Record(
                    record_id=f"p{number}-{kind}-{len(records)}",
                    page=number,
                    heading=heading or page_title,
                    text=chunk,
                    search_text=f"{page_title}\n{heading}\n{chunk}",
                    source=SOURCE_NAME,
                ))
    return records


class PDFKnowledgeBase:
    def __init__(self, directory: str | Path | None = None) -> None:
        data = Path(directory) / "data" if directory and (Path(directory) / "data").is_dir() else DATA
        path = next((data / name for name in KNOWLEDGE_FILES if (data / name).exists()), None)
        if path is None:
            raise FileNotFoundError("data/knowledge.md not found. Run: py build_knowledge.py")
        self.path = path
        self.records = load_records(path)
        self.retriever = HybridRetriever(self.records)

    def search(self, query: str | list[str], top_k: int = 8, per_page: int = 2) -> list[Hit]:
        """Search one or more query variants and return diverse, de-duplicated chunks."""
        queries = [query] if isinstance(query, str) else [q for q in query if q and q.strip()]
        best: dict[str, Hit] = {}
        for index, variant in enumerate(queries):
            weight = 1.0 if index == 0 else 0.85
            for hit in self.retriever.search(variant, top_k=top_k * 4):
                scored = Hit(hit.record, hit.score * weight, hit.phrase_score)
                previous = best.get(hit.record.record_id)
                # A chunk found by several variants is more likely relevant.
                if previous:
                    scored = Hit(hit.record, max(previous.score, scored.score) + 0.1 * min(previous.score, scored.score),
                                 max(previous.phrase_score, hit.phrase_score))
                best[hit.record.record_id] = scored
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
        results: list[Hit] = []
        per_page_count: dict[str, int] = {}
        for hit in ranked:
            if per_page_count.get(hit.record.page, 0) >= per_page:
                continue
            per_page_count[hit.record.page] = per_page_count.get(hit.record.page, 0) + 1
            results.append(hit)
            if len(results) >= top_k:
                break
        return results

"""The two supplied SET e-books form InvestCore's answerable knowledge base."""
from __future__ import annotations

import re
from pathlib import Path

import fitz

from retriever import HybridRetriever, Record, Hit, contains_term, normalize_text

PDF_PREFIXES = ("TSI_eBook_028_Inv_", "TSI_eBook_062_Inv_")


def split_page(text: str, limit: int = 1050) -> list[str]:
    """Keep original PDF words while grouping short visual lines into context."""
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in text.splitlines()]
    paragraphs = [part for part in paragraphs if part]
    chunks: list[str] = []
    current = ""
    for part in paragraphs:
        if current and len(current) + len(part) + 1 > limit:
            chunks.append(current)
            current = current[-180:] + " " + part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def records_from_markdown(markdown_path: Path, sources: list[Path]) -> list[Record]:
    """Index the user-supplied page-by-page Markdown with PDF page mappings."""
    content = markdown_path.read_text(encoding="utf-8")
    page_pattern = re.compile(r"(?ms)^## หน้า PDF (\d+)\s*\n(.*?)(?=^## หน้า PDF \d+\s*$|\Z)")
    pages = [(int(number), body) for number, body in page_pattern.findall(content)]
    if len(pages) != 134 or [number for number, _ in pages] != list(range(1, 135)):
        raise ValueError("Markdown ต้องมีหน้า PDF ครบ 1–134 และเรียงตามลำดับ")

    if len(sources) == 1:
        with fitz.open(sources[0]) as document:
            first_source_pages = document.page_count
        expected_total = first_source_pages
    else:
        with fitz.open(sources[0]) as first_document:
            first_source_pages = first_document.page_count
        with fitz.open(sources[1]) as second_document:
            expected_total = first_source_pages + second_document.page_count
    if expected_total != 134:
        raise ValueError(f"จำนวนหน้า PDF ต้นทางไม่ตรงกับ Markdown: {expected_total}")

    records: list[Record] = []
    serial = 1
    for combined_page, raw_body in pages:
        if len(sources) == 1 or combined_page <= first_source_pages:
            source = sources[0]
            pdf_page = combined_page
        else:
            source = sources[1]
            pdf_page = combined_page - first_source_pages
        body_lines = []
        for line in raw_body.splitlines():
            line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
            if "หน้านี้ไม่มีข้อความที่คัดลอกได้จาก PDF" in line or "ไม่มีข้อความที่ดึงได้จากหน้านี้" in line:
                continue
            body_lines.append(line)
        page_text = "\n".join(body_lines).strip()
        headings = [
            match.group(1).strip(" ?!：:") for match in re.finditer(r"(?m)^\s*#{1,6}\s+(.{2,65})$", raw_body)
        ]
        headings = [heading for heading in headings if heading and "หน้า PDF" not in heading]
        for chunk in split_page(page_text):
            if len(chunk) < 12:
                continue
            page_title = f"PDF page {pdf_page}"
            records.append(Record(
                record_id=f"MD-{serial}", question=" | ".join(headings) or page_title, answer=chunk,
                keywords="", source=source.name, page=str(combined_page), excerpt=chunk,
                search_text=f"{chunk} {' '.join(headings)}", source_page=str(pdf_page),
            ))
            serial += 1
    if not records:
        raise ValueError("ไม่พบข้อความที่ค้นหาได้ใน Markdown")
    return records


def page_reference(record: Record) -> str:
    """Cite the continuous 134-page set and map it to its original PDF file."""
    if record.source_page:
        return f"หน้า PDF รวม {record.page} (ไฟล์นี้หน้า {record.source_page})"
    return f"หน้า PDF {record.page}"


class PDFKnowledgeBase:
    def __init__(self, directory: str | Path) -> None:
        folder = Path(directory)
        paths = [next(iter(sorted(folder.glob(f"{prefix}*.pdf"))), None) for prefix in PDF_PREFIXES]
        if any(path is None for path in paths):
            combined = next(iter(sorted(folder.glob("Datasetหุ้น.pdf"))), None)
            if combined is None:
                raise FileNotFoundError("Required InvestCore PDF files were not found")
            paths = [combined]

        source_markdown = Path(__file__).resolve().parent / "data" / "Datasetหุ้น_ครบทุกหน้า.md"
        if not source_markdown.exists():
            raise FileNotFoundError("The page-by-page Datasetหุ้น_ครบทุกหน้า.md was not found")
        records = records_from_markdown(source_markdown, paths)
        self.retriever = HybridRetriever(records)
        foreign_records = [record for record in records if "062" in record.source]
        if not foreign_records and len(paths) == 1:
            foreign_records = [record for record in records if int(record.page) > 60]
        self.foreign_retriever = HybridRetriever(foreign_records)

    @staticmethod
    def has_exact_heading(hit: Hit, term: str) -> bool:
        return any(normalize_text(title) == normalize_text(term) for title in hit.record.question.split(" | "))

    @staticmethod
    def has_definition(hit: Hit, term: str) -> bool:
        return bool(contains_term(hit.record.answer, term) and re.search(r"(?:คือ|หมายถึง|ย่อมาจาก|กรอบความคิด|กรอบควำมคิด)", hit.record.answer))

    def clarification_options(self, term: str) -> list[str]:
        options = []
        for record in self.retriever.records:
            for phrase in re.findall(re.escape(term) + r"(?:ของ|จาก)[ก-๙A-Za-z/]{2,30}", record.answer):
                if phrase.endswith(("ได้", "ที่", "การที่", "เพิ่มขึ้น")) or len(phrase) > 36:
                    continue
                if phrase not in options:
                    options.append(phrase)
        return options

    @staticmethod
    def topic_term(query: str) -> str | None:
        """Recognize a short topic request without inventing a topic from a sentence."""
        text = re.sub(r"\s+", " ", query).strip(" ?!？。")
        text = re.sub(r"^(?:ช่วย)?(?:อธิบาย|บอก|ขอความหมายของ|ความหมายของ)\s*", "", text, flags=re.I)
        text = re.sub(r"\s*(?:คือ(?:อะไร)?|หมายถึง(?:อะไร)?|แปลว่าอะไร|คือยังไง)\s*$", "", text, flags=re.I).strip()
        if not text or len(text) > 48 or len(text.split()) > 5:
            return None
        if text in {"ดีไหม", "อันนี้", "อันนั้น", "อะไร", "ทำยังไง", "ยังไง", "คืออะไร"}:
            return None
        return text

    def search(self, query: str):
        # These are named products/topics whose detailed treatment is in the
        # foreign-investment playbook.  Filtering avoids a brief mention of
        # the same acronym in the Thai-stock book displacing the explanation.
        product = re.search(r"\b(ETF|DR|DW|FIF|MSCI|FCD|FCN|ELN)\b", query, re.IGNORECASE)
        term = self.topic_term(query)
        queries = [query]
        if term:
            for variant in (term, f"{term} คืออะไร", f"ความหมายของ {term}"):
                if normalize_text(variant) not in {normalize_text(q) for q in queries}:
                    queries.append(variant)
        if product and term and normalize_text(term) == normalize_text(product.group(1)):
            queries.append(f"{term} ย่อมาจาก")
        # Thai users often say “ค่าเงิน” while the handbook consistently uses
        # “อัตราแลกเปลี่ยน”. Add both handbook terms to retrieval for this topic.
        if re.search(r"ค่าเงิน|อัตราแลกเปลี่ยน|แลกเปลี่ยนเงินตรา|exchange rate|currency risk", query, re.IGNORECASE):
            queries.append(f"{query} อัตราแลกเปลี่ยน ค่าเงิน ผลตอบแทน การเปลี่ยนแปลง")
        retriever = self.foreign_retriever if re.search(r"\b(?:ETF|DR|DW|FIF|MSCI|FCD|FCN|ELN)\b|ต่างประเทศ|offshore|fund flow|ค่าเงิน|อัตราแลกเปลี่ยน|exchange rate|currency risk", query, re.IGNORECASE) else self.retriever
        merged: dict[tuple[str, str], tuple[float, Hit]] = {}
        for query_index, variant in enumerate(queries):
            weight = 1.0 if query_index == 0 else 0.45
            for rank, hit in enumerate(retriever.search(variant, top_k=24), 1):
                key = (hit.record.source, hit.record.page)
                previous_score, previous_hit = merged.get(key, (0.0, hit))
                # Keep the best original passage on a page; variants only help discovery.
                representative = hit if query_index == 0 or hit.dense_score > previous_hit.dense_score else previous_hit
                merged[key] = (previous_score + weight / (60 + rank), representative)
        results = []
        for score, hit in merged.values():
            if term:
                titles = [title.strip() for title in hit.record.question.split(" | ")]
                if any(normalize_text(title) == normalize_text(term) for title in titles):
                    score += 0.025
                if contains_term(hit.record.answer, term) and re.search(r"(?:คือ|หมายถึง|ย่อมาจาก|กรอบความคิด|กรอบควำมคิด)", hit.record.answer):
                    score += 0.010
            results.append((score, hit))
        results.sort(key=lambda item: item[0], reverse=True)
        return [Hit(hit.record, hit.dense_score, hit.bm25_score, score) for score, hit in results[:24]]

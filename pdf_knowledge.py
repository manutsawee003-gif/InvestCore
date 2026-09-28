"""The two supplied SET e-books form InvestCore's answerable knowledge base."""
from __future__ import annotations

import re
from pathlib import Path

import fitz

from retriever import HybridRetriever, Record

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
        for chunk in split_page(page_text):
            if len(chunk) < 12:
                continue
            page_title = f"PDF page {pdf_page}"
            records.append(Record(
                record_id=f"MD-{serial}", question=page_title, answer=chunk,
                keywords="", source=source.name, page=str(combined_page), excerpt=chunk,
                search_text=chunk, source_page=str(pdf_page),
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

    def search(self, query: str):
        # These are named products/topics whose detailed treatment is in the
        # foreign-investment playbook.  Filtering avoids a brief mention of
        # the same acronym in the Thai-stock book displacing the explanation.
        product = re.search(r"\b(ETF|DR|DW|FIF|MSCI|FCD|FCN|ELN)\b", query, re.IGNORECASE)
        if product and re.search(r"คือ|อะไร|หมายถึง|ย่อมาจาก|what is", query, re.IGNORECASE):
            # Definition pages use “ย่อมาจาก” more often than the wording
            # “คืออะไร”, so expand that natural-language question.
            return self.foreign_retriever.search(f"{product.group(1)} ย่อมาจาก", top_k=24)
        # Thai users often say “ค่าเงิน” while the handbook consistently uses
        # “อัตราแลกเปลี่ยน”. Add both handbook terms to retrieval for this topic.
        if re.search(r"ค่าเงิน|อัตราแลกเปลี่ยน|แลกเปลี่ยนเงินตรา|exchange rate|currency risk", query, re.IGNORECASE):
            expanded = f"{query} อัตราแลกเปลี่ยน ค่าเงิน ผลตอบแทน การเปลี่ยนแปลง"
            return self.foreign_retriever.search(expanded, top_k=24)
        if re.search(r"\b(?:ETF|DR|DW|FIF|MSCI|FCD|FCN|ELN)\b|ต่างประเทศ|offshore|fund flow", query, re.IGNORECASE):
            return self.foreign_retriever.search(query, top_k=24)
        return self.retriever.search(query, top_k=24)

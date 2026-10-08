"""Create page-addressable Markdown from the InvestCore PDF with Thai OCR."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import re
from pathlib import Path

import fitz


MIN_NATIVE_CHARS = 100


def ocr_page(page: fitz.Page, tesseract: str, tessdata_dir: Path, dpi: int) -> str:
    """Render one page and OCR it without keeping intermediate image files."""
    with tempfile.TemporaryDirectory(prefix="investcore-ocr-") as directory:
        image = Path(directory) / "page.png"
        page.get_pixmap(dpi=dpi, alpha=False).save(image)
        result = subprocess.run(
            [tesseract, str(image), "stdout", "-l", "tha+eng", "--tessdata-dir", str(tessdata_dir)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return result.stdout.strip()


def build_markdown(pdf: Path, output: Path, tesseract: str, tessdata_dir: Path, dpi: int) -> list[int]:
    """Write Markdown and return all pages that received an OCR pass."""
    if not shutil.which(tesseract) and not Path(tesseract).is_file():
        raise RuntimeError(f"Tesseract executable was not found: {tesseract}")
    if not (tessdata_dir / "tha.traineddata").is_file():
        raise RuntimeError(f"Thai language data was not found in: {tessdata_dir}")

    existing_files = sorted(output.parent.glob("*.md"))
    heading_prefix = "## PDF page "
    if existing_files:
        existing = existing_files[0].read_text(encoding="utf-8")
        match = re.search(r"(?m)^(## .+?)(1)\s*$", existing)
        if match:
            heading_prefix = match.group(1)
    sections = [
        f"# Content from {pdf.name}",
        "",
        "> Native PDF text is preserved; low-text pages include Thai and English OCR.",
    ]
    ocr_pages: list[int] = []
    with fitz.open(pdf) as document:
        for number, page in enumerate(document, start=1):
            native_text = page.get_text("text").strip()
            parts = [f"{heading_prefix}{number}"]
            if native_text:
                parts.extend(["", native_text])
            if len(native_text) < MIN_NATIVE_CHARS:
                ocr_text = ocr_page(page, tesseract, tessdata_dir, dpi)
                ocr_pages.append(number)
                if ocr_text:
                    parts.extend(["", "### OCR text", "", ocr_text])
                else:
                    parts.extend(["", "> OCR found no readable text on this page."])
            sections.extend(["", *parts])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return ocr_pages


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build searchable Markdown with Thai OCR")
    parser.add_argument("--pdf", type=Path, default=next(project.glob("Dataset*.pdf")))
    parser.add_argument("--output", type=Path, default=project / "data" / "knowledge_ocr.md")
    parser.add_argument("--tesseract", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    parser.add_argument("--tessdata-dir", type=Path, default=project / "data" / "tessdata")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    pages = build_markdown(args.pdf, args.output, args.tesseract, args.tessdata_dir, args.dpi)
    print(f"Wrote {args.output}; OCR applied to {len(pages)} pages: {pages}")


if __name__ == "__main__":
    main()

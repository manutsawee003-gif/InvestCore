"""Build data/knowledge.md from the PDF using Typhoon OCR on every page.

Most pages in the dataset are designed infographics. Their headings use
decorative fonts that PDF text extraction drops, and the extracted body text
has broken Thai vowels (e.g. "จ าเป็น" instead of "จำเป็น"). Typhoon OCR reads
the rendered page image instead, so headings and body text become searchable.

Each page is cached in data/ocr_pages/ so the build can resume after errors.
A file in data/ocr_overrides/NNN.md replaces the OCR of page NNN (used for
pages where the OCR model misread a diagram).

    py build_knowledge.py            # OCR missing pages and rebuild knowledge.md
    py build_knowledge.py --force    # re-OCR every page
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import shutil
import subprocess
import tempfile

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

PROJECT = Path(__file__).resolve().parent
PROMPT = (
    "Extract all text from this Thai investment handbook page. Return markdown. "
    "Include every heading, title, label, caption, table and callout, in reading order. "
    "Keep Thai text exactly as written. For pictures without text, write a one-line description."
)


def fix_thai(text: str) -> str:
    """Repair the Thai vowel damage found in this PDF's embedded text layer."""
    # "จ าเป็น" -> "จำเป็น": SARA AM was extracted as space + SARA AA.
    text = re.sub(r"([ก-ฮ]) ?([่-๋]) า", r"\1\2ำ", text)
    text = re.sub(r"([ก-ฮ]) า", r"\1ำ", text)
    lines = []
    for line in text.splitlines():
        # Lines from one decorative font swap the two vowels: "ควำมส ำเร็จ".
        if " ำ" in line or re.search(r"ควำม|กำร|อย่ำง|ว่ำ|จำก", line):
            line = line.replace(" ำ", "\0").replace("ำ", "า").replace("\0", "ำ")
        lines.append(line)
    return "\n".join(lines)


TESSERACT = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def tesseract_letters(png: bytes) -> int | None:
    """Count letters Tesseract sees; None when Tesseract is unavailable."""
    tessdata = PROJECT / "data" / "tessdata"
    if not Path(TESSERACT).is_file() or not (tessdata / "tha.traineddata").is_file():
        return None
    with tempfile.TemporaryDirectory() as folder:
        image = Path(folder) / "page.png"
        image.write_bytes(png)
        out = subprocess.run([TESSERACT, str(image), "stdout", "-l", "tha+eng", "--tessdata-dir", str(tessdata)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    return len(re.findall(r"[ก-๙A-Za-z]", out))


def trim_loops(text: str) -> str:
    """Vision models sometimes loop. Drop lines once a line (or its start) repeats too often."""
    seen: dict[str, int] = {}
    kept = []
    for line in text.replace("\\n", "\n").splitlines():
        key = re.sub(r"\W", "", line)[:25]
        if key:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 3:
                continue
        kept.append(line)
    return "\n".join(kept).strip()


def ocr_page(client: OpenAI, model: str, page_png: bytes) -> str:
    image = base64.b64encode(page_png).decode()
    for attempt in range(5):
        try:
            result = client.chat.completions.create(
                model=model, temperature=0.1, max_completion_tokens=5000, frequency_penalty=0.3,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ]}],
            )
            return trim_loops(result.choices[0].message.content or "")
        except Exception as error:  # rate limits and transient network errors
            wait = 5 * (attempt + 1)
            print(f"  retry in {wait}s: {error}", flush=True)
            time.sleep(wait)
    raise RuntimeError("OCR failed after 5 attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=PROJECT / "Datasetหุ้น.pdf")
    parser.add_argument("--output", type=Path, default=PROJECT / "data" / "knowledge.md")
    parser.add_argument("--cache", type=Path, default=PROJECT / "data" / "ocr_pages")
    parser.add_argument("--model", default="typhoon-ocr-v1.5")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pages", type=int, nargs="*", help="re-OCR only these page numbers")
    args = parser.parse_args()

    load_dotenv(PROJECT / ".env")
    client = OpenAI(api_key=os.environ["TYPHOON_API_KEY"],
                    base_url=os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1"), timeout=180)
    args.cache.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(args.pdf) as document:
        native = {n: fix_thai(page.get_text().strip()) for n, page in enumerate(document, 1)}
        if args.pages:
            todo = args.pages
        else:
            todo = [n for n in native if args.force or not (args.cache / f"{n:03d}.md").exists()]
        images = {n: document[n - 1].get_pixmap(dpi=args.dpi).tobytes("png") for n in todo}

    def work(n: int) -> int:
        letters = tesseract_letters(images[n]) if not native[n] else None
        # Blank or pattern-only pages make the OCR model invent a whole page.
        text = "" if letters is not None and letters < 5 else ocr_page(client, args.model, images[n])
        (args.cache / f"{n:03d}.md").write_text(text, encoding="utf-8")
        return n

    print(f"OCR {len(todo)} of {len(native)} pages", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(work, n) for n in todo]), 1):
            print(f"  page {future.result()} done ({done}/{len(todo)})", flush=True)

    sections = [f"# Content from {args.pdf.name}", "",
                "> Each page: Typhoon OCR of the page image, then the repaired PDF text layer."]
    for n in sorted(native):
        # Hand-checked transcriptions in data/ocr_overrides/ win over model OCR.
        override = PROJECT / "data" / "ocr_overrides" / f"{n:03d}.md"
        source = override if override.exists() else args.cache / f"{n:03d}.md"
        ocr = trim_loops(source.read_text(encoding="utf-8")).replace("&amp;", "&")
        sections += ["", f"## หน้า PDF {n}", "", ocr]
        if native[n]:
            sections += ["", "### ข้อความจากชั้นข้อความ PDF", "", native[n]]
    args.output.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

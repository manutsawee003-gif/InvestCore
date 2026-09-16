from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retriever import Hit, HybridRetriever


def setting(name: str, default: str = "") -> str:
    """Read local .env first, then Streamlit Cloud secrets, without exposing values."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or default)

REFUSAL = "ขออภัยค่ะ ไม่พบข้อมูลที่ยืนยันได้ในชุดข้อมูล InvestCore จึงไม่สามารถตอบนอกเหนือจากชุดข้อมูลนี้ได้ค่ะ"
GREETING = "สวัสดีค่ะ ฉันคือ InvestCore แชตบอตให้ความรู้พื้นฐานด้านหุ้นและตลาดทุน ฉันตอบจากชุดข้อมูลที่กำหนดเท่านั้นค่ะ"


def is_smalltalk(question: str) -> bool:
    return bool(re.fullmatch(r"\s*(สวัสดี|หวัดดี|ดี|hello|hi|คุณทำอะไรได้บ้าง|ช่วยอะไรได้บ้าง)[!?. ]*", question.lower()))


class RAGService:
    def __init__(self, dataset_path: str | Path) -> None:
        load_dotenv()
        self.retriever = HybridRetriever.from_excel(dataset_path)
        self.threshold = float(setting("MIN_DENSE_SIM", "0.25"))
        key = setting("TYPHOON_API_KEY")
        self.client = OpenAI(api_key=key, base_url=setting("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")) if key else None
        self.model = setting("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct")

    def answer(self, question: str) -> tuple[str, list[Hit], str]:
        if is_smalltalk(question):
            return GREETING, [], "smalltalk"
        hits = self.retriever.search(question)
        if not hits or (hits[0].dense_score < self.threshold
                        and not self.retriever.exact_id_exists(question)
                        and not self.retriever.exact_keyword_exists(question)):
            return REFUSAL, hits, "out_of_scope"
        if self.client is None:
            return self._deterministic_answer(hits), hits, "no_api_key"
        try:
            text = self._ask_typhoon(question, hits)
            if not self._is_safe_answer(text):
                return self._deterministic_answer(hits), hits, "verification_fallback"
            return self._with_citations(text, hits), hits, "typhoon"
        except Exception:
            return self._deterministic_answer(hits), hits, "provider_fallback"

    def _ask_typhoon(self, question: str, hits: list[Hit]) -> str:
        context = "\n\n".join(
            f"[RECORD {h.record.record_id}]\nคำถาม: {h.record.question}\nคำตอบในชุดข้อมูล: {h.record.answer}\nแหล่งอ้างอิง: {h.record.source}"
            for h in hits
        )
        system = """คุณคือ InvestCore แชตบอตให้ความรู้พื้นฐานด้านหุ้นและตลาดทุน ตอบภาษาเดียวกับผู้ใช้
กฎบังคับ: ใช้ข้อเท็จจริงจาก CONTEXT เท่านั้น ห้ามใช้ความรู้เดิม ห้ามเดา ห้ามเพิ่มคำอธิบาย/ตัวเลข/หลักทรัพย์ที่ไม่มีใน CONTEXT
ห้ามแนะนำให้ซื้อ ขาย หรือถือหลักทรัพย์ใด และห้ามให้ข้อมูลราคาหุ้นปัจจุบัน
หาก CONTEXT ตอบไม่ได้ ให้ตอบคำเดียวว่า NOT_ENOUGH_EVIDENCE
ตอบกระชับและไม่กล่าวอ้างว่าได้ค้นอินเทอร์เน็ต"""
        result = self.client.chat.completions.create(
            model=self.model, temperature=0.1, max_completion_tokens=500, top_p=0.6,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}],
        )
        return (result.choices[0].message.content or "").strip()

    def _is_safe_answer(self, text: str) -> bool:
        return bool(text) and "NOT_ENOUGH_EVIDENCE" not in text and len(text) <= 3000

    def _deterministic_answer(self, hits: list[Hit]) -> str:
        # Safe degradation: show only Excel fields; never invent a response if Typhoon is unavailable.
        body = "\n\n".join(h.record.answer for h in hits[:2])
        return self._with_citations(body, hits)

    @staticmethod
    def _with_citations(text: str, hits: list[Hit]) -> str:
        refs = "\n".join(f"- ID {h.record.record_id}: {h.record.source}, หน้า {h.record.page}" for h in hits[:2])
        return f"{text}\n\nอ้างอิงจากชุดข้อมูล:\n{refs}"

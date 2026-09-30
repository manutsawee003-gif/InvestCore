import unittest
from pathlib import Path

import fitz

from pdf_knowledge import PDFKnowledgeBase
from rag_service import RAGService
from retriever import contains_term


PDF_DIRECTORY = Path(__file__).resolve().parents[2]


class ShortTopicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RAGService(PDF_DIRECTORY)
        cls.service.client = None
        cls.kb = cls.service.knowledge_base

    def test_mindset_forms_find_same_evidence(self):
        variants = ["mindset", " MINDSET ? ", "Mindset คือ", "mindset คืออะไร", "อธิบาย mindset", "mindset หมายถึงอะไร"]
        pages = [self.kb.search(query)[0].record.page for query in variants]
        self.assertEqual(pages, ["11"] * len(variants))
        self.assertIn("กรอบควำมคิด", self.kb.search("mindset")[0].record.answer)

    def test_ten_real_topics_short_and_full_agree(self):
        topics = ["mindset", "GDP", "CFO", "portfolio", "capital gain", "technical analysis", "DR", "ETF", "DW", "FIF", "MSCI", "FCD", "DCA", "Fund Flow"]
        for topic in topics:
            with self.subTest(topic=topic):
                short = self.kb.search(topic)[0]
                full = self.kb.search(f"{topic} คืออะไร")[0]
                self.assertEqual(short.record.page, full.record.page)
                self.assertTrue(contains_term(short.record.answer, topic))

    def test_ambiguous_and_context_free_questions_clarify(self):
        for query in ("ความเสี่ยง", "ดีไหม", "อันนี้"):
            with self.subTest(query=query):
                answer, hits, route = self.service.answer(query)
                self.assertEqual(route, "needs_clarification")
                self.assertFalse(hits)
                self.assertIn("หัวข้อ", answer)

    def test_unknown_terms_and_out_of_scope_do_not_force_answer(self):
        for query in ("asdfxyz", "ดาวพลูโต", "หุ้นตัวไหนจะขึ้นพรุ่งนี้"):
            with self.subTest(query=query):
                _, hits, route = self.service.answer(query)
                self.assertEqual(route, "out_of_scope")
                self.assertFalse(hits)

    def test_context_only_for_explicit_reference(self):
        history = [{"role": "user", "content": "ETF คืออะไร"}]
        self.assertEqual(self.service._retrieval_query("mindset", history), "mindset")
        self.assertIn("ETF คืออะไร", self.service._retrieval_query("แล้วอันนั้นต่างจาก DR อย่างไร", history))

    def test_acronym_boundary(self):
        self.assertTrue(contains_term("DR และ ETF", "DR"))
        self.assertFalse(contains_term("DRIVER", "DR"))
        self.assertFalse(contains_term("MINDSETTINGS", "MINDSET"))

    def test_pdf_page_mapping_and_source_text(self):
        pdf = PDF_DIRECTORY / "Datasetหุ้น.pdf"
        with fitz.open(pdf) as document:
            self.assertEqual(document.page_count, 134)
            for topic, page in (("MINDSET", 11), ("DCA", 45), ("ETF", 100)):
                with self.subTest(topic=topic):
                    hit = self.kb.search(topic)[0]
                    self.assertEqual(int(hit.record.page), page)
                    self.assertEqual(hit.record.source, pdf.name)
                    self.assertIn(topic.lower(), document[page - 1].get_text().lower())

    def test_citation_comes_from_selected_record(self):
        hit = self.kb.search("mindset")[0]
        answer = self.service._with_citations("ตีความสั้น ๆ: กรอบความคิด", [hit])
        self.assertIn(f"{hit.record.source}, หน้า PDF รวม {hit.record.page}", answer)
        self.assertEqual(answer.count(hit.record.source), 1)


if __name__ == "__main__":
    unittest.main()

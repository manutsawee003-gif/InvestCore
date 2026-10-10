import unittest

from evaluate import sample_phrases
from pdf_knowledge import PDFKnowledgeBase
from rag_service import RAGService
from retriever import contains_term, key_phrases, normalize_text


class KnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RAGService()
        cls.service.client = None  # tests never call the external model
        cls.kb = cls.service.knowledge_base

    def top_pages(self, query, k=3):
        return [hit.record.page for hit in self.kb.search(query, top_k=k)]

    def test_every_page_is_indexed(self):
        pages = {int(r.page) for r in self.kb.records}
        self.assertGreaterEqual(len(pages), 125)

    def test_decorative_headings_are_searchable(self):
        # These headings exist only in the page image, not in the PDF text layer.
        self.assertIn("12", self.top_pages("ตลาดหุ้นคืออะไร"))
        self.assertIn("12", self.top_pages("mai"))

    def test_known_topics(self):
        for query, page in (("mindset", "11"), ("DCA", "45"), ("SET Index", "12")):
            with self.subTest(query=query):
                self.assertIn(page, self.top_pages(query, k=5))

    def test_thai_sara_am_is_repaired(self):
        text = "\n".join(r.text for r in self.kb.records)
        self.assertNotIn("จ าเป็น", text)
        self.assertTrue(self.kb.search("จำเป็น"))

    def test_random_phrases_are_found(self):
        hits = 0
        samples = sample_phrases(self.kb, 150, seed=11)
        for phrase, pages in samples:
            hits += bool(pages & {h.record.page for h in self.kb.search(phrase, top_k=5)})
        self.assertGreaterEqual(hits / len(samples), 0.95)

    def test_answer_without_api_key_shows_source_text(self):
        answer, hits, route = self.service.answer("DCA คืออะไร")
        self.assertEqual(route, "no_api_key")
        self.assertTrue(hits)
        self.assertIn("หน้า PDF", answer)

    def test_smalltalk_does_not_retrieve(self):
        _, hits, route = self.service.answer("สวัสดีครับ")
        self.assertEqual(route, "smalltalk")
        self.assertEqual(hits, [])

    def test_follow_up_uses_previous_question_offline(self):
        queries = self.service.search_queries("แล้วอันนั้นมีความเสี่ยงไหม", [{"role": "user", "content": "ETF คืออะไร"}])
        self.assertTrue(any("ETF" in q for q in queries))


class TextTests(unittest.TestCase):
    def test_key_phrases_strip_question_words(self):
        self.assertEqual(key_phrases("ETF คืออะไรครับ"), ["etf"])
        self.assertIn("dca", key_phrases("ข้อดีของ DCA มีอะไรบ้าง"))

    def test_normalize(self):
        self.assertEqual(normalize_text("แรง ขาย ?"), normalize_text("แรง-ขาย!!"))

    def test_acronym_boundary(self):
        self.assertTrue(contains_term("DR และ ETF", "DR"))
        self.assertFalse(contains_term("DRIVER", "DR"))


if __name__ == "__main__":
    unittest.main()

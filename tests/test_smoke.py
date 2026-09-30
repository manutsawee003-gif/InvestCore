import unittest
from pathlib import Path

from pdf_knowledge import PDFKnowledgeBase
from rag_service import RAGService


PDF_DIRECTORY = Path(__file__).resolve().parents[2]


class RAGSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RAGService(PDF_DIRECTORY)
        # Tests must never call an external model, even if a local key exists.
        cls.service.client = None

    def test_loads_combined_pdf_sections(self):
        records = self.service.knowledge_base.retriever.records
        self.assertGreater(len(records), 100)
        self.assertTrue(any(int(record.page) <= 60 for record in records))
        self.assertTrue(any(int(record.page) > 60 for record in records))
        self.assertTrue(all(record.source == "Datasetหุ้น.pdf" for record in records))

    def test_etf_question_is_grounded_in_foreign_investment_pdf(self):
        hits = self.service.knowledge_base.search("ETF คืออะไร")
        self.assertTrue(any(int(hit.record.page) > 60 for hit in hits))

    def test_follow_up_keeps_conversation_context_for_retrieval(self):
        query = self.service._retrieval_query(
            "แล้วอันนั้นต่างจาก DR อย่างไร",
            [{"role": "user", "content": "ETF คืออะไร"}],
        )
        hits = self.service.knowledge_base.search(query)
        self.assertTrue(hits)
        self.assertTrue(any(int(hit.record.page) > 60 for hit in hits))

    def test_smalltalk_does_not_retrieve(self):
        _, hits, route = self.service.answer("สวัสดีครับ")
        self.assertEqual(route, "smalltalk")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()

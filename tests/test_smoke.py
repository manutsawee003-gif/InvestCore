import unittest
from pathlib import Path

from rag_service import RAGService
from retriever import HybridRetriever, normalize_text

DATASET = Path(__file__).resolve().parents[1] / "data" / "Dataset_หุ้นพื้นฐาน_1200_QA.xlsx"


class RAGSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever.from_excel(DATASET)

    def test_loads_all_records(self):
        self.assertEqual(len(self.retriever.records), 1200)

    def test_explicit_record_id_is_honoured(self):
        self.assertEqual(self.retriever.search("ID 1")[0].record.record_id, "1")

    def test_search_normalization_removes_spacing_and_punctuation(self):
        self.assertEqual(normalize_text("แรง ขาย ?!"), normalize_text("แรง-ขาย"))

    def test_exact_dataset_keyword_is_in_scope(self):
        service = RAGService(DATASET)
        _, hits, route = service.answer("หุ้น")
        self.assertTrue(hits)
        self.assertNotEqual(route, "out_of_scope")

    def test_smalltalk_does_not_retrieve_or_call_llm(self):
        service = RAGService(DATASET)
        for greeting in ("สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ!", "หวัดดีครับ", "hello"):
            with self.subTest(greeting=greeting):
                _, hits, route = service.answer(greeting)
                self.assertEqual(route, "smalltalk")
                self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()

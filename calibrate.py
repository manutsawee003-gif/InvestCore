"""Print retrieval-score distributions; replace question lists with reviewed evaluation questions."""
from pathlib import Path
from statistics import mean

from retriever import HybridRetriever

DATASET = Path(__file__).resolve().parent / "data" / "Dataset_หุ้นพื้นฐาน_1200_QA.xlsx"
IN_SCOPE = ["Mindset คืออะไร", "ตลาดหลักทรัพย์คืออะไร"]
OUT_OF_SCOPE = ["วันนี้ฝนตกไหม", "วิธีทำอาหาร"]
GAPS = ["หุ้น AOT วันนี้ราคาเท่าไร", "ควรซื้อหุ้นตัวไหน"]

if __name__ == "__main__":
    retriever = HybridRetriever.from_excel(DATASET)
    for name, questions in {"in_scope": IN_SCOPE, "out_of_scope": OUT_OF_SCOPE, "known_gaps": GAPS}.items():
        scores = [retriever.search(q)[0].dense_score for q in questions]
        print(name, {"min": min(scores), "max": max(scores), "mean": round(mean(scores), 4)})

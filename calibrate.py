"""Evaluate PDF retrieval against the separate 300-question workbook.

This script never adds Excel questions or answers to the chatbot's knowledge
base.  It only compares the page retrieved from the PDFs with the labelled
page in the evaluation workbook.
"""
from pathlib import Path

import pandas as pd

from pdf_knowledge import PDFKnowledgeBase


PROJECT = Path(__file__).resolve().parent
PDF_DIRECTORY = PROJECT.parent
EVALUATION = next(PDF_DIRECTORY.glob("Dataset_300*.xlsx"))


def source_family(source: str) -> str:
    return "028" if "028" in source else "062"


if __name__ == "__main__":
    knowledge_base = PDFKnowledgeBase(PDF_DIRECTORY)
    questions = pd.read_excel(EVALUATION, sheet_name="Dataset 300 QA").fillna("")
    top1 = top5 = 0
    scores: list[float] = []
    for _, row in questions.iterrows():
        hits = knowledge_base.search(str(row["Question"]))
        family = source_family(str(row["Source"]))
        expected_page = str(row["Page"])
        matching_ranks = [
            rank for rank, hit in enumerate(hits, start=1)
            if family in hit.record.source and hit.record.source_page == expected_page
        ]
        scores.append(hits[0].dense_score)
        top1 += matching_ranks[:1] == [1]
        top5 += bool(matching_ranks)
    print({
        "questions": len(questions),
        "page_recall_at_1": round(top1 / len(questions), 3),
        "page_recall_at_5": round(top5 / len(questions), 3),
        "top_score_min": round(min(scores), 3),
        "top_score_median": round(float(pd.Series(scores).median()), 3),
    })

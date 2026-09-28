# InvestCore — PDF-grounded investment education chatbot

InvestCore answers from these two supplied SET e-books only:

- `TSI_eBook_028_Inv_คู่มือ21-DayChallenge(1).pdf`
- `TSI_eBook_062_Inv_Playbook-21-Day-Challenge-Foreign-Investment.pdf`

The 300-question Excel workbook is an evaluation set. It is not loaded into
the chatbot and cannot influence an answer.

## Run locally

1. Copy `.env.example` to `.env` and set `TYPHOON_API_KEY`.
2. Install packages: `py -m pip install -r requirements.txt`
3. Run the app: `py -m streamlit run app.py`

The searchable knowledge file is
`data/Datasetหุ้น_ครบทุกหน้า.md`, containing pages 1–134 in order. Pages 1–60
map to PDF 028; pages 61–134 map to PDF 062. The two original PDFs must remain
in the parent `Chatbot InvestCore` folder so page references can be mapped back
to each original volume.

## Answer policy

- Answers show a short interpretation and cite the combined PDF page from
  1–134, with the corresponding page in the original PDF in parentheses. The
  retrieved passage is available in a collapsible control.
- The chatbot does not recommend a specific security to buy, sell, or hold.
- Investment questions outside the two books receive a scope-aware response.
- The most recent chat turns are used to resolve follow-up questions.

If retrieval or source verification is unavailable, InvestCore declines to
summarize rather than show a possibly unrelated passage as the answer.

from pathlib import Path

import streamlit as st

from pdf_knowledge import page_reference
from rag_service import RAGService

st.set_page_config(page_title="InvestCore", page_icon="📈")
st.title("InvestCore")
st.caption("แชตบอตความรู้การลงทุน ตอบจากคู่มือ Datasetหุ้น.pdf ทั้ง 134 หน้า")


@st.cache_resource
def get_service() -> RAGService:
    return RAGService(Path(__file__).resolve().parent)


WELCOME = ("สวัสดีครับ ผม InvestCore ถามอะไรจากคู่มือก็ได้เลยครับ เช่น ตลาดหุ้นคืออะไร, "
           "P/E คืออะไร, DCA ทำยังไง, ETF ต่างจาก DR อย่างไร")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

with st.sidebar:
    st.write("แหล่งความรู้: Datasetหุ้น.pdf (OCR ทุกหน้า)")
    if st.button("เริ่มบทสนทนาใหม่"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("พิมพ์คำถามเกี่ยวกับหุ้นและการลงทุน"):
    history = st.session_state.messages[1:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นข้อมูลจากคู่มือ..."):
            answer, hits, _ = get_service().answer(question, history)
        st.markdown(answer)
        if hits:
            pages = sorted({int(h.record.page) for h in hits})
            with st.expander("ดูข้อความต้นฉบับจากคู่มือ (หน้า " + ", ".join(map(str, pages)) + ")"):
                for hit in hits:
                    st.caption(f"{page_reference(hit.record)} · {hit.record.heading}")
                    st.text(hit.record.text)
    st.session_state.messages.append({"role": "assistant", "content": answer})

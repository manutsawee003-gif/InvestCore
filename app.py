from pathlib import Path

import streamlit as st

from pdf_knowledge import page_reference
from rag_service import RAGService


PDF_DIRECTORY = Path(__file__).resolve().parent
if not (PDF_DIRECTORY / "Datasetหุ้น.pdf").exists():
    PDF_DIRECTORY = PDF_DIRECTORY.parent

st.set_page_config(page_title="InvestCore", page_icon="📈")
st.title("InvestCore")
st.caption("fix 30/9/26")
st.caption("แชตบอตความรู้จากคู่มือ SET สองเล่ม โดยค้นจาก Markdown ครบ 134 หน้า")
st.info("เลขหน้าอ้างอิงใช้หน้า PDF รวม 1–134 พร้อมเลขหน้าในไฟล์ต้นฉบับ เพื่อเปิดตรวจได้ตรงกัน")


@st.cache_resource
def get_service() -> RAGService:
    return RAGService(PDF_DIRECTORY)


WELCOME = "สวัสดีครับ ผม InvestCore ช่วยตอบคำถามจากคู่มือทั้ง 2 เล่มได้ ลองถามเรื่องหุ้น ETF, DR, DCA, การวิเคราะห์หุ้น หรือการจัดพอร์ตได้ครับ"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

with st.sidebar:
    st.write("แหล่งความรู้: Markdown ที่ถอดจาก PDF 028 และ 062")
    if st.button("เริ่มบทสนทนาใหม่"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("พิมพ์คำถามเกี่ยวกับหุ้นและการลงทุน"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นข้อมูลจากคู่มือ..."):
            answer, hits, _ = get_service().answer(question, st.session_state.messages[-7:-1])
        st.write(answer)
        if hits:
            with st.expander("ดูข้อความต้นฉบับจากคู่มือ"):
                st.caption(f"{hits[0].record.source} — {page_reference(hits[0].record)}")
                st.code(hits[0].record.answer, language=None)
    st.session_state.messages.append({"role": "assistant", "content": answer})

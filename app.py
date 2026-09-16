from pathlib import Path

import streamlit as st

from rag_service import RAGService

DATASET = Path(__file__).resolve().parent / "data" / "Dataset_หุ้นพื้นฐาน.xlsx"

st.set_page_config(page_title="InvestCore", page_icon="📈")
st.title("InvestCore")
st.caption("แชตบอตให้ความรู้พื้นฐานด้านหุ้นและตลาดทุน — เริ่มต้นเข้าใจหุ้น จากแก่นความรู้ที่ถูกต้อง")
st.info("ตอบเฉพาะข้อมูลจาก Dataset_หุ้นพื้นฐาน.xlsx และไม่ใช่คำแนะนำซื้อขายหลักทรัพย์")

@st.cache_resource
def get_service() -> RAGService:
    return RAGService(DATASET)

WELCOME = "สวัสดีค่ะ ฉันคือ InvestCore สอบถามความรู้พื้นฐานด้านหุ้นและตลาดทุนได้เลยค่ะ"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

with st.sidebar:
    st.write("แหล่งข้อมูล: Excel 800 Q&A")
    if st.button("เริ่มบทสนทนาใหม่"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("พิมพ์คำถามเกี่ยวกับหุ้นและตลาดทุน"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นข้อมูลในชุดข้อมูล..."):
            answer, hits, route = get_service().answer(question)
        st.write(answer)
        if hits:
            with st.expander("รายละเอียดการค้นคืน (ตรวจสอบได้)"):
                st.dataframe([{"ID": h.record.record_id, "คำถามในชุดข้อมูล": h.record.question, "dense": round(h.dense_score, 3)} for h in hits], hide_index=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})

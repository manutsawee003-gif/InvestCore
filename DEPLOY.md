# Deploy InvestCore

โปรเจกต์นี้พร้อมนำขึ้น Streamlit Community Cloud แล้ว

1. อัปโหลดโฟลเดอร์นี้ขึ้น GitHub โดยไม่อัปโหลด `.env`
2. ตรวจว่ามี `app.py`, `requirements.txt` และ `data/Dataset_หุ้นพื้นฐาน_1200_QA.xlsx`
3. เปิด `https://share.streamlit.io` แล้วเลือก **Create app**
4. เลือก repository, branch และไฟล์ `app.py`
5. เปิด App settings > Secrets แล้ววางค่าใน `STREAMLIT_SECRETS.example.toml` โดยเปลี่ยน API key เป็นค่าจริง
6. Deploy แล้วแชร์ URL ที่ลงท้ายด้วย `.streamlit.app`

โค้ดใช้ path แบบ relative และอ่านค่า Typhoon จาก Streamlit Secrets หรือ `.env` ได้ทั้งสองแบบ

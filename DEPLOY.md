# Deploy InvestCore บน Streamlit Community Cloud

โปรเจกต์นี้ใช้ `app.py` เป็นหน้าเว็บ, `data/knowledge.md` เป็นฐานความรู้ (สร้างจาก `Datasetหุ้น.pdf` ด้วย `build_knowledge.py`) ไฟล์นี้ต้องอยู่ใน repository

## 1. ส่งโค้ดล่าสุดขึ้น GitHub

Repository นี้ตั้ง `origin` ไปที่ `https://github.com/manutsawee003-gif/InvestCore.git` และใช้ branch `main` อยู่แล้ว ตรวจไฟล์ที่จะส่งก่อนเสมอ:

```powershell
git status
git add app.py build_knowledge.py evaluate.py pdf_knowledge.py rag_service.py retriever.py requirements.txt tests data/knowledge.md data/ocr_pages DEPLOY.md README.md STREAMLIT_SECRETS.example.toml .env.example
git commit -m "Rebuild knowledge base with Typhoon OCR and new retrieval"
git push origin main
```

ถ้ามีไฟล์อื่นที่ตั้งใจแก้เพิ่มเติม ให้ตรวจและเพิ่มเอง `.env` และ `.streamlit/secrets.toml` ถูก ignore และต้องไม่ส่งขึ้น GitHub

## 2. สร้างแอป

1. เปิด https://share.streamlit.io และลงชื่อเข้าใช้ด้วย GitHub ที่มีสิทธิ์ใน repository
2. กด **Create app** แล้วเลือก **Yup, I have an app**
3. เลือก repository `manutsawee003-gif/InvestCore`, branch `main`, entrypoint `app.py`
4. ใน **Advanced settings** เลือก Python 3.12 แล้วใส่ Secrets ตามด้านล่าง
5. กด **Deploy** และรอจนแอปเปิดที่ URL `.streamlit.app`

## 3. ตั้ง Secrets

วางข้อความนี้ในช่อง **Secrets** ของ Streamlit Cloud โดยแทน API key ด้วยค่าจริงจาก Typhoon เท่านั้น:

```toml
TYPHOON_API_KEY = "YOUR_REAL_KEY"
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"
```

หาก deploy ไปแล้ว ให้เปิด App settings > Secrets เพื่อเพิ่มหรือแก้ค่าได้ ไม่ต้องใส่ key ใน GitHub

## 4. ตรวจหลัง deploy

ลองถาม `ตลาดหุ้นคืออะไร`, `mindset`, `DCA คืออะไร` และ `ETF ต่างจาก DR อย่างไร` คำตอบควรมีเลขหน้าอ้างอิง และกด "ดูข้อความต้นฉบับจากคู่มือ" เพื่อตรวจได้ ตรวจ Cloud logs หากแอปไม่เริ่มหรือไม่ตอบ

การแก้ไฟล์ในเครื่องยังไม่ขึ้นเว็บจนกว่าจะ commit และ push ไปที่ branch ที่ deploy ไว้

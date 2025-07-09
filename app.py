import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# โหลด API KEY จากไฟล์ .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ตรวจสอบว่าโหลด API KEY ได้หรือไม่
if not api_key:
    st.error("❌ ไม่พบ API KEY กรุณาตรวจสอบไฟล์ .env")
    st.stop()

# ตั้งค่าโมเดล
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")  # เปลี่ยนชื่อให้ตรงกับเวอร์ชันที่ใช้งาน

# ฟังก์ชันถามหมอ
def ask_doctor(question):
    prompt = f"""
    คุณคือแพทย์ผู้เชี่ยวชาญ ให้คำแนะนำด้านสุขภาพอย่างชัดเจน สุภาพ และเข้าใจง่าย
    โปรดตอบคำถามของผู้ป่วยในเชิงให้คำแนะนำเบื้องต้น (ไม่ใช่การวินิจฉัย)

    คำถาม: {question}

    คำตอบ:
    """
    try:
        response = model.generate_content(prompt)
        return response.text + "\n\n⚠️ ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่แทนคำวินิจฉัยจากแพทย์จริง"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {e}"

# UI
st.set_page_config(page_title="ปรึกษาหมอ AI", page_icon="🩺")
st.title("🩺 ปรึกษาหมอ AI")
user_question = st.text_input("อาการหรือคำถามของคุณ", placeh

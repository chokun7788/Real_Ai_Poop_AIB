import streamlit as st
import google.generativeai as genai

# ตั้งค่า API Key
genai.configure(api_key="AIzaSyAeopIvwWG1SGwPCemZehjf2RY1pU2ItXk")

# ใช้ชื่อโมเดลให้ถูกต้อง (ณ ตอนนี้โมเดลล่าสุดคือ gemini-1.5-pro)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# ฟังก์ชันให้คำปรึกษาแพทย์
def ask_doctor(question):
    prompt = f"""
    คุณคือแพทย์ผู้เชี่ยวชาญ ให้คำแนะนำสุขภาพทั่วไปที่สุภาพ เข้าใจง่าย
    คำถามจากผู้ป่วย: {question}
    คำแนะนำจากคุณหมอ:
    """
    try:
        response = model.generate_content([prompt])
        return response.text + "\n\n⚠️ ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่แทนคำวินิจฉัยจากแพทย์จริง"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {e}"

# Streamlit UI
st.set_page_config(page_title="หมอ AI", page_icon="🩺")
st.title("🩺 ปรึกษาหมอ AI")
st.markdown("พิมพ์อาการของคุณ แล้วหมอ AI จะให้คำแนะนำเบื้องต้น")

question = st.text_input("อาการหรือคำถามของคุณ", placeholder="ตัวอย่าง: ไอแห้ง เจ็บคอ ทำไงดี")

if st.button("สอบถามหมอ AI"):
    if question.strip():
        with st.spinner("⏳ กำลังให้คำปรึกษา..."):
            answer = ask_doctor(question)
            st.success(answer)
    else:
        st.warning("⚠️ กรุณากรอกคำถามก่อนนะครับ")

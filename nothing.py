import streamlit as st
import google.generativeai as genai
from PIL import Image
import random

# --- 1. ตั้งค่าหน้าเว็บ (ต้องเป็นคำสั่งแรกสุด) ---
st.set_page_config(page_title="AI วิเคราะห์อุจจาระ", page_icon="🔬")


# --- 2. การตั้งค่า API Key (วิธีทดลองแบบใส่คีย์โดยตรง) ---
# ⚠️ **สำคัญ:** ให้นำ API Key ของคุณมาใส่ในเครื่องหมายคำพูดด้านล่างนี้
# ⚠️ **คำเตือน:** ห้ามอัปโหลดไฟล์ที่มีคีย์นี้ขึ้น GitHub หรือ Deploy เด็ดขาด!
try:
    GOOGLE_API_KEY = "AIzaSyBzBDxh15q8aR_I_YVAJR8ZnOX-7dAtvwo" 
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "ใส่_API_KEY_ของคุณตรงนี้เลย":
        st.error("กรุณาใส่ Gemini API Key ของคุณในโค้ดบรรทัดที่ 16")
        api_key_configured = False
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        api_key_configured = True

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการตั้งค่า API Key: {e}")
    api_key_configured = False


# --- ส่วนจำลองการทำงานของโมเดล Image Classification ---
# *** ในอนาคตคุณจะต้องนำโมเดลจริงๆ ของคุณมาใส่แทนที่ฟังก์ชันนี้ ***
def classify_image_mock(image):
    """ฟังก์ชันจำลองการจำแนกประเภทภาพ (ตอนนี้จะสุ่มผลลัพธ์เพื่อการสาธิต)"""
    classes = ["Blood", "Normal", "Yellow", "Green", "Diarrhea", "Mucus"]
    mock_result = random.choice(classes)
    return mock_result


# --- ส่วนเรียกใช้งาน Gemini API ---
def get_gemini_explanation(stool_class):
    """ส่งชื่อคลาสที่ได้ไปให้ Gemini อธิบายเพิ่มเติม"""
    class_map = {
        "Blood": "มีเลือดปน (Blood)",
        "Normal": "ปกติ (Normal)",
        "Yellow": "สีเหลือง (Yellow)",
        "Green": "สีเขียว (Green)",
        "Diarrhea": "ท้องร่วง/ท้องเสีย (Diarrhea)",
        "Mucus": "มีมูกปน (Mucus)"
    }
    friendly_name = class_map.get(stool_class, stool_class)
    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านสุขภาพที่ให้ข้อมูลเบื้องต้นที่เป็นประโยชน์

    จากผลการจำแนกประเภทของอุจจาระว่าเป็นแบบ "{friendly_name}"

    กรุณาให้ข้อมูลโดยละเอียดเป็นภาษาไทย โดยแบ่งหัวข้อให้ชัดเจนดังนี้:

    1.  **สาเหตุที่เป็นไปได้:** (อธิบายสาเหตุที่อาจทำให้เกิดอุจจาระลักษณะนี้ มีอะไรบ้าง)
    2.  **ความเสี่ยงหรือโรคที่อาจเกี่ยวข้อง:** (ระบุโรคหรือภาวะผิดปกติที่อาจเป็นสัญญาณเตือน)
    3.  **คำแนะนำเบื้องต้นและการดูแลตัวเอง:** (ควรทำอย่างไร เช่น การปรับเปลี่ยนอาหาร การดื่มน้ำ หรือเมื่อไหร่ที่ควรไปพบแพทย์)

    **ข้อความสำคัญ:** โปรดเน้นย้ำในตอนท้ายว่าข้อมูลนี้เป็นเพียงคำแนะนำเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ผู้เชี่ยวชาญได้
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเรียก Gemini API: {e}"


# --- ส่วนหน้าเว็บ Streamlit (UI) ---
st.title("🔬 AI ช่วยวิเคราะห์สุขภาพจากอุจจาระ")
st.write("อัปโหลดภาพอุจจาระของคุณเพื่อให้ AI ช่วยวิเคราะห์และให้คำแนะนำเบื้องต้น")

st.warning(
    "⚠️ **ข้อควรระวัง:** ผลลัพธ์จาก AI นี้เป็นเพียงข้อมูลเบื้องต้นเพื่อการศึกษาเท่านั้น "
    "**ไม่สามารถใช้แทนคำวินิจฉัยจากแพทย์ได้** หากมีอาการผิดปกติหรือกังวลใจ กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ"
)

# จะแสดงส่วนอัปโหลดไฟล์ก็ต่อเมื่อตั้งค่า API Key สำเร็จแล้วเท่านั้น
if api_key_configured:
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)
        
        if st.button("🚀 เริ่มการวิเคราะห์"):
            with st.spinner("กำลังวิเคราะห์ภาพและสอบถามผู้เชี่ยวชาญ AI... กรุณารอสักครู่"):
                classification_result = classify_image_mock(image)
                explanation = get_gemini_explanation(classification_result)
                
                st.success(f"✅ วิเคราะห์เสร็จสิ้น! ผลลัพธ์คือ: **{classification_result}**")
                st.markdown("---")
                st.subheader("คำอธิบายเพิ่มเติมจาก AI:")
                st.markdown(explanation)
# ถ้าตั้งค่าคีย์ไม่สำเร็จ จะแสดงข้อความนี้
else:
    st.info("กรุณาตั้งค่า Gemini API Key ในโค้ดให้ถูกต้องเพื่อเริ่มใช้งานแอปพลิเคชัน")
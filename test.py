import streamlit as st
from fastai.vision.all import *
from pathlib import Path
import pathlib # สำหรับ PosixPath patch
import sys
from PIL import Image
import pandas as pd
import plotly.express as px
import requests # เพิ่ม requests สำหรับดาวน์โหลดไฟล์
import os # เพิ่ม os สำหรับการจัดการไฟล์/โฟลเดอร์ (ถ้าจำเป็น)

# --- Monkey patch สำหรับ PosixPath บน Windows ---
# (ส่วนนี้อาจจะไม่จำเป็นบน Streamlit Cloud ซึ่งเป็น Linux แต่ใส่ไว้ก็ไม่เสียหาย)
_original_posix_path = None 
if sys.platform == "win32": # Patch นี้จะทำงานเฉพาะเมื่อรันบน Windows (เครื่อง Local ของคุณ)
    if hasattr(pathlib, 'PosixPath') and not isinstance(pathlib.PosixPath, pathlib.WindowsPath):
        _original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        # print("INFO: PosixPath patch applied.") # เอา print ออกได้เมื่อ deploy จริง
# --- สิ้นสุดส่วนโค้ด Patch ---

# --- !!! กำหนดข้อมูลโมเดลของคุณที่นี่ !!! ---
MODEL_FILENAME = "convnextv2_thev1_best_for_good.pkl" # <--- ชื่อไฟล์โมเดล .pkl ของคุณ
# !!! คุณต้องหา DIRECT DOWNLOAD LINK ของไฟล์ .pkl จาก Cloud Storage ของคุณมาใส่ตรงนี้ !!!
MODEL_DOWNLOAD_URL = "YOUR_DIRECT_DOWNLOAD_LINK_TO_THE_MODEL_PKL_FILE_HERE"  # <--- ***** แก้ไข URL นี้ *****
# ตัวอย่าง URL (สมมติ) จาก Dropbox: "https://www.dropbox.com/s/xxxxxxxxx/your_model.pkl?dl=1"
# ตัวอย่าง URL (สมมติ) จาก GitHub Release asset: "https://github.com/YourUser/YourRepo/releases/download/v1.0/your_model.pkl"

MODEL_LOCAL_PATH = Path(MODEL_FILENAME) # กำหนดว่าจะเซฟไฟล์โมเดลที่ดาวน์โหลดมาด้วยชื่ออะไรใน container ของ Streamlit
# --------------------------------------------------------------------

@st.cache_resource # โหลดโมเดลแค่ครั้งเดียว
def download_and_load_model(model_url, local_path, filename):
    # 1. ดาวน์โหลดไฟล์โมเดลถ้ายังไม่มีอยู่ในเครื่อง (container ของ Streamlit Cloud)
    if not local_path.is_file():
        st.info(f"Model file '{filename}' not found locally. Attempting to download from URL...")
        if model_url == "YOUR_DIRECT_DOWNLOAD_LINK_TO_THE_MODEL_PKL_FILE_HERE" or not model_url:
            st.error("Model download URL is not configured. Please set MODEL_DOWNLOAD_URL in the script.")
            return None
        try:
            with st.spinner(f"Downloading {filename}... (This may take a moment for large files)"):
                response = requests.get(model_url, stream=True)
                response.raise_for_status() # ตรวจสอบว่า request สำเร็จหรือไม่ (ถ้าไม่สำเร็จจะ raise error)
                
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192*16): # ดาวน์โหลดทีละส่วน
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                print(f"Model downloaded to {local_path}")
                st.success(f"Model '{filename}' downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            if os.path.exists(local_path): # ถ้าดาวน์โหลดไม่สำเร็จ ให้ลบไฟล์ที่อาจจะเสียทิ้ง
                os.remove(local_path)
            return None # คืนค่า None ถ้าดาวน์โหลดไม่สำเร็จ
    else:
        print(f"Model file '{filename}' found locally at {local_path}.")
    
    # 2. โหลดโมเดล (ไม่ว่าจะดาวน์โหลดมาใหม่ หรือมีอยู่แล้ว)
    if local_path.is_file(): # ตรวจสอบอีกครั้งว่าไฟล์มีอยู่จริงหลังจากการดาวน์โหลด (ถ้ามี)
        print(f"Attempting to load model from: {local_path}")
        try:
            learn = load_learner(local_path)
            print("Model loaded successfully!")
            return learn # คืนค่า learner object ที่โหลดแล้ว
        except Exception as e:
            st.error(f"Error loading model from {local_path}: {e}")
            # ให้ข้อมูลเพิ่มเติมที่เป็นประโยชน์สำหรับการ debug
            st.info("Ensure the model file is a valid FastAI learner export (.pkl) and all necessary dependencies "
                    "(like 'timm', 'cloudpickle', 'fasttransform' if used by your model) "
                    "are listed in your requirements.txt file.")
            return None # คืนค่า None ถ้าโหลดโมเดลไม่สำเร็จ
    return None # ถ้าไฟล์ยังคงไม่มีอยู่

# เรียกใช้ฟังก์ชันเพื่อดาวน์โหลด (ถ้าจำเป็น) และโหลดโมเดล
# ผลลัพธ์ (learner object หรือ None) จะถูกเก็บไว้ในตัวแปร learn
learn = download_and_load_model(MODEL_DOWNLOAD_URL, MODEL_LOCAL_PATH, MODEL_FILENAME)

# --- ส่วนของหน้าตา Streamlit App (เหมือนเดิม) ---
st.title("💩 Poop Classification")

if learn is None: # ถ้าโมเดลโหลดไม่สำเร็จ (learn เป็น None)
    st.error("Critical Error: AI Model could not be loaded. The application cannot continue.")
    st.warning("Please check the deployment logs on Streamlit Cloud for more details, or contact the app administrator.")
    st.stop() # และหยุดการทำงานของ script ที่เหลือทั้งหมด ไม่ต้องแสดง UI ส่วนอื่นต่อ

st.subheader("Upload your Poop Image")
uploaded_file = st.file_uploader(
    "Click This For Upload Image", 
    type=["jpg", "jpeg", "png"], 
    help="Limit 200MB per file"
)

if uploaded_file is not None:
    image_to_display = Image.open(uploaded_file)
    
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("##### This is your Image")
        st.image(image_to_display, use_column_width=True)

        if st.button("🚽 Click For Predict", use_container_width=True):
            with st.spinner("Predicting..."):
                img_bytes = uploaded_file.getvalue()
                try:
                    pil_image = PILImage.create(img_bytes)
                except Exception as e_pil:
                    st.error(f"Error converting uploaded file to an image: {e_pil}")
                    if 'prediction_made' in st.session_state: del st.session_state.prediction_made 
                    st.stop()
                
                try:
                    pred_class, pred_idx, probs = learn.predict(pil_image) 
                    st.markdown("---")
                    st.markdown(f"#### Result is : **{pred_class}**")
                    st.markdown(f"##### {pred_class} : **{probs[pred_idx]:.1%}**")
                
                    st.session_state.prediction_made = True
                    st.session_state.predicted_class = pred_class
                    st.session_state.probabilities = probs.numpy() 
                    st.session_state.class_names = list(learn.dls.vocab)
                except Exception as e_predict:
                    st.error(f"Error during prediction: {e_predict}")
                    if 'prediction_made' in st.session_state: del st.session_state.prediction_made
    
    if 'prediction_made' in st.session_state and st.session_state.prediction_made:
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Probabilities Chart")
            
            df_probs = pd.DataFrame({
                'Class': st.session_state.class_names,
                'Probability': st.session_state.probabilities * 100
            })
            
            fig = px.pie(df_probs, values='Probability', names='Class', 
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+name')
            fig.update_layout(showlegend=True, legend_title_text='Classes')
            
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("App by Chokun7788 (with AI assistant)")
import streamlit as st
import sys
sys.path.append(r'D:\Final_Project\streamlit_UI')
import image_prediction
from PIL import Image
from load_model import load_model_xmt

model, device = load_model_xmt()

# Cập nhật hiển thị trong app
def app():
    st.write("Upload a Picture to Classify as Benign or Malignant Tumor.")
    file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])
    
    if file_uploaded is not None:
        image, processed_image = image_prediction.process_and_save_image(file_uploaded, model, device)
        
        # Tạo hai cột để hiển thị hình ảnh và kết quả
        c1, buff, c2 = st.columns([2, 0.5, 2])
        
        # Cột bên trái hiển thị hình ảnh
        c1.image(image, width=500)
        
        # Cột bên phải hiển thị kết quả phân loại
        c2.subheader("Classification Result")
        
        # Màu đỏ cho 'Malignant' và màu xanh cho 'Benign'
        if processed_image == "Malignant":
            result_text = f'<span style="color:red;">The image is classified as **{processed_image}**.</span>'
        else:
            result_text = f'<span style="color:green;">The image is classified as **{processed_image}**.</span>'
        
        # Hiển thị kết quả với màu sắc tùy thuộc vào phân loại
        c2.markdown(result_text, unsafe_allow_html=True)





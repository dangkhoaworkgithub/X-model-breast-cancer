import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F
import sys
sys.path.append(r'D:\Final_Project\streamlit_UI')
from load_model import load_model_xmt
from PIL import ImageFont, ImageDraw
from xmodel import XMT

model, device = load_model_xmt()

# Hàm xử lý và dự đoán ảnh
def process_and_save_image(image_path, model, device):
    # Đọc ảnh bằng plt.imread
    image = plt.imread(image_path)

    # Tiền xử lý ảnh: chuyển đổi ảnh thành tensor và chuẩn hóa
    normalize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = normalize_transform(np.array(image)).unsqueeze(0).to(device)

    # Thực hiện dự đoán
    with torch.no_grad():  # Không cần gradient khi dự đoán
        prediction = model(image_tensor)
        _, predicted_class = torch.max(prediction, 1)  # Lấy lớp dự đoán
        pred_label = predicted_class.item()

    # Lớp 1 là "Malignant" và lớp 0 là "Benign"
    label = "Malignant" if pred_label == 1 else "Benign"
    
    return image, label
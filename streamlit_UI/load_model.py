import cv2
import torch
from torchvision import transforms
from xmodel import XMT
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F


# Hàm tải mô hình XMT và trọng số
def load_model_xmt(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Định nghĩa mô hình XMT (mô hình dự đoán ung thư)
    model = XMT(image_size=224, patch_size=7, num_classes=2, channels=512, dim=1024, depth=6, heads=8, mlp_dim=2048)
    model.to(device)

    # Tải checkpoint chứa trọng số và các thông tin khác
    checkpoint = torch.load(r"D:\Final_Project\x-model\weight\xmodel_deepfake_sample_1.pth", map_location=device)

    # Chỉ lấy phần 'state_dict' chứa trọng số của mô hình
    state_dict = checkpoint['state_dict']

    # Nạp trọng số vào mô hình
    model.load_state_dict(state_dict)

    # Đặt mô hình ở chế độ đánh giá (evaluation mode)
    model.eval()
    print("Model loaded successfully.")
    return model, device
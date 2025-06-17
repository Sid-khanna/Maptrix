import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models import stackhourglass  # Use correct architecture for PSMNet

# === Parameters ===
DATAPATH = r"C:\Users\Siddh\Desktop\Maptrix\images\testing"
LEFT_DIR = os.path.join(DATAPATH, "image_2")
RIGHT_DIR = os.path.join(DATAPATH, "image_3")
OUTPUT_DIR = "results/disparities"
CKPT_PATH = r"C:\Users\Siddh\Desktop\Maptrix\new_code\pretrained_model_KITTI2015.tar"
MAX_DISP = 192

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model ===
def load_model():
    print("🔄 Loading model...")
    model = stackhourglass(maxdisp=MAX_DISP)
    model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(CKPT_PATH)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.eval()
    print("✅ Model loaded successfully.")
    return model

# === Image pre-processing ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_pair(left_path, right_path):
    left_img = Image.open(left_path).convert('RGB')
    right_img = Image.open(right_path).convert('RGB')

    imgL = transform(left_img)
    imgR = transform(right_img)

    _, h, w = imgL.shape
    pad_h = 16 - (h % 16) if h % 16 != 0 else 0
    pad_w = 16 - (w % 16) if w % 16 != 0 else 0

    # Pad right and bottom edges
    imgL = nn.functional.pad(imgL, (0, pad_w, 0, pad_h)).unsqueeze(0).cuda()
    imgR = nn.functional.pad(imgR, (0, pad_w, 0, pad_h)).unsqueeze(0).cuda()

    return imgL, imgR, h, w  # Return original size too

# === Save disparity as color image and .npy ===
def save_disparity(disp, name):
    disp_img_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    disp_npy_path = os.path.join(OUTPUT_DIR, f"{name}.npy")

    np.save(disp_npy_path, disp)
    plt.imsave(disp_img_path, disp, cmap='plasma')

# === Main inference loop ===
if __name__ == "__main__":
    model = load_model()
    image_files = sorted(os.listdir(LEFT_DIR))

    for i, fname in enumerate(image_files):
        left_path = os.path.join(LEFT_DIR, fname)
        right_path = os.path.join(RIGHT_DIR, fname)

        print(f"[{i+1}/{len(image_files)}] Processing: {fname}")
        imgL, imgR, h, w = load_image_pair(left_path, right_path)

        with torch.no_grad():
            output = model(imgL, imgR)
            disparity = output.squeeze().cpu().numpy()
            disparity = disparity[:h, :w]  # Crop to original size

        base_name = os.path.splitext(fname)[0]
        save_disparity(disparity, base_name)

    print("🏁 All done. Disparity maps saved to:", OUTPUT_DIR)

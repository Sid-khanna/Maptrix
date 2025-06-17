import cv2
import os

# === Parameters ===
IMAGE_DIR = r"C:\Users\Siddh\Desktop\Maptrix\new_code\disparities"
VIDEO_PATH = "disparity_video.mp4"
FPS = 5  # frames per second

# === Gather image paths ===
images = sorted([img for img in os.listdir(IMAGE_DIR) if img.endswith(".png")])
if not images:
    raise ValueError("No PNG images found in the directory!")

# === Read first image to get size ===
first_img = cv2.imread(os.path.join(IMAGE_DIR, images[0]))
height, width, _ = first_img.shape

# === Define video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (width, height))

# === Write each image to video ===
for img_name in images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    out.write(img)

out.release()
print(f"✅ Video saved to: {VIDEO_PATH}")

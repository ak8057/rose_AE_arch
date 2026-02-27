# ==========================================================
# LAPTOP CLIENT - ROSEv2 LATENT → HF SERVER
# ==========================================================

import torch
import torch.nn as nn
import requests
import numpy as np
import time
import cv2
from PIL import Image
from torchvision import transforms
import base64
import zlib
# ==========================================================
# CONFIG
# ==========================================================

SERVER_URL = "https://abhaykumar8057-mobile-vit-wae.hf.space/predict"
API_KEY = "SECRET123"

DEVICE = "cpu"
IMG_SIZE = 224
LATENT_DIM = 128
ENCODER_PATH = "models/rose_v2_encoder.pth"   # local encoder

import torch
import torch.nn as nn
import requests
import numpy as np
import time
import cv2
import csv  # Added for logging
import os   # Added for directory management
from PIL import Image
from torchvision import transforms

# ... [Keep your DWConv, ResidualBlock, and ROSEv2_Encoder classes exactly as they are] ...

# ==========================================================
# LOGGING CONFIG
# ==========================================================
LOG_FILE = "latent_logs.csv"

def log_latent_vector(latent_np, source_type):
    """Logs latent vector stats and timestamp to a CSV file."""
    file_exists = os.path.isfile(LOG_FILE)
    
    # Calculate basic stats to make the log readable
    stats = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_type,
        "mean": np.mean(latent_np),
        "std": np.std(latent_np),
        "min": np.min(latent_np),
        "max": np.max(latent_np),
        "shape": str(latent_np.shape)
    }

    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)
    
    # Optional: Save the actual raw vector as a .npy file for deep analysis
    # np.save(f"logs/latent_{int(time.time())}.npy", latent_np)
    print(f"📝 Latent logged to {LOG_FILE}")

# ==========================================================
# ROSEv2 ARCHITECTURE (MUST MATCH TRAINING)
# ==========================================================

class DWConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            DWConv(ch, ch),
            DWConv(ch, ch)
        )

    def forward(self, x):
        return x + self.block(x)

class ROSEv2_Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.res1 = ResidualBlock(32)
        self.enc2 = DWConv(32, 64, stride=2)
        self.res2 = ResidualBlock(64)
        self.enc3 = DWConv(64, latent_dim, stride=2)
        self.res3 = ResidualBlock(latent_dim)

    def forward(self, x):
        x = self.res1(self.enc1(x))
        x = self.res2(self.enc2(x))
        return self.res3(self.enc3(x))

# ==========================================================
# LOAD ENCODER
# ==========================================================

print("Loading ROSEv2 encoder...")
encoder = ROSEv2_Encoder(LATENT_DIM)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
encoder.eval()
print("Encoder ready.")

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==========================================================
# HELPER FUNCTION: Encode + Send
# ==========================================================

def process_and_send(pil_image, source_name="Unknown"):
    img_tensor = transform(pil_image).unsqueeze(0)
    start_total = time.time()

    with torch.no_grad():
        latent = encoder(img_tensor)

    end_encode = time.time()
    latent_np = latent.numpy().astype(np.float16)

    # --- NEW LOGGING FEATURE ---
    log_latent_vector(latent_np, source_name)
    # ---------------------------

    raw_bytes = latent_np.tobytes()
    compressed = zlib.compress(raw_bytes)
    encoded = base64.b64encode(compressed).decode("utf-8")

    payload = {
        "shape": list(latent_np.shape),
        "data": encoded
    }

    upload_kb = len(payload["data"]) / 2 / 1024
    headers = {"api-key": API_KEY}

    try:
        response = requests.post(SERVER_URL, json=payload, headers=headers)
        end_total = time.time()

        network_time = end_total - end_encode
        total_time = end_total - start_total
        bandwidth = upload_kb / network_time if network_time > 0 else 0

        print("\n==============================")
        print("📦 Upload Size:", f"{upload_kb:.2f} KB")
        print("🧠 Encoding:", f"{(end_encode-start_total)*1000:.2f} ms")
        print("🌐 Network+Server:", f"{network_time*1000:.2f} ms")
        print("⏱ Total:", f"{total_time*1000:.2f} ms")
        print("🚀 Bandwidth:", f"{bandwidth:.2f} KB/s")

        if response.status_code == 200:
            r = response.json()

            if "class" in r:
                print("✅ Prediction:", r["class"])
                print("🎯 Confidence:", r["confidence"])
            else:
                print("❌ Server error:", r)
        else:
            print("❌ HTTP Error:", response.text)

    except Exception as e:
        print("❌ Connection failed:", e)


# ==========================================================
# START CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

print("\n📷 LIVE CLIENT STARTED")
print("Press 'c' → capture webcam")
print("Press 'u' → upload image")
print("Press 'q' → quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("ROSEv2 Client", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Webcam Capture
    if key == ord('c'):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print("\n📷 Captured from webcam")
        process_and_send(img)

    # Upload Image Mode
    if key == ord('u'):
        print("\n📂 Enter image path:")
        path = input("Path: ").strip()

        try:
            img = Image.open(path).convert("RGB")
            print("📂 Loaded image:", path)
            process_and_send(img)
        except Exception as e:
            print("❌ Failed to load image:", e)

cap.release()
cv2.destroyAllWindows()

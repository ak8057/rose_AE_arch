# ==========================================================
# CLIENT - RAW HTTP (EXPERIMENT VERSION)
# ==========================================================

import torch
import torch.nn as nn
import requests
import numpy as np
import time
from PIL import Image
from torchvision import transforms
import zlib

# ==========================================================
# CONFIG
# ==========================================================

SERVER_URL = "https://abhaykumar8057-mobile-vit-wae-rawhttp.hf.space/predict"
API_KEY = "SECRET123"

DEVICE = "cpu"
IMG_SIZE = 224
LATENT_DIM = 128
ENCODER_PATH = "models/rose_v2_encoder.pth"

# ==========================================================
# MODEL ARCHITECTURE
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
# LOAD MODEL
# ==========================================================

print("Loading encoder...")
encoder = ROSEv2_Encoder(LATENT_DIM)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
encoder.eval()
print("✅ Encoder ready")

# ==========================================================
# TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==========================================================
# CORE FUNCTION
# ==========================================================

def process_and_send(pil_image):

    img_tensor = transform(pil_image).unsqueeze(0)
    total_start = time.time()

    # -------- ENCODE --------
    with torch.no_grad():
        latent = encoder(img_tensor)

    encode_end = time.time()

    # -------- FLOAT16 --------
    latent_np = latent.numpy().astype(np.float16)

    # -------- SERIALIZE + COMPRESS --------
    raw_bytes = latent_np.tobytes()
    compressed = zlib.compress(raw_bytes)

    headers = {
        "api-key": API_KEY,
        "shape": str(list(latent_np.shape)),
        "dtype": "float16"
    }

    upload_kb = len(compressed) / 1024

    try:
        net_start = time.time()

        response = requests.post(
            SERVER_URL,
            data=compressed,   # 🔥 RAW BYTES
            headers=headers,
            timeout=20
        )

        net_end = time.time()
        total_end = time.time()

        encode_time = encode_end - total_start
        network_time = net_end - net_start
        total_time = total_end - total_start

        if response.status_code == 200:
            r = response.json()

            print("\n==============================")
            print(f"📦 Payload: {upload_kb:.2f} KB")
            print(f"🧠 Encode: {encode_time*1000:.2f} ms")
            print(f"🌐 Network: {network_time*1000:.2f} ms")
            print(f"⏱ Total: {total_time*1000:.2f} ms")
            print(f"🎯 Confidence: {r.get('confidence')}")
            print("==============================")

            return {
                "payload": upload_kb,
                "encode_time": encode_time,
                "network_time": network_time,
                "total_time": total_time,
            }

    except Exception as e:
        print("❌ Error:", e)
        return None

# ==========================================================
# EXPERIMENT RUNNER
# ==========================================================

def run_experiment(image_path, runs=10):

    img = Image.open(image_path).convert("RGB")
    results = []

    print(f"\n🚀 Running {runs} trials...\n")

    for i in range(runs):
        print(f"Run {i+1}/{runs}")

        res = process_and_send(img)

        if res:
            results.append(res)

    if len(results) == 0:
        print("❌ No successful runs")
        return

    # -------- AVERAGES --------
    avg_payload = np.mean([r["payload"] for r in results])
    avg_encode = np.mean([r["encode_time"] for r in results])
    avg_network = np.mean([r["network_time"] for r in results])
    avg_total = np.mean([r["total_time"] for r in results])

    # -------- STD --------
    std_network = np.std([r["network_time"] for r in results])
    std_total = np.std([r["total_time"] for r in results])

    print("\n==============================")
    print("📊 FINAL AVERAGE RESULTS (RAW HTTP)")
    print(f"📦 Payload: {avg_payload:.2f} KB")
    print(f"🧠 Encode: {avg_encode*1000:.2f} ms")
    print(f"🌐 Network: {avg_network*1000:.2f} ms ± {std_network*1000:.2f}")
    print(f"⏱ Total: {avg_total*1000:.2f} ms ± {std_total*1000:.2f}")
    print("==============================")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    run_experiment("test.jpg", runs=5)


#  # ==========================================================
# # LAPTOP CLIENT - ROSEv2 LATENT → RAW BINARY HTTP SERVER
# # ==========================================================

# import torch
# import torch.nn as nn
# import requests
# import numpy as np
# import time
# import cv2
# import csv
# import os
# from PIL import Image
# from torchvision import transforms
# import zlib

# # ==========================================================
# # CONFIG
# # ==========================================================

# SERVER_URL = "https://abhaykumar8057-mobile-vit-wae-rawhttp.hf.space/predict"   # CHANGE THIS
# API_KEY = "SECRET123"

# DEVICE = "cpu"
# IMG_SIZE = 224
# LATENT_DIM = 128
# ENCODER_PATH = "models/rose_v2_encoder.pth"

# LOG_FILE = "latent_logs.csv"

# # ==========================================================
# # LOGGING
# # ==========================================================

# def log_latent_vector(latent_np, source_type):
#     file_exists = os.path.isfile(LOG_FILE)

#     stats = {
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "source": source_type,
#         "mean": np.mean(latent_np),
#         "std": np.std(latent_np),
#         "min": np.min(latent_np),
#         "max": np.max(latent_np),
#         "shape": str(latent_np.shape)
#     }

#     with open(LOG_FILE, mode='a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=stats.keys())
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(stats)

#     print(f"📝 Latent logged to {LOG_FILE}")

# # ==========================================================
# # MODEL ARCHITECTURE
# # ==========================================================

# class DWConv(nn.Module):
#     def __init__(self, in_c, out_c, stride=1):
#         super().__init__()
#         self.dw = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
#         self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
#         self.bn = nn.BatchNorm2d(out_c)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.act(self.bn(self.pw(self.dw(x))))

# class ResidualBlock(nn.Module):
#     def __init__(self, ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             DWConv(ch, ch),
#             DWConv(ch, ch)
#         )

#     def forward(self, x):
#         return x + self.block(x)

# class ROSEv2_Encoder(nn.Module):
#     def __init__(self, latent_dim=128):
#         super().__init__()
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(3, 32, 3, 2, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
#         self.res1 = ResidualBlock(32)
#         self.enc2 = DWConv(32, 64, stride=2)
#         self.res2 = ResidualBlock(64)
#         self.enc3 = DWConv(64, latent_dim, stride=2)
#         self.res3 = ResidualBlock(latent_dim)

#     def forward(self, x):
#         x = self.res1(self.enc1(x))
#         x = self.res2(self.enc2(x))
#         return self.res3(self.enc3(x))

# # ==========================================================
# # LOAD MODEL
# # ==========================================================

# print("Loading encoder...")
# encoder = ROSEv2_Encoder(LATENT_DIM)
# encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
# encoder.eval()
# print("✅ Encoder ready")

# # ==========================================================
# # TRANSFORM
# # ==========================================================

# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor()
# ])

# # ==========================================================
# # CORE FUNCTION
# # ==========================================================

# def process_and_send(pil_image, source_name="Unknown"):

#     img_tensor = transform(pil_image).unsqueeze(0)
#     total_start = time.time()

#     # -------- ENCODING --------
#     with torch.no_grad():
#         latent = encoder(img_tensor)

#     encode_end = time.time()

#     # -------- FLOAT16 --------
#     latent_np = latent.numpy().astype(np.float16)

#     # -------- LOG --------
#     log_latent_vector(latent_np, source_name)

#     # -------- SERIALIZATION --------
#     raw_bytes = latent_np.tobytes()

#     # -------- COMPRESSION --------
#     compressed = zlib.compress(raw_bytes)

#     # -------- HEADERS --------
#     headers = {
#         "api-key": API_KEY,
#         "shape": str(list(latent_np.shape)),   # send shape as string
#         "dtype": "float16"
#     }

#     upload_kb = len(compressed) / 1024

#     # -------- NETWORK --------
#     try:
#         net_start = time.time()

#         response = requests.post(
#             SERVER_URL,
#             data=compressed,   # 🔥 RAW BYTES HERE
#             headers=headers,
#             timeout=20
#         )

#         net_end = time.time()
#         total_end = time.time()

#         # -------- METRICS --------
#         encode_time = encode_end - total_start
#         network_time = net_end - net_start
#         total_time = total_end - total_start

#         bandwidth = upload_kb / (network_time + 1e-9)

#         print("\n==============================")
#         print(f"📦 Upload Size: {upload_kb:.2f} KB")
#         print(f"🧠 Encoding: {encode_time*1000:.2f} ms")
#         print(f"🌐 Network+Server: {network_time*1000:.2f} ms")
#         print(f"⏱ Total: {total_time*1000:.2f} ms")
#         print(f"🚀 Bandwidth: {bandwidth:.2f} KB/s")

#         # -------- RESPONSE --------
#         if response.status_code == 200:
#             r = response.json()
#             print("✅ Prediction:", r.get("class"))
#             print("🎯 Confidence:", r.get("confidence"))
#         else:
#             print("❌ HTTP Error:", response.text)

#         print("==============================")

#     except Exception as e:
#         print("❌ Connection failed:", e)

# # ==========================================================
# # CAMERA LOOP
# # ==========================================================

# cap = cv2.VideoCapture(0)

# print("\n📷 RAW BINARY CLIENT STARTED")
# print("Press 'c' → capture")
# print("Press 'u' → upload image")
# print("Press 'q' → quit\n")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cv2.imshow("ROSEv2 Client", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('q'):
#         break

#     if key == ord('c'):
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         print("\n📷 Captured")
#         process_and_send(img, "webcam")

#     if key == ord('u'):
#         path = input("\nEnter image path: ").strip()
#         try:
#             img = Image.open(path).convert("RGB")
#             process_and_send(img, "upload")
#         except Exception as e:
#             print("❌ Failed:", e)

# cap.release()
# cv2.destroyAllWindows()
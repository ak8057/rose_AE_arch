# # ==========================================================
# # ROSEv2 DECODER + MOBILEVIT SERVER
# # ==========================================================

# import torch
# import torch.nn as nn
# import numpy as np
# import timm
# from fastapi import FastAPI, Header

# # ==========================================================
# # CONFIG
# # ==========================================================

# DEVICE = "cpu"  # HF free tier
# LATENT_DIM = 128
# API_KEY = "SECRET123"

# CLASS_NAMES = ['infectious', 'organic', 'recyclable', 'sharps']

# DECODER_PATH = "models/rose_v2_decoder.pth"
# CLASSIFIER_PATH = "models/mobilevit_compressed_best.pth"

# # ==========================================================
# # ROSE BLOCKS
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


# class ROSEv2_Decoder(nn.Module):
#     def __init__(self, latent_dim=128):
#         super().__init__()
#         self.up1 = nn.ConvTranspose2d(latent_dim, 64, 3, 2, 1, 1)
#         self.res1 = ResidualBlock(64)
#         self.up2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
#         self.res2 = ResidualBlock(32)
#         self.up3 = nn.ConvTranspose2d(32, 3, 3, 2, 1, 1)

#     def forward(self, z):
#         x = self.res1(self.up1(z))
#         x = self.res2(self.up2(x))
#         return torch.sigmoid(self.up3(x))


# # ==========================================================
# # LOAD MODELS
# # ==========================================================

# print("🚀 Starting ROSE + MobileViT Server")

# decoder = ROSEv2_Decoder(LATENT_DIM).to(DEVICE)
# decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
# decoder.eval()

# classifier = timm.create_model(
#     "mobilevit_s",
#     pretrained=False,
#     num_classes=len(CLASS_NAMES)
# ).to(DEVICE)

# classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
# classifier.eval()

# print("✅ Models loaded")
# print("Running on:", DEVICE)

# # ==========================================================
# # FASTAPI
# # ==========================================================

# app = FastAPI()


# @app.get("/")
# def home():
#     return {"status": "ROSEv2 + MobileViT server running"}


# # ==========================================================
# # PREDICT
# # ==========================================================

# @app.post("/predict")
# def predict(data: dict, api_key: str = Header(None)):

#     if api_key != API_KEY:
#         return {"error": "Unauthorized"}

#     try:
#         # ----------------------------------
#         # Deserialize latent
#         # ----------------------------------
#         z = np.frombuffer(bytes.fromhex(data["data"]), dtype=np.float32)
#         z = z.reshape(data["shape"])
#         z = torch.tensor(z).to(DEVICE)

#         with torch.no_grad():

#             # ----------------------------------
#             # Decode
#             # ----------------------------------
#             img = decoder(z)
#             img = torch.clamp(img, 0, 1)

#             # ----------------------------------
#             # IMPORTANT: SAME AS TRAINING
#             # ----------------------------------
#             mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
#             std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)

#             img = (img - mean) / std

#             # ----------------------------------
#             # Classify
#             # ----------------------------------
#             logits = classifier(img)
#             probs = torch.softmax(logits, dim=1)
#             pred_idx = probs.argmax(1).item()

#         result = {
#             "class": CLASS_NAMES[pred_idx],
#             "confidence": float(probs[0, pred_idx])
#         }

#         print("Prediction:", result)
#         return result

#     except Exception as e:
#         print("ERROR:", e)
#         return {"error": str(e)}


# # ==========================================================
# # IMPORTANT FOR HF
# # ==========================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=7860)
# ==========================================================
# ROSEv2 DECODER + MOBILEVIT SERVER
# ==========================================================

import torch
import torch.nn as nn
import numpy as np
import timm
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
import base64

import zlib

# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "cpu"  # HF free tier
LATENT_DIM = 128
API_KEY = "SECRET123"

CLASS_NAMES = ['infectious', 'organic', 'recyclable', 'sharps']

DECODER_PATH = "models/rose_v2_decoder.pth"
CLASSIFIER_PATH = "models/mobilevit_compressed_best.pth"


app = FastAPI()

# ADD THIS SECTION:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your laptop IP to connect
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep your @app.post("/predict") exactly as it is!

# ==========================================================
# ROSEv2 ARCHITECTURE (same as training)
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


class ROSEv2_Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(latent_dim, 64, 3, 2, 1, 1)
        self.res1 = ResidualBlock(64)
        self.up2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.res2 = ResidualBlock(32)
        self.up3 = nn.ConvTranspose2d(32, 3, 3, 2, 1, 1)

    def forward(self, z):
        x = self.res1(self.up1(z))
        x = self.res2(self.up2(x))
        return torch.sigmoid(self.up3(x))


# ==========================================================
# LOAD MODELS
# ==========================================================

print("🚀 Starting ROSEv2 + MobileViT Server")

decoder = ROSEv2_Decoder(LATENT_DIM).to(DEVICE)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
decoder.eval()

classifier = timm.create_model(
    "mobilevit_s",
    pretrained=False,
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier.eval()

print("✅ Models loaded")
print("🔥 Running on:", DEVICE)

# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI()


@app.get("/")
def home():
    return {"status": "ROSEv2 + MobileViT server running"}


# ==========================================================
# PREDICTION
# ==========================================================

@app.post("/predict")
def predict(data: dict, api_key: str = Header(None)):

    if api_key != API_KEY:
        return {"error": "Unauthorized"}

    try:
        # -------------------------
        # Deserialize latent
        # -------------------------
        compressed = base64.b64decode(data["data"])

        # Decompress
        raw_bytes = zlib.decompress(compressed)

        # Read as float16
        z = np.frombuffer(raw_bytes, dtype=np.float16)
        
        # Convert back to float32 for model
        z = z.astype(np.float32)
        
        z = z.reshape(data["shape"])
        z = torch.tensor(z).to(DEVICE)

        # -------------------------
        # Decode
        # -------------------------
        with torch.no_grad():
            img = decoder(z)
            img = torch.clamp(img, 0, 1)

            # 🚨 IMPORTANT
            # Training used ONLY ToTensor()
            # so DO NOT normalize
            logits = classifier(img)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()

        # -------------------------
        # DEBUG PRINTS
        # -------------------------
        print("\n==============================")
        print("Image mean:", img.mean().item())
        print("Image std :", img.std().item())
        print("Probabilities:", probs.cpu().numpy())
        print("Prediction:", CLASS_NAMES[pred_idx])
        print("==============================\n")

        return {
            "class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[0, pred_idx])
        }

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}


# ==========================================================
# IMPORTANT FOR CLOUD
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

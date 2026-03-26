# ==========================================================
# FASTAPI SERVER - RAW BINARY (ROSEv2 + MobileViT)
# ==========================================================

import torch
import torch.nn as nn
import numpy as np
import timm
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
import zlib
import ast

# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "cpu"
LATENT_DIM = 128
API_KEY = "SECRET123"

CLASS_NAMES = ['infectious', 'organic', 'recyclable', 'sharps']

DECODER_PATH = "models/rose_v2_decoder.pth"
CLASSIFIER_PATH = "models/mobilevit_compressed_best.pth"

# ==========================================================
# FASTAPI INIT
# ==========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

print("🚀 Starting RAW BINARY Server")

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
# ROUTES
# ==========================================================

@app.get("/")
def home():
    return {"status": "RAW Binary ROSEv2 server running"}

# ==========================================================
# PREDICT ENDPOINT
# ==========================================================

@app.post("/predict")
async def predict(request: Request, api_key: str = Header(None)):

    if api_key != API_KEY:
        return {"error": "Unauthorized"}

    try:
        # -------------------------
        # READ RAW BINARY DATA
        # -------------------------
        compressed = await request.body()

        # -------------------------
        # DECOMPRESS
        # -------------------------
        raw_bytes = zlib.decompress(compressed)

        # -------------------------
        # READ HEADERS
        # -------------------------
        shape_str = request.headers.get("shape")
        dtype_str = request.headers.get("dtype", "float16")

        shape = ast.literal_eval(shape_str)

        if dtype_str == "float16":
            dtype = np.float16
        else:
            dtype = np.float32

        # -------------------------
        # RECONSTRUCT LATENT
        # -------------------------
        z = np.frombuffer(raw_bytes, dtype=dtype)
        z = z.astype(np.float32)
        z = z.reshape(shape)
        z = torch.tensor(z).to(DEVICE)

        # -------------------------
        # DECODE + CLASSIFY
        # -------------------------
        with torch.no_grad():
            img = decoder(z)
            img = torch.clamp(img, 0, 1)

            logits = classifier(img)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()

        # -------------------------
        # DEBUG
        # -------------------------
        print("\n==============================")
        print("Shape:", shape)
        print("Dtype:", dtype_str)
        print("Image mean:", img.mean().item())
        print("Image std :", img.std().item())
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
# RUN
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

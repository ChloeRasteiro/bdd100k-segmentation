from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from app.model import predict, NUM_CLASSES, CLASS_NAMES

app = FastAPI(
    title="BDD100K Segmentation API",
    description="Semantic segmentation — U-Net + EfficientNet-B3, 19 classes",
    version="1.0"
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "UNet + EfficientNet-B3",
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES
    }

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    segmented = predict(image)

    buf = io.BytesIO()
    segmented.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@app.get("/health")
def health():
    return {"status": "healthy"}


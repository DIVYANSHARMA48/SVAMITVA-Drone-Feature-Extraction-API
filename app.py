from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np
import cv2
from PIL import Image
import io
from utils import load_model, classify_roof, device
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = FastAPI(title="SVAMITVA Drone Feature Extraction API")

MODEL_PATH = "model.pth"
model = load_model(MODEL_PATH)

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

@app.get("/")
async def root():
    return {"message": "Welcome to the SVAMITVA Drone Feature Extraction API. Use /predict to analyze drone images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Preprocess
    augmented = transform(image=image_np)
    tensor = augmented['image'].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(tensor)
        mask = torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5
        mask = (mask * 255).astype(np.uint8)

    # Find building contours for roof classification
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roof_results = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:  # small noise ignore
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        building_patch = image_np[y:y+h, x:x+w]
        
        # Resize to CLIP size if needed
        if building_patch.shape[0] < 224 or building_patch.shape[1] < 224:
            building_patch = cv2.resize(building_patch, (224, 224))
        
        roof_type = classify_roof(building_patch)
        roof_results.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "roof_type": roof_type,
            "area_px": int(cv2.contourArea(cnt))
        })

    # Convert mask to base64 for easy frontend display
    _, buffer = cv2.imencode('.png', mask)
    mask_b64 = buffer.tobytes()

    return JSONResponse({
        "status": "success",
        "building_count": len(roof_results),
        "roofs": roof_results,
        "mask_base64": mask_b64.hex()   # frontend can decode
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
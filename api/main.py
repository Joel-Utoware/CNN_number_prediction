import shutil
import os
import base64
import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from PIL import Image

from model.load_model import get_best_model, predict_digit
app = FastAPI()
#open with uvicorn api.main:app --reload

# load model
model = get_best_model()

#temporary folder
upload_dir = Path("temp_uploads")
upload_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="api/templates")

@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    # 1️⃣ validate mime type
    if file.content_type != "image/png":
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "prediction": None,
                "error": "Only PNG files are allowed."
            }
        )

    # 2️⃣ validate extension
    if not file.filename.lower().endswith(".png"):
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "prediction": None,
                "error": "File must have a .png extension."
            }
        )

    temp_file_path = upload_dir / file.filename

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3️⃣ validate actual image
    try:
        with Image.open(temp_file_path) as img:
            img.verify()
    except Exception:
        os.remove(temp_file_path)
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "prediction": None,
                "error": "Invalid or corrupted PNG image."
            }
        )

    try:
        prediction, probs = predict_digit(temp_file_path, model)
        confidence = round(float(probs[prediction]) * 100, 2)

        image = Image.open(temp_file_path)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "prediction": prediction,
                "confidence": confidence,
                "image_base64": image_base64
            }
        )

    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)

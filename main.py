import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN verbose logs

import shutil
import json
from fastapi import FastAPI, UploadFile, Form, Request, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

app = FastAPI(title="MediScan AI - Medical Image Classification")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Directory to store uploaded images
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Directory to store models
MODEL_DIR = "."

# Load models with compile=False to handle cross-version compatibility
print("Loading models...")
model1 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "braintumor.h5"), compile=False)
model2 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "Tuberculosis_model.h5"), compile=False)
model3 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "pneumonia_model.h5"), compile=False)
print("All models loaded successfully!")

# Infer preprocessing sizes from model input shapes
MODEL1_SIZE = tuple(model1.input_shape[1:3])   # e.g. (224, 224)
MODEL2_SIZE = tuple(model2.input_shape[1:3])   # e.g. (224, 224)
MODEL3_SIZE = tuple(model3.input_shape[1:3])   # e.g. (224, 224)
MODEL1_CHANNELS = model1.input_shape[3]         # 3 for RGB, 1 for grayscale
MODEL2_CHANNELS = model2.input_shape[3]
MODEL3_CHANNELS = model3.input_shape[3]

print(f"Brain Tumor Model: {model1.input_shape}")
print(f"Tuberculosis Model: {model2.input_shape}")
print(f"Pneumonia Model:    {model3.input_shape}")

# Model metadata
MODEL_INFO = {
    1: {
        "name": "Brain Tumor Detection",
        "type": "MRI/CT Scan",
        "icon": "brain",
        "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"],
        "description": "Analyzes MRI/CT scans to detect and classify brain tumors into four categories."
    },
    2: {
        "name": "Tuberculosis Detection",
        "type": "Chest X-Ray",
        "icon": "lungs",
        "classes": ["Normal", "Tuberculosis"],
        "description": "Analyzes chest X-rays to detect signs of tuberculosis infection."
    },
    3: {
        "name": "Pneumonia Detection",
        "type": "Chest X-Ray",
        "icon": "lungs",
        "classes": ["Normal", "Pneumonia"],
        "description": "Analyzes chest X-rays to detect pneumonia patterns."
    }
}

# Result descriptions and recommendations
RESULT_INFO = {
    "glioma": {
        "severity": "high",
        "color": "danger",
        "description": "Glioma is a type of tumor that occurs in the brain and spinal cord. It begins in glial cells that surround nerve cells.",
        "recommendation": "Immediate consultation with a neurosurgeon and oncologist is strongly recommended. Further imaging (advanced MRI with contrast) and biopsy may be needed."
    },
    "meningioma": {
        "severity": "medium",
        "color": "warning",
        "description": "Meningioma is a tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most are benign.",
        "recommendation": "Consult a neurologist or neurosurgeon. Regular monitoring with follow-up MRI scans is typically recommended."
    },
    "no_tumor": {
        "severity": "low",
        "color": "success",
        "description": "No tumor detected in the brain scan. The neural tissue appears within normal radiological parameters.",
        "recommendation": "No immediate action required. Continue with routine health check-ups as advised by your physician."
    },
    "pituitary": {
        "severity": "medium",
        "color": "warning",
        "description": "A pituitary tumor (adenoma) is an abnormal growth in the pituitary gland. Most are benign and grow slowly.",
        "recommendation": "Consult an endocrinologist and neurosurgeon. Hormone level testing and follow-up MRI are typically recommended."
    },
    "Tuberculosis": {
        "severity": "high",
        "color": "danger",
        "description": "Signs consistent with tuberculosis (TB) detected in the chest X-ray. TB is a serious bacterial infection primarily affecting the lungs.",
        "recommendation": "Immediate consultation with a pulmonologist is necessary. Sputum culture and sensitivity tests should be performed. Begin isolation protocols and notify public health authorities."
    },
    "Normal": {
        "severity": "low",
        "color": "success",
        "description": "No significant pathological findings detected. The scan appears within normal radiological parameters.",
        "recommendation": "No immediate action required. Continue with routine health check-ups as advised by your physician."
    },
    "Pneumonia": {
        "severity": "high",
        "color": "danger",
        "description": "Signs of pneumonia detected in the chest X-ray. Pneumonia is an infection that inflames the air sacs in one or both lungs.",
        "recommendation": "Consult a pulmonologist or general physician immediately. Antibiotic therapy may be required. Rest and adequate hydration are essential."
    }
}


def save_uploaded_file(file, destination):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


def preprocess_image(image_path, target_size=(224, 224), color_mode='rgb'):
    img = keras_image.load_img(image_path, target_size=target_size, color_mode=color_mode)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})


@app.post("/api/predict")
async def predict(
    choice: int = Form(...),
    data: UploadFile = File(...)
):
    try:
        file_location = os.path.join(UPLOAD_DIR, "upload_temp.jpg")
        save_uploaded_file(data, file_location)

        confidence = 0.0
        result = ""
        all_predictions = []

        color_mode1 = 'rgb' if MODEL1_CHANNELS == 3 else 'grayscale'
        color_mode2 = 'rgb' if MODEL2_CHANNELS == 3 else 'grayscale'
        color_mode3 = 'rgb' if MODEL3_CHANNELS == 3 else 'grayscale'

        if choice == 1:
            # Brain Tumor model
            img = preprocess_image(file_location, target_size=MODEL1_SIZE, color_mode=color_mode1)
            predictions = model1.predict(img)
            predicted_class = int(np.argmax(predictions))
            confidence = float(np.max(predictions)) * 100
            class_labels = ["glioma", "meningioma", "no_tumor", "pituitary"]
            display_labels = MODEL_INFO[1]["classes"]
            result = class_labels[predicted_class]
            all_predictions = [
                {"label": display_labels[i], "confidence": float(predictions[0][i]) * 100}
                for i in range(len(class_labels))
            ]

        elif choice == 2:
            # Tuberculosis model
            img = preprocess_image(file_location, target_size=MODEL2_SIZE, color_mode=color_mode2)
            prediction = model2.predict(img)
            # Handle both binary sigmoid output and softmax
            pred_flat = prediction.flatten()
            if len(pred_flat) == 1:
                raw_val = float(pred_flat[0])
                if raw_val >= 0.5:
                    result = "Tuberculosis"
                    confidence = raw_val * 100
                else:
                    result = "Normal"
                    confidence = (1 - raw_val) * 100
                all_predictions = [
                    {"label": "Normal", "confidence": (1 - raw_val) * 100},
                    {"label": "Tuberculosis", "confidence": raw_val * 100}
                ]
            else:
                predicted_class = int(np.argmax(pred_flat))
                confidence = float(np.max(pred_flat)) * 100
                labels = ["Normal", "Tuberculosis"]
                result = labels[min(predicted_class, len(labels)-1)]
                all_predictions = [
                    {"label": labels[i], "confidence": float(pred_flat[i]) * 100}
                    for i in range(len(pred_flat))
                ]

        elif choice == 3:
            # Pneumonia model
            img = preprocess_image(file_location, target_size=MODEL3_SIZE, color_mode=color_mode3)
            prediction = model3.predict(img)
            pred_flat = prediction.flatten()
            if len(pred_flat) == 1:
                raw_val = float(pred_flat[0])
                if raw_val >= 0.5:
                    result = "Pneumonia"
                    confidence = raw_val * 100
                else:
                    result = "Normal"
                    confidence = (1 - raw_val) * 100
                all_predictions = [
                    {"label": "Normal", "confidence": (1 - raw_val) * 100},
                    {"label": "Pneumonia", "confidence": raw_val * 100}
                ]
            else:
                predicted_class = int(np.argmax(pred_flat))
                confidence = float(np.max(pred_flat)) * 100
                labels = ["Normal", "Pneumonia"]
                result = labels[min(predicted_class, len(labels)-1)]
                all_predictions = [
                    {"label": labels[i], "confidence": float(pred_flat[i]) * 100}
                    for i in range(len(pred_flat))
                ]

        else:
            return JSONResponse({"error": "Invalid choice"}, status_code=400)

        result_details = RESULT_INFO.get(result, {
            "severity": "unknown",
            "color": "info",
            "description": "Analysis complete.",
            "recommendation": "Please consult a medical professional."
        })

        return JSONResponse({
            "success": True,
            "result": result,
            "display_result": result.replace("_", " ").title(),
            "confidence": round(confidence, 2),
            "model_info": MODEL_INFO[choice],
            "all_predictions": all_predictions,
            "severity": result_details["severity"],
            "color": result_details["color"],
            "description": result_details["description"],
            "recommendation": result_details["recommendation"]
        })

    except Exception as e:
        return JSONResponse({"error": str(e), "success": False}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

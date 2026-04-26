from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import shutil
import os
import uuid
import requests
import cv2
import numpy as np

import cloudinary
import cloudinary.uploader

from ai_makeup import generate_ai_makeup
from overlay_makeup import blend_makeup


app = FastAPI()

# -----------------------------
# CORS (React Native allow)
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Cloudinary configuration
# -----------------------------

cloudinary.config(
    cloud_name="dpljhbbf6",
    api_key="557913527713992",
    api_secret="DyYSpFUYciF_hhJll4YJwC-deIU"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# --------------------------------
# FACE VALIDATION FUNCTION
# --------------------------------

def validate_face_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False, "Invalid image"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    img_h, img_w = image.shape[:2]
    image_area = img_w * img_h

    valid_faces = []

    # Filter small false faces
    for (x, y, w, h) in faces:

        face_area = w * h

        if face_area / image_area > 0.05:
            valid_faces.append((x, y, w, h))

    print("Detected faces:", len(valid_faces))

    # -----------------------------
    # No face
    # -----------------------------

    if len(valid_faces) == 0:
        return False, "No face detected"

    # -----------------------------
    # Multiple real faces
    # -----------------------------

    if len(valid_faces) >= 2:
        return False, "Multiple faces detected"

    (x, y, w, h) = valid_faces[0]

    face_ratio = (w * h) / image_area

    # -----------------------------
    # Face too far
    # -----------------------------

    if face_ratio < 0.12:
        return False, "Move closer to camera"

    # -----------------------------
    # Face center check
    # -----------------------------

    face_center_x = x + w / 2
    face_center_y = y + h / 2

    if not (img_w * 0.2 < face_center_x < img_w * 0.8):
        return False, "Center your face in frame"

    if not (img_h * 0.2 < face_center_y < img_h * 0.8):
        return False, "Adjust camera position"

    # -----------------------------
    # Blur detection
    # -----------------------------

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    print("Blur score:", blur)

    if blur < 20:
        return False, "Image is blurry"

    return True, "Valid face"


# --------------------------------
# VALIDATE FACE API
# --------------------------------

@app.post("/validate-face")
async def validate_face(file: UploadFile = File(...)):

    temp_name = f"temp_{uuid.uuid4()}.jpg"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_name)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    valid, message = validate_face_image(temp_path)

    os.remove(temp_path)

    return {
        "valid": valid,
        "message": message
    }


# --------------------------------
# UPLOAD IMAGES
# --------------------------------

@app.post("/upload")
async def upload_images(
    source: UploadFile = File(...),
    reference: UploadFile = File(...)
):

    unique_id = str(uuid.uuid4())

    source_filename = f"source_{unique_id}.jpg"
    reference_filename = f"reference_{unique_id}.jpg"

    source_path = os.path.join(UPLOAD_FOLDER, source_filename)
    reference_path = os.path.join(UPLOAD_FOLDER, reference_filename)

    with open(source_path, "wb") as buffer:
        shutil.copyfileobj(source.file, buffer)

    with open(reference_path, "wb") as buffer:
        shutil.copyfileobj(reference.file, buffer)

    return {
        "source": source_filename,
        "reference": reference_filename
    }


# --------------------------------
# APPLY MAKEUP
# --------------------------------

@app.post("/apply-makeup")
async def apply_makeup(source: str, reference: str):

    source_path = os.path.join(UPLOAD_FOLDER, source)
    reference_path = os.path.join(UPLOAD_FOLDER, reference)

    # Run AI model
    ai_url = generate_ai_makeup(source_path)

    ai_filename = f"ai_{uuid.uuid4()}.jpg"
    ai_path = os.path.join(UPLOAD_FOLDER, ai_filename)

    img_data = requests.get(ai_url).content

    with open(ai_path, "wb") as handler:
        handler.write(img_data)

    # Blend AI + reference
    final_path = blend_makeup(source_path, ai_path, reference_path)

    # Upload to Cloudinary
    upload_result = cloudinary.uploader.upload(final_path)

    cloud_url = upload_result["secure_url"]

    return {
        "result_image": cloud_url
    }
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
from app.services.preprocessing import preprocess_with_visualization

router = APIRouter(prefix="/preprocess", tags=["Preprocessing"])


def _encode_image(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


@router.post("")
def preprocess_image_endpoint(file: UploadFile = File(...)):
    """
    Accept a face image and return all preprocessing stages as base64 images.
    Stages: original -> grayscale -> resized -> Gaussian blur -> histogram equalization -> normalized.
    """
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    stages = preprocess_with_visualization(image)
    result = {}
    for stage_name, stage_image in stages.items():
        if len(stage_image.shape) == 2:
            stage_image_bgr = cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)
        else:
            stage_image_bgr = stage_image
        result[stage_name] = _encode_image(stage_image_bgr)

    return {
        "stages": list(stages.keys()),
        "images": result,
        "description": {
            "original": "Raw input image from camera",
            "grayscale": "Converted from BGR to grayscale to reduce computational complexity",
            "resized": "Scaled to 160x160 pixels for consistent processing",
            "gaussian_blur": "Applied Gaussian blur (5x5 kernel) to reduce noise",
            "equalized": "Histogram equalization improves contrast under poor lighting",
            "normalized": "Pixel values scaled to [0,1] range for numerical stability",
        },
    }

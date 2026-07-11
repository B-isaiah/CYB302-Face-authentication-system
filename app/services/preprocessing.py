import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a face image for feature extraction.
    Pipeline: grayscale -> resize(160x160) -> Gaussian blur -> histogram equalization -> normalize.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (160, 160))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    normalized = equalized / 255.0
    return normalized


def preprocess_with_visualization(image: np.ndarray) -> dict:
    """
    Preprocess and return each stage as a separate image for visualization.
    Returns dict with keys: original, grayscale, resized, gaussian_blur, equalized, normalized.
    """
    stages = {}
    stages["original"] = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stages["grayscale"] = gray
    resized = cv2.resize(gray, (160, 160))
    stages["resized"] = resized
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    stages["gaussian_blur"] = blurred
    equalized = cv2.equalizeHist(blurred)
    stages["equalized"] = equalized
    normalized = equalized / 255.0
    stages["normalized"] = (normalized * 255).astype(np.uint8)
    return stages

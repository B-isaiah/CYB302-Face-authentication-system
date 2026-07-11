import os
import numpy as np
import dlib
import cv2
from app.config import MODELS_DIR


SHAPE_PREDICTOR_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL_PATH = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

_detector = None
_sp = None
_facerec = None


def _get_models():
    """Lazy-load dlib models (face detector, shape predictor, face recognizer)."""
    global _detector, _sp, _facerec
    if _detector is None:
        _detector = dlib.get_frontal_face_detector()
    if _sp is None:
        if os.path.exists(SHAPE_PREDICTOR_PATH):
            _sp = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    if _facerec is None:
        if os.path.exists(FACE_REC_MODEL_PATH):
            _facerec = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
    return _detector, _sp, _facerec


def download_models():
    """Download dlib model files from dlib.net if not already present."""
    import urllib.request
    base = "http://dlib.net/files"
    files = [
        ("shape_predictor_68_face_landmarks.dat", f"{base}/shape_predictor_68_face_landmarks.dat.bz2", True),
        ("dlib_face_recognition_resnet_model_v1.dat", f"{base}/dlib_face_recognition_resnet_model_v1.dat.bz2", True),
    ]
    for name, url, compressed in files:
        dest = os.path.join(MODELS_DIR, name)
        if os.path.exists(dest):
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, dest + ".tmp")
        if compressed:
            import bz2
            with bz2.BZ2File(dest + ".tmp") as f:
                data = f.read()
            with open(dest, "wb") as f:
                f.write(data)
            os.remove(dest + ".tmp")
        else:
            os.rename(dest + ".tmp", dest)
        print(f"Downloaded {name}")


def extract_embedding(image: np.ndarray) -> np.ndarray | None:
    """
    Detect a face in the image and extract a 128D face embedding using dlib.
    Returns None if no face is detected.
    Falls back to HOG features if dlib models are not loaded.
    """
    detector, sp, facerec = _get_models()
    if sp is None or facerec is None:
        return _fallback_embedding(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, 1)
    if len(faces) == 0:
        return None
    shape = sp(rgb, faces[0])
    embedding = np.array(facerec.compute_face_descriptor(rgb, shape))
    return embedding


def _fallback_embedding(image: np.ndarray) -> np.ndarray:
    """Fallback: extract HOG features if dlib recognition model unavailable."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    hog = cv2.HOGDescriptor()
    features = hog.compute(resized)
    return features.flatten()

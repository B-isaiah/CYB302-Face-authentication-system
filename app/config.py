import os
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR}/database/biometric.db"
ENROLLMENT_DIR = BASE_DIR / "images" / "enrollment"
TEST_DIR = BASE_DIR / "images" / "test"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SHAPE_PREDICTOR_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"

SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FACE_RECOGNITION_MODEL_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat"

if not SHAPE_PREDICTOR_PATH.exists():
    print("Downloading shape predictor model (~96MB)...")
    bz2_path = str(SHAPE_PREDICTOR_PATH) + ".bz2"
    urllib.request.urlretrieve(SHAPE_PREDICTOR_URL, bz2_path)
    import bz2
    with bz2.BZ2File(bz2_path, "rb") as src, open(SHAPE_PREDICTOR_PATH, "wb") as dst:
        dst.write(src.read())
    os.remove(bz2_path)
    print("Shape predictor model downloaded.")

if not FACE_RECOGNITION_MODEL_PATH.exists():
    print("Downloading face recognition model (~22MB)...")
    urllib.request.urlretrieve(FACE_RECOGNITION_MODEL_URL, FACE_RECOGNITION_MODEL_PATH)
    print("Face recognition model downloaded.")

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.92"))

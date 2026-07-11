import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import inspect, text
from app.database import engine, Base
from app.config import REPORTS_DIR, BASE_DIR
from app.routes import enroll, verify, metrics, preprocess, multimodal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

inspector = inspect(engine)
columns = [c["name"] for c in inspector.get_columns("users")]
if "hog_feature" not in columns:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN hog_feature BLOB"))
        conn.commit()
    logger.info("Added hog_feature column to users table")

app = FastAPI(
    title="Face Recognition Authentication System",
    description="CYB302 - Biometrics Security Lab Project\n\nA university access control system that authenticates staff using facial recognition instead of passwords or ID cards.",
    version="1.0.0",
    contact={
        "name": "CYB302 Student",
        "email": "student@university.edu",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(enroll.router)
app.include_router(verify.router)
app.include_router(metrics.router)
app.include_router(preprocess.router)
app.include_router(multimodal.router)

os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

TEMPLATES_DIR = BASE_DIR / "templates"
CAPTURE_HTML = TEMPLATES_DIR / "capture.html"


@app.get("/capture", include_in_schema=False)
def capture_page():
    if CAPTURE_HTML.exists():
        return FileResponse(str(CAPTURE_HTML))
    return {"error": "capture.html not found"}


@app.get("/")
def root():
    """Root endpoint listing all available API endpoints."""
    return {
        "message": "Face Recognition Authentication System API",
        "project": "CYB302 - Biometrics Security Lab",
        "scenario": "A university replaces passwords with face recognition for staff entering secure research labs.",
        "endpoints": {
            "enroll_user": "POST /enroll - Enroll a new staff member (name + face image)",
            "list_users": "GET /enroll/users - List all enrolled users",
            "delete_user": "DELETE /enroll/users/{id} - Delete an enrolled user",
            "verify_user": "POST /verify - Verify a face against enrolled users",
            "get_metrics": "GET /metrics - Get FAR, FRR, EER for current threshold",
            "evaluate": "POST /metrics/evaluate - Full evaluation with plots",
            "evaluate_dataset": "POST /metrics/evaluate-dataset - Real evaluation using ORL dataset",
            "threshold_sweep": "GET /metrics/threshold-sweep - Compare thresholds",
            "preprocess": "POST /preprocess - View preprocessing stages of an image",
            "multimodal_verify": "POST /multimodal/verify - Multimodal (dlib+HOG) verification",
            "multimodal_evaluate": "POST /multimodal/evaluate - Compare unimodal vs multimodal performance",
            "capture_page": "GET /capture - Webcam capture page (browser)",
        },
        "lab_tasks": {
            "task_1": "Data Capture - POST /enroll",
            "task_2": "Image Preprocessing - POST /preprocess",
            "task_3": "Feature Extraction - 128D face embeddings via dlib",
            "task_4": "Matching - Cosine similarity / Euclidean distance",
            "task_5": "Threshold Selection - GET /metrics/threshold-sweep",
            "task_6": "Performance Evaluation - GET /metrics, POST /metrics/evaluate",
            "task_7": "Multimodal - POST /multimodal/evaluate (dlib + HOG score-level fusion)",
            "task_8": "Security - Encrypted templates (Fernet AES), README discussion",
        },
    }


@app.on_event("startup")
def startup_event():
    logger.info("Face Recognition Authentication System started")
    logger.info(f"Database: {engine.url}")
    logger.info(f"Reports directory: {REPORTS_DIR}")

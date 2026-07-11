import os
import uuid
from pathlib import Path
from app.config import ENROLLMENT_DIR


def save_upload_image(file_bytes: bytes, filename: str) -> str:
    """Save an uploaded image to the enrollment directory with a unique filename."""
    ext = os.path.splitext(filename)[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = ENROLLMENT_DIR / unique_name
    with open(filepath, "wb") as f:
        f.write(file_bytes)
    return str(filepath)

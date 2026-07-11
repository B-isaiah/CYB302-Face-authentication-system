import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.utils import save_upload_image
from app.services.feature_extraction import extract_embedding
from app.services.encryption import encrypt_embedding
from app.services.preprocessing import preprocess_image
from app.services.multimodal import extract_hog_features
from app.schemas import UserResponse, MessageResponse

router = APIRouter(prefix="/enroll", tags=["Enrollment"])


@router.post("", response_model=MessageResponse)
def enroll_user(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    contents = file.file.read()
    filepath = save_upload_image(contents, file.filename or "image.jpg")
    image = cv2.imread(filepath)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")
    embedding = extract_embedding(image)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in image")
    encrypted_face = encrypt_embedding(embedding)

    hog_feat = extract_hog_features(image)
    encrypted_hog = encrypt_embedding(hog_feat) if hog_feat is not None else None

    user = User(name=name, image_paths=filepath, face_embedding=encrypted_face, hog_feature=encrypted_hog)
    db.add(user)
    db.commit()
    return MessageResponse(message=f"User '{name}' enrolled successfully")


@router.get("/users", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()


@router.delete("/users/{user_id}", response_model=MessageResponse)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return MessageResponse(message=f"User '{user.name}' deleted")

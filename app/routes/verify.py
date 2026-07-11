import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.services.feature_extraction import extract_embedding
from app.services.encryption import decrypt_embedding
from app.services.matching import match_embeddings
from app.config import SIMILARITY_THRESHOLD
from app.schemas import VerifyResponse

router = APIRouter(prefix="/verify", tags=["Verification"])


@router.post("", response_model=VerifyResponse)
def verify_user(
    file: UploadFile = File(...),
    threshold: float = Query(SIMILARITY_THRESHOLD, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")
    probe_embedding = extract_embedding(image)
    if probe_embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in image")
    users = db.query(User).all()
    gallery = []
    for u in users:
        if u.face_embedding:
            try:
                emb = decrypt_embedding(u.face_embedding)
                gallery.append((u.id, u.name, emb))
            except Exception:
                continue
    if not gallery:
        raise HTTPException(status_code=400, detail="No enrolled users found")
    result = match_embeddings(probe_embedding, gallery, threshold=threshold, metric="cosine")
    if result is None:
        return VerifyResponse(identity=None, similarity=0.0, decision="Access Denied")
    uid, name, score = result
    return VerifyResponse(identity=name, similarity=round(score, 4), decision="Access Granted")

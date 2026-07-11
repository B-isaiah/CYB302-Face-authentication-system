import os
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.services.encryption import decrypt_embedding
from app.services.multimodal import (
    extract_multimodal_features,
    multimodal_match,
    extract_hog_features,
    hog_cosine_similarity,
    fuse_scores,
)
from app.services.feature_extraction import extract_embedding
from app.services.matching import cosine_similarity
from app.services.performance import compute_metrics, plot_roc_curve, plot_det_curve
from app.config import REPORTS_DIR
from app.datasets.orl import get_subjects

router = APIRouter(prefix="/multimodal", tags=["Multimodal"])


def _load_gallery(db: Session):
    users = db.query(User).all()
    gallery = []
    for u in users:
        if u.face_embedding and u.hog_feature:
            try:
                dlib_emb = decrypt_embedding(u.face_embedding)
                hog_feat = decrypt_embedding(u.hog_feature)
                gallery.append((u.id, u.name, dlib_emb, hog_feat))
            except Exception:
                continue
    return gallery


@router.post("/verify")
def multimodal_verify(
    file: UploadFile = File(...),
    threshold: float = Query(0.6, ge=0.0, le=1.0),
    weight: float = Query(0.7, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    probe = extract_multimodal_features(image)
    if probe["dlib"] is None:
        raise HTTPException(status_code=400, detail="No face detected")
    if probe["hog"] is None:
        raise HTTPException(status_code=400, detail="HOG extraction failed")

    gallery = _load_gallery(db)
    if not gallery:
        raise HTTPException(status_code=400, detail="No enrolled users with multimodal data")

    result = multimodal_match(probe, gallery, threshold=threshold, weight=weight)
    if result is None:
        return {
            "identity": None,
            "dlib_score": 0.0,
            "hog_score": 0.0,
            "fused_score": 0.0,
            "decision": "Access Denied",
        }
    uid, name, ds, hs, fs = result
    return {
        "identity": name,
        "dlib_score": ds,
        "hog_score": hs,
        "fused_score": fs,
        "weight": weight,
        "decision": "Access Granted",
    }


@router.post("/evaluate")
def multimodal_evaluate(
    threshold: float = Query(0.6, ge=0.0, le=1.0),
    weight: float = Query(0.7, ge=0.0, le=1.0),
    enroll_count: int = Query(5, ge=1, le=9),
):
    """Compare unimodal (dlib only) vs multimodal (dlib + HOG fused) performance on ORL dataset."""
    from app.database import SessionLocal

    subjects = get_subjects()
    if not subjects:
        raise HTTPException(400, detail="No ORL dataset found. Run scripts/bulk_enroll_orl.py first")

    db = SessionLocal()
    try:
        enrolled = db.query(User).all()
    finally:
        db.close()

    if not enrolled:
        raise HTTPException(400, detail="No enrolled users. Run bulk enrollment first")

    dlib_gallery = []
    hog_gallery = []
    for u in enrolled:
        try:
            dlib_emb = decrypt_embedding(u.face_embedding)
            hog_feat = decrypt_embedding(u.hog_feature) if u.hog_feature else None
            dlib_gallery.append((u.id, u.name, dlib_emb))
            if hog_feat is not None:
                hog_gallery.append((u.id, u.name, hog_feat))
        except Exception:
            continue

    if not hog_gallery:
        raise HTTPException(400, detail="No HOG features found. Re-run bulk enrollment to store them")

    scores_dlib_gen = []
    scores_dlib_imp = []
    scores_hog_gen = []
    scores_hog_imp = []
    scores_fused_gen = []
    scores_fused_imp = []

    for subj_name, images in subjects:
        test_images = images[enroll_count:]
        test_dlib = []
        test_hog = []
        test_subj_names = []

        for img_path in test_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            emb = extract_embedding(img)
            if emb is None:
                continue
            hog = extract_hog_features(img)
            test_dlib.append(emb)
            test_hog.append(hog)
            test_subj_names.append(subj_name)

        for i in range(len(test_dlib)):
            for uid, name, gemb in dlib_gallery:
                ds = cosine_similarity(test_dlib[i], gemb)
                if name == test_subj_names[i]:
                    scores_dlib_gen.append(ds)
                else:
                    scores_dlib_imp.append(ds)

            for uid, name, ghog in hog_gallery:
                hs = hog_cosine_similarity(test_hog[i], ghog)
                if name == test_subj_names[i]:
                    scores_hog_gen.append(hs)
                else:
                    scores_hog_imp.append(hs)

    if not scores_dlib_gen or not scores_dlib_imp:
        raise HTTPException(400, detail="Insufficient scores computed")

    dlib_metrics = compute_metrics(scores_dlib_gen, scores_dlib_imp, threshold)
    hog_metrics = compute_metrics(scores_hog_gen, scores_hog_imp, threshold)

    all_min = min(min(scores_dlib_gen + scores_dlib_imp + scores_hog_gen + scores_hog_imp), 0)
    all_max = max(scores_dlib_gen + scores_dlib_imp + scores_hog_gen + scores_hog_imp)

    def norm(s, mn, mx):
        return (s - mn) / (mx - mn) if mx != mn else 0.5

    for d, h in zip(scores_dlib_gen, scores_hog_gen):
        dn = norm(d, all_min, all_max)
        hn = norm(h, all_min, all_max)
        scores_fused_gen.append(fuse_scores(dn, hn, weight))

    for d, h in zip(scores_dlib_imp, scores_hog_imp):
        dn = norm(d, all_min, all_max)
        hn = norm(h, all_min, all_max)
        scores_fused_imp.append(fuse_scores(dn, hn, weight))

    fused_metrics = compute_metrics(scores_fused_gen, scores_fused_imp, threshold)

    roc_path = plot_roc_curve(scores_fused_gen, scores_fused_imp)
    det_path = plot_det_curve(scores_fused_gen, scores_fused_imp)

    return {
        "data_source": "ORL Dataset — Unimodal vs Multimodal Comparison",
        "enroll_count": enroll_count,
        "fusion_weight_dlib": weight,
        "fusion_weight_hog": round(1 - weight, 2),
        "unimodal_dlib": {
            "far": round(dlib_metrics["far"], 4),
            "frr": round(dlib_metrics["frr"], 4),
            "eer": round(dlib_metrics["eer"], 4),
            "auc": round(dlib_metrics["auc"], 4),
        },
        "unimodal_hog": {
            "far": round(hog_metrics["far"], 4),
            "frr": round(hog_metrics["frr"], 4),
            "eer": round(hog_metrics["eer"], 4),
            "auc": round(hog_metrics["auc"], 4),
        },
        "multimodal_fused": {
            "far": round(fused_metrics["far"], 4),
            "frr": round(fused_metrics["frr"], 4),
            "eer": round(fused_metrics["eer"], 4),
            "auc": round(fused_metrics["auc"], 4),
        },
        "roc_plot": os.path.basename(roc_path),
        "det_plot": os.path.basename(det_path),
    }

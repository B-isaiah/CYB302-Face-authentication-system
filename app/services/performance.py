import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from app.config import REPORTS_DIR
from app.database import SessionLocal
from app.models import User
from app.services.encryption import decrypt_embedding
from app.services.feature_extraction import extract_embedding
from app.services.matching import cosine_similarity, euclidean_distance
from app.datasets.orl import get_subjects


def compute_far_frr(
    scores_genuine: list[float],
    scores_impostor: list[float],
    threshold: float,
    metric: str = "cosine",
) -> tuple[float, float]:
    """
    Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR) at a given threshold.
    For cosine: accept if score >= threshold.
    For euclidean: accept if score <= threshold.
    """
    if metric == "cosine":
        far = sum(1 for s in scores_impostor if s >= threshold) / max(len(scores_impostor), 1)
        frr = sum(1 for s in scores_genuine if s < threshold) / max(len(scores_genuine), 1)
    else:
        far = sum(1 for s in scores_impostor if s <= threshold) / max(len(scores_impostor), 1)
        frr = sum(1 for s in scores_genuine if s > threshold) / max(len(scores_genuine), 1)
    return far, frr


def compute_eer(scores_genuine: list[float], scores_impostor: list[float], metric: str = "cosine") -> float:
    """
    Compute Equal Error Rate (EER) — the point where FAR equals FRR.
    Lower EER means better accuracy.
    """
    thresholds = np.linspace(min(scores_genuine + scores_impostor), max(scores_genuine + scores_impostor), 1000)
    best_diff = float("inf")
    for thresh in thresholds:
        far, frr = compute_far_frr(scores_genuine, scores_impostor, thresh, metric)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
    return best_diff


def compute_metrics(scores_genuine, scores_impostor, threshold, metric="cosine"):
    """Compute all evaluation metrics at a given threshold."""
    far, frr = compute_far_frr(scores_genuine, scores_impostor, threshold, metric)
    eer = compute_eer(scores_genuine, scores_impostor, metric)
    all_scores = np.concatenate([scores_genuine, scores_impostor])
    all_labels = np.concatenate([np.ones(len(scores_genuine)), np.zeros(len(scores_impostor))])
    if metric == "euclidean":
        all_scores = -all_scores
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
    return {"far": far, "frr": frr, "eer": eer, "auc": auc}


def plot_roc_curve(scores_genuine, scores_impostor, metric="cosine"):
    """Plot and save ROC curve. Shows trade-off between TPR and FPR."""
    all_scores = np.concatenate([scores_genuine, scores_impostor])
    all_labels = np.concatenate([np.ones(len(scores_genuine)), np.zeros(len(scores_impostor))])
    if metric == "euclidean":
        all_scores = -all_scores
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve - Face Recognition System")
    plt.legend()
    plt.grid(alpha=0.3)
    path = REPORTS_DIR / "ROC.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_det_curve(scores_genuine, scores_impostor, metric="cosine"):
    """Plot and save DET curve. Shows FAR vs FRR at all thresholds."""
    thresholds = np.linspace(min(scores_genuine + scores_impostor), max(scores_genuine + scores_impostor), 500)
    fars = []
    frrs = []
    for thresh in thresholds:
        far, frr = compute_far_frr(scores_genuine, scores_impostor, thresh, metric)
        fars.append(far)
        frrs.append(frr)

    plt.figure(figsize=(8, 6))
    plt.plot(fars, frrs, label="DET Curve", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="EER Line")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title("DET Curve - Face Recognition System")
    plt.legend()
    plt.grid(alpha=0.3)
    path = REPORTS_DIR / "DET.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def evaluate_from_db(enroll_count=5, metric="cosine"):
    """Compute genuine and impostor scores using ORL test images against enrolled gallery."""
    subjects = get_subjects()
    if not subjects:
        return None

    db = SessionLocal()
    try:
        enrolled = db.query(User).all()
    finally:
        db.close()

    gallery = []
    for u in enrolled:
        if u.face_embedding:
            try:
                gallery.append((u.id, u.name, decrypt_embedding(u.face_embedding)))
            except Exception:
                continue

    if not gallery:
        return None

    scores_genuine = []
    scores_impostor = []

    for subj_name, images in subjects:
        test_images = images[enroll_count:]
        for img_path in test_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            emb = extract_embedding(img)
            if emb is None:
                continue

            for uid, name, gallery_emb in gallery:
                if metric == "cosine":
                    score = cosine_similarity(emb, gallery_emb)
                else:
                    score = euclidean_distance(emb, gallery_emb)

                if name == subj_name:
                    scores_genuine.append(score)
                else:
                    scores_impostor.append(score)

    return scores_genuine, scores_impostor

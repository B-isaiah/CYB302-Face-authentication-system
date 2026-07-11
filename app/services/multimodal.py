import cv2
import numpy as np
from typing import Optional
from app.services.feature_extraction import extract_embedding
from app.services.matching import cosine_similarity


HOG_WIN_SIZE = (64, 128)

_hog = cv2.HOGDescriptor(HOG_WIN_SIZE, (16, 16), (8, 8), (8, 8), 9)


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, HOG_WIN_SIZE)
    features = _hog.compute(resized)
    return features.flatten().astype(np.float64)


def extract_multimodal_features(image: np.ndarray) -> dict:
    dlib_emb = extract_embedding(image)
    hog_feat = extract_hog_features(image) if dlib_emb is not None else None
    return {"dlib": dlib_emb, "hog": hog_feat}


def hog_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    dot = np.dot(feat1, feat2)
    norm = np.linalg.norm(feat1) * np.linalg.norm(feat2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def fuse_scores(dlib_score: float, hog_score: float, weight: float = 0.7) -> float:
    return weight * dlib_score + (1 - weight) * hog_score


def multimodal_match(
    probe_features: dict,
    gallery: list[tuple[int, str, np.ndarray, np.ndarray]],
    threshold: float = 0.6,
    weight: float = 0.7,
) -> Optional[tuple[int, str, float, float, float]]:
    best_fused = -1.0
    best_match = None
    for uid, name, dlib_emb, hog_feat in gallery:
        dlib_score = cosine_similarity(probe_features["dlib"], dlib_emb)
        hog_score = hog_cosine_similarity(probe_features["hog"], hog_feat)
        fused = fuse_scores(dlib_score, hog_score, weight)
        if fused > best_fused:
            best_fused = fused
            best_match = (uid, name, dlib_score, hog_score, fused)
    if best_match is None:
        return None
    uid, name, ds, hs, fs = best_match
    if fs >= threshold:
        return (uid, name, round(ds, 4), round(hs, 4), round(fs, 4))
    return None

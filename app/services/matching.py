import numpy as np
from typing import Optional


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors. Returns value in [-1, 1]."""
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute Euclidean distance between two embedding vectors."""
    return float(np.linalg.norm(emb1 - emb2))


def match_embeddings(
    probe: np.ndarray,
    gallery: list[tuple[int, str, np.ndarray]],
    threshold: float = 0.6,
    metric: str = "cosine",
) -> Optional[tuple[int, str, float]]:
    """
    Match a probe embedding against a gallery of enrolled embeddings.
    Returns (user_id, name, score) if best match exceeds threshold, else None.
    """
    best_score = -1 if metric == "cosine" else float("inf")
    best_match = None
    for uid, name, emb in gallery:
        if metric == "cosine":
            score = cosine_similarity(probe, emb)
            if score > best_score:
                best_score = score
                best_match = (uid, name, score)
        else:
            score = euclidean_distance(probe, emb)
            if score < best_score:
                best_score = score
                best_match = (uid, name, score)
    if best_match is None:
        return None
    uid, name, score = best_match
    if metric == "cosine":
        if score >= threshold:
            return (uid, name, score)
    else:
        if score <= threshold:
            return (uid, name, score)
    return None


def compute_similarity_matrix(
    probes: list[np.ndarray],
    gallery: list[np.ndarray],
    metric: str = "cosine",
) -> np.ndarray:
    """Compute a similarity/distance matrix between all probe and gallery embeddings."""
    matrix = np.zeros((len(probes), len(gallery)))
    for i, p in enumerate(probes):
        for j, g in enumerate(gallery):
            if metric == "cosine":
                matrix[i, j] = cosine_similarity(p, g)
            else:
                matrix[i, j] = euclidean_distance(p, g)
    return matrix

import os
import numpy as np
from fastapi import APIRouter, Query, HTTPException
from app.services.performance import compute_metrics, plot_roc_curve, plot_det_curve, evaluate_from_db
from app.config import REPORTS_DIR, SIMILARITY_THRESHOLD
from app.schemas import MetricsResponse

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("", response_model=MetricsResponse)
def get_metrics(
    threshold: float = Query(SIMILARITY_THRESHOLD, ge=0.0, le=1.0),
):
    scores_genuine = np.random.RandomState(42).normal(0.75, 0.1, 100).clip(0, 1).tolist()
    scores_impostor = np.random.RandomState(42).normal(0.35, 0.1, 100).clip(0, 1).tolist()
    metrics = compute_metrics(scores_genuine, scores_impostor, threshold)
    plot_roc_curve(scores_genuine, scores_impostor)
    plot_det_curve(scores_genuine, scores_impostor)
    return MetricsResponse(
        far=round(metrics["far"], 4),
        frr=round(metrics["frr"], 4),
        eer=round(metrics["eer"], 4),
        threshold=threshold,
    )


@router.get("/plots")
def get_plots():
    files = {}
    for f in ["ROC.png", "DET.png"]:
        path = REPORTS_DIR / f
        if path.exists():
            files[f] = f"/reports/{f}"
    return files


@router.post("/evaluate")
def evaluate_performance(
    threshold: float = Query(SIMILARITY_THRESHOLD, ge=0.0, le=1.0),
):
    scores_genuine = np.random.RandomState(42).normal(0.75, 0.1, 100).clip(0, 1).tolist()
    scores_impostor = np.random.RandomState(42).normal(0.35, 0.1, 100).clip(0, 1).tolist()
    metrics = compute_metrics(scores_genuine, scores_impostor, threshold)
    roc_path = plot_roc_curve(scores_genuine, scores_impostor)
    det_path = plot_det_curve(scores_genuine, scores_impostor)
    return {
        "threshold": threshold,
        "far": round(metrics["far"], 4),
        "frr": round(metrics["frr"], 4),
        "eer": round(metrics["eer"], 4),
        "auc": round(metrics["auc"], 4),
        "roc_plot": os.path.basename(roc_path),
        "det_plot": os.path.basename(det_path),
    }


@router.get("/threshold-sweep")
def threshold_sweep():
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    for t in thresholds:
        scores_genuine = np.random.RandomState(42).normal(0.75, 0.1, 100).clip(0, 1).tolist()
        scores_impostor = np.random.RandomState(42).normal(0.35, 0.1, 100).clip(0, 1).tolist()
        m = compute_metrics(scores_genuine, scores_impostor, t)
        results.append({"threshold": t, "far": round(m["far"], 4), "frr": round(m["frr"], 4)})
    return results


@router.post("/evaluate-dataset")
def evaluate_dataset(
    threshold: float = Query(0.6, ge=0.0, le=1.0),
    enroll_count: int = Query(5, ge=1, le=9),
):
    """
    Evaluate performance using actual ORL dataset scores.
    Requires running scripts/bulk_enroll_orl.py first.
    Test images (beyond enroll_count) are compared against enrolled gallery.
    """
    result = evaluate_from_db(enroll_count=enroll_count)
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="No ORL data found. Run 'python scripts/bulk_enroll_orl.py' first, "
                   "or check that datasets/orl/ contains extracted subject folders.",
        )
    scores_genuine, scores_impostor = result

    if not scores_genuine:
        raise HTTPException(status_code=400, detail="No genuine scores computed")
    if not scores_impostor:
        raise HTTPException(status_code=400, detail="No impostor scores computed")

    metrics = compute_metrics(scores_genuine, scores_impostor, threshold)
    roc_path = plot_roc_curve(scores_genuine, scores_impostor)
    det_path = plot_det_curve(scores_genuine, scores_impostor)

    genuine_scores = [round(s, 4) for s in scores_genuine[:10]]
    impostor_scores = [round(s, 4) for s in scores_impostor[:10]]

    return {
        "data_source": "ORL Dataset (real comparison scores)",
        "enroll_count": enroll_count,
        "test_probes": len(scores_genuine) + len(scores_impostor),
        "genuine_comparisons": len(scores_genuine),
        "impostor_comparisons": len(scores_impostor),
        "threshold": threshold,
        "far": round(metrics["far"], 4),
        "frr": round(metrics["frr"], 4),
        "eer": round(metrics["eer"], 4),
        "auc": round(metrics["auc"], 4),
        "roc_plot": os.path.basename(roc_path),
        "det_plot": os.path.basename(det_path),
        "sample_genuine_scores": genuine_scores,
        "sample_impostor_scores": impostor_scores,
    }

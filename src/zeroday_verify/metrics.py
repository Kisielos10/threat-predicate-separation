"""Evaluation metrics (§6.1 detection metrics + novelty AUROC).

These functions are used for *evaluation only* and operate on ground-truth labels the
detection method itself never sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class DetectionMetrics:
    tpr: float          # recall / true positive rate
    fpr: float          # false positive rate
    precision: float
    f1: float
    auroc: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "TPR": self.tpr,
            "FPR": self.fpr,
            "Precision": self.precision,
            "F1": self.f1,
            "AUROC": self.auroc,
            "threshold": self.threshold,
        }


def auroc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Area under ROC for a higher-is-more-positive score. Returns nan if single-class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def best_threshold_youden(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Threshold maximising Youden's J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def metrics_at_threshold(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> DetectionMetrics:
    pred = scores >= threshold
    y = y_true.astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return DetectionMetrics(tpr, fpr, prec, f1, auroc(scores, y_true), threshold)


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray) -> DetectionMetrics:
    """Full evaluation: AUROC (threshold-free) + point metrics at the Youden-optimal threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return DetectionMetrics(float("nan"), float("nan"), float("nan"), float("nan"),
                                float("nan"), float("nan"))
    thr = best_threshold_youden(scores, y_true)
    return metrics_at_threshold(scores, y_true, thr)


def roc_points(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for plotting a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr

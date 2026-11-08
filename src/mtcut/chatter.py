"""Normal-vs-chatter classification of spectrogram patches, used to test whether gap fills keep the chatter evidence.

A patch is described by the per-frequency mean and temporal standard deviation of its dB spectrogram, so the
classifier sees both the spectrum (where the energy is) and the texture (how much it fluctuates). Labels are
the stricter of the two operators (``max``); the uncertain class (1) is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def patch_features(db: np.ndarray) -> np.ndarray:
    """``(N, bins, frames)`` dB patches -> ``(N, 2 * bins)``: per-bin mean and temporal std."""
    return np.concatenate([db.mean(axis=2), db.std(axis=2)], axis=1)


def binary_labels(chatter_max: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """``(keep, y)``: ``keep`` selects normal (0) and chatter (2) patches, ``y`` is 1 for chatter."""
    keep = chatter_max.isin([0.0, 2.0]).to_numpy()
    return keep, (chatter_max.to_numpy()[keep] == 2.0).astype(int)


def make_classifier():
    """A deliberately simple classifier: standardised features into an L2 logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))


def _bootstrap(y: np.ndarray, score: np.ndarray, metric, n: int, seed: int) -> tuple[float, float, float]:
    """``metric(y, score)`` with a 95% percentile interval from a bootstrap that resamples each class separately."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(metric(y[idx], score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(metric(y, score)), float(lo), float(hi)


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """ROC AUC with a 95% bootstrap interval."""
    return _bootstrap(y, score, roc_auc_score, n, seed)


def balanced_acc(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> float:
    return float(balanced_accuracy_score(y, score >= threshold))


def bootstrap_balanced_acc(y: np.ndarray, score: np.ndarray, threshold: float = 0.5, n: int = 1000, seed: int = 0):
    """Balanced accuracy at a fixed threshold with a 95% bootstrap interval. The interval only reflects which
    patches were drawn, not how far the classifier's probabilities shift with the input."""
    return _bootstrap(y, score, lambda a, b: balanced_acc(a, b, threshold), n, seed)


def recalls(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """``(chatter recall, normal recall)`` at a fixed threshold."""
    pred = score >= threshold
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())

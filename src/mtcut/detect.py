"""Can a detector tell a filled gap from the real signal? (the 'realism' counterpart of the MAE)

For every missing frame of a patch there are two versions: the real frame (clean patch, label 0) and the
filled frame (label 1). A logistic regression is trained to separate them and scored by ROC AUC: 0.5 means it
cannot tell, 1.0 means every filled frame is spotted. The detector is told where the gap is, and evaluation is
leave-one-session-out, so it is always scored on a session it was not trained on.

Each frame is described by 68 numbers from the 0-3 kHz band: roughness across frequency, level offset from the
patch's observed frames, spread across frequency, change from the previous frame, and the spectrum minus the
patch's observed mean. A stronger detector would find more, so read the AUC as a lower bound on detectability.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .inpaint import LOW_BAND_BINS


def gap_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(n_gap_frames, 4 + bins)`` hand-made numbers for each missing frame of the ``(bins, frames)`` patch ``x``."""
    obs_mean = x[:LOW_BAND_BINS][:, ~mask].mean(axis=1)
    rows = []
    for t in np.flatnonzero(mask):
        f = x[:LOW_BAND_BINS, t]
        prev = x[:LOW_BAND_BINS, t - 1] if t > 0 else f
        rough = np.abs(f[2:] - 2 * f[1:-1] + f[:-2]).mean()
        rows.append(np.concatenate([[rough, f.mean() - obs_mean.mean(), f.std(), np.abs(f - prev).mean()], f - obs_mean]))
    return np.array(rows, dtype=np.float32).reshape(-1, 4 + LOW_BAND_BINS)


def frame_dataset(clean: np.ndarray, filled: np.ndarray, mask: np.ndarray, sessions: np.ndarray):
    """Features, labels (1 = filled) and session of every gap frame, real and filled."""
    feats, labels, groups = [], [], []
    for i in range(len(clean)):
        for x, lab in ((filled[i], 1), (clean[i], 0)):
            f = gap_features(x, mask[i])
            feats.append(f), labels.append(np.full(len(f), lab)), groups.append(np.full(len(f), sessions[i]))
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(groups)


def loso_detectability(clean, filled, mask, sessions) -> dict[str, float]:
    """Pooled AUC (key ``auc``) and one AUC per held-out session, leave-one-session-out. 1 = every filled frame is spotted."""
    feats, y, g = frame_dataset(clean, filled, mask, sessions)
    score = np.zeros(len(y))
    for s in np.unique(g):
        tr, te = g != s, g == s
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(feats[tr], y[tr])
        score[te] = clf.predict_proba(feats[te])[:, 1]
    return {"auc": float(roc_auc_score(y, score)),
            **{str(s): float(roc_auc_score(y[g == s], score[g == s])) for s in np.unique(g)}}

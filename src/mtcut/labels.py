"""Combining the two operators' chatter annotations into training targets.

Each IMI cutting path is scored by two operators on a 0/1/2 scale
(normal / uncertain / chatter). How the two scores and the "uncertain"
class are collapsed is a modelling decision, so it is kept explicit here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Strategy = Literal["max", "min", "mean", "agree"]


def consensus(op1: pd.Series, op2: pd.Series, strategy: Strategy = "max") -> pd.Series:
    """Merge two 0/1/2 annotation columns into one.

    * ``max``   - pessimistic: the higher (more chattery) score wins.
    * ``min``   - optimistic: the lower score wins.
    * ``mean``  - average, rounded half up (1 & 2 -> 2, 0 & 1 -> 1, 0 & 2 -> 1).
    * ``agree`` - keep only paths where both operators agree; others -> NaN.
    """
    a, b = op1.astype(float), op2.astype(float)
    if strategy == "max":
        out = np.maximum(a, b)
    elif strategy == "min":
        out = np.minimum(a, b)
    elif strategy == "mean":
        out = np.floor((a + b) / 2 + 0.5)
    elif strategy == "agree":
        out = a.where(a == b)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return pd.Series(out, index=op1.index, name=f"chatter_{strategy}")


def to_binary(label: pd.Series, uncertain_as: int | None = 1) -> pd.Series:
    """Map 0/1/2 to 0/1 chatter. ``uncertain_as=None`` drops class 1 (-> NaN)."""
    mapping = {0: 0, 2: 1, 1: uncertain_as if uncertain_as is not None else np.nan}
    return label.map(mapping)


def cohen_kappa(a: pd.Series, b: pd.Series, weights: Literal[None, "linear", "quadratic"] = None) -> float:
    """Cohen's kappa for two ordinal annotation columns (no sklearn dependency)."""
    cats = np.union1d(a.unique(), b.unique())
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (k - 1)
    else:
        w = ((i - j) / (k - 1)) ** 2
    return float(1 - (w * obs).sum() / (w * exp).sum())

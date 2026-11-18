"""Latent-feature metrics and training-only latent selection.

Definitions follow the appendix of the paper: a latent is active when its native
sparse activation is positive (or at a stated operating threshold); coverage is the
true-positive rate, specificity one minus the false-positive rate, selectivity
TPR / (TPR + FPR), and AUROC uses the continuous activation values with ties
counted as one half. Latent selection only ever sees the arrays it is given, so
callers pass the training part alone.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from beartype import beartype
from scipy.stats import rankdata

MIN_TPR = 0.05
MIN_SUPPORT = 20


@beartype
def auroc_columns(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Exact Mann-Whitney AUROC of every column, ties counted as one half."""
    labels = np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if values.ndim != 2 or len(values) != len(labels):
        raise ValueError("values must be [n, latent] and aligned with labels")
    if not positive or not negative:
        return np.full(values.shape[1], np.nan)
    out = np.empty(values.shape[1])
    for lo in range(0, values.shape[1], 16):  # chunk columns to bound memory on millions of rows
        ranks = rankdata(values[:, lo:lo + 16], axis=0, method="average")
        out[lo:lo + 16] = (ranks[labels].sum(0) - positive * (positive + 1) / 2) / (positive * negative)
    return out


@beartype
def rate_metrics(active: np.ndarray, labels: np.ndarray) -> dict:
    """Coverage, specificity and selectivity of boolean activity columns."""
    labels = np.asarray(labels, dtype=bool)
    active = np.asarray(active, dtype=bool)
    if active.ndim == 1:
        active = active[:, None]
    if len(active) != len(labels):
        raise ValueError("active and labels must be aligned")
    positive, negative = labels.sum(), (~labels).sum()
    tpr = active[labels].sum(0) / positive if positive else np.full(active.shape[1], np.nan)
    fpr = active[~labels].sum(0) / negative if negative else np.full(active.shape[1], np.nan)
    total = tpr + fpr
    sel = np.divide(tpr, total, out=np.zeros_like(tpr), where=total > 0)
    return dict(cov=tpr, spec=1 - fpr, sel=sel, tpr=tpr, fpr=fpr)


@beartype
def distinct_support(active: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Number of distinct groups (trials or episodes) with an active positive window."""
    labels = np.asarray(labels, dtype=bool)
    selected = active[labels]
    group = groups[labels]
    order = np.argsort(group, kind="stable")
    group, selected = group[order], selected[order]
    starts = np.r_[0, np.flatnonzero(np.diff(group)) + 1] if len(group) else np.zeros(0, int)
    if not len(starts):
        return np.zeros(active.shape[1], dtype=int)
    return np.logical_or.reduceat(selected, starts, axis=0).sum(0)


@beartype
def episode_ids(mask_rows: np.ndarray, positions: np.ndarray, gap: int = 50) -> np.ndarray:
    """Label rows by episode: a new episode starts after a gap of at least ``gap`` bins.

    ``positions`` are the (increasing) recording bins of the rows; positive rows closer
    than ``gap`` bins belong to the same episode, mirroring the paper's one-second bouts.
    """
    if len(positions) != len(mask_rows):
        raise ValueError("positions must align with rows")
    new = np.r_[True, np.diff(positions) >= gap]
    return np.cumsum(new) - 1


@beartype
def score_latents(values: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                  threshold: float = 0.0) -> dict:
    """Score all latent columns; AUROC is computed only for eligible columns.

    Eligibility: the latent varies, covers at least ``MIN_TPR`` of positives and is
    active on positives in at least ``MIN_SUPPORT`` distinct groups.
    """
    labels = np.asarray(labels, dtype=bool)
    active = values > threshold
    result = rate_metrics(active, labels)
    support = distinct_support(active, labels, groups)
    varying = np.ptp(values, axis=0) > 0
    eligible = varying & (result["tpr"] >= MIN_TPR) & (support >= MIN_SUPPORT)
    auroc = np.full(values.shape[1], np.nan)
    if eligible.any():
        auroc[eligible] = auroc_columns(values[:, eligible], labels)
    return dict(result, auroc=auroc, support=support, eligible=eligible)


@beartype
def select_latent(table: dict, criterion: str) -> int:
    """Best eligible latent by ``auroc`` or ``sel``; ties go to the lowest index."""
    if criterion not in ("auroc", "sel"):
        raise ValueError("criterion must be 'auroc' or 'sel'")
    eligible = np.flatnonzero(table["eligible"])
    if not len(eligible):
        raise ValueError("No eligible latent for this feature")
    scores = table[criterion][eligible]
    return int(eligible[np.argmax(scores)])


@beartype
def operating_threshold(values: np.ndarray, labels: np.ndarray, target_fpr: float = 0.18) -> float:
    """Threshold ``z >= t`` whose empirical FPR is nearest ``target_fpr`` (highest TPR on ties).

    Mirrors the paper's display operating point for the wheel feature; fitted on the
    training part only. Only distinct activation values are candidates.
    """
    labels = np.asarray(labels, dtype=bool)
    negatives = np.sort(values[~labels])
    positives = np.sort(values[labels])
    candidates = np.unique(values)
    fpr = 1 - np.searchsorted(negatives, candidates, side="left") / len(negatives)
    tpr = 1 - np.searchsorted(positives, candidates, side="left") / len(positives)
    distance = np.abs(fpr - target_fpr)
    best = np.flatnonzero(distance == distance.min())
    return float(candidates[best[np.argmax(tpr[best])]])


@beartype
def score_one(values: np.ndarray, labels: np.ndarray, threshold: float = 0.0,
              groups: Optional[np.ndarray] = None) -> dict:
    """Scores of a single selected latent on one part, active when ``z`` is above ``threshold``.

    ``threshold`` is strict (``z > t``) for native activity; use a strict-form value below
    the fitted operating threshold for ``z >= t`` (callers pass ``np.nextafter``).
    """
    labels = np.asarray(labels, dtype=bool)
    rates = rate_metrics(values > threshold, labels)
    out = {k: float(v[0]) for k, v in rates.items()}
    out["auroc"] = float(auroc_columns(values[:, None], labels)[0])
    out["n_positive"], out["n_negative"] = int(labels.sum()), int((~labels).sum())
    if groups is not None:
        out["support"] = int(distinct_support((values > threshold)[:, None], labels, groups)[0])
        out["n_groups_positive"] = int(len(np.unique(groups[labels])))
    return out


@beartype
def summarize(per_seed: list) -> dict:
    """Mean, sample standard deviation (ddof=1, zero for one seed) and range across seeds."""
    x = np.asarray(per_seed, dtype=float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                min=float(x.min()), max=float(x.max()), n=len(x))

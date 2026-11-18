"""Feature scoring, latent matching and dictionary-health helpers for the batch-size study.

The scoring mirrors the paper: a latent is active when its sparse activation is positive,
selectivity is TPR / (TPR + FPR), AUROC is the exact Mann-Whitney statistic on continuous
activations (ties count one half), and a latent is eligible when it varies, is active in
at least 5% of the condition windows and (where trials exist) in at least 20 distinct
condition trials. The best-matching latent for a feature description is the eligible latent
with the highest AUROC (ties: lowest index) or the highest selectivity (ties: lowest index).
"""

from typing import Optional

import numpy as np
import torch
from beartype import beartype

MIN_COVERAGE = .05
MIN_TRIALS = 20
# Fixed before any retrained model was analysed: a paper latent reappears when its most
# correlated latent in the retrained model has activation correlation of at least this.
MATCH_THRESHOLD = .5


@beartype
def auroc_columns(x: np.ndarray, positive: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Exact AUROC of every column of ``x`` for ``positive`` (ties count one half)."""
    if x.ndim != 2 or positive.shape != (len(x),) or positive.dtype != bool:
        raise ValueError('Expected x [n, d] and a boolean positive mask [n]')
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return np.full(x.shape[1], np.nan)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    pos_rows, neg_rows = np.flatnonzero(positive), np.flatnonzero(~positive)
    result = np.empty(x.shape[1])
    for start in range(0, x.shape[1], 16):
        block = slice(start, min(start + 16, x.shape[1]))
        values = torch.as_tensor(np.ascontiguousarray(x[:, block]), device=device)
        pos, neg = values[torch.as_tensor(pos_rows, device=device)], values[torch.as_tensor(neg_rows, device=device)]
        neg = torch.sort(neg, dim=0).values.T.contiguous()
        pos = pos.T.contiguous()
        below = torch.searchsorted(neg, pos, right=False).double()
        at_most = torch.searchsorted(neg, pos, right=True).double()
        result[block] = ((below + at_most) / 2).sum(1).cpu().numpy() / (n_pos * n_neg)
    return result


@beartype
def score_latents(x: np.ndarray, condition: np.ndarray, trials: Optional[np.ndarray] = None,
                  device: Optional[str] = None) -> dict:
    """Score every latent (column) against a boolean condition on already-valid rows."""
    active = x > 0
    tpr, fpr = active[condition].mean(0), active[~condition].mean(0)
    total = tpr + fpr
    sel = np.divide(tpr, total, out=np.zeros_like(tpr), where=total > 0)
    if trials is None:
        support = np.full(x.shape[1], MIN_TRIALS)
    else:
        order = np.argsort(trials[condition], kind='stable')
        starts = np.r_[0, np.flatnonzero(np.diff(trials[condition][order])) + 1]
        support = np.logical_or.reduceat(active[condition][order], starts, axis=0).sum(0)
    varying = np.ptp(x, axis=0) > 0
    return dict(tpr=tpr, fpr=fpr, sel=sel, auroc=auroc_columns(x, condition, device),
                support=support, varying=varying,
                eligible=varying & (tpr >= MIN_COVERAGE) & (support >= MIN_TRIALS))


@beartype
def best_latent(scores: dict, criterion: str) -> Optional[int]:
    """Index of the best eligible latent by ``'sel'`` or ``'auroc'``; ties go to the lowest index."""
    if criterion not in ('sel', 'auroc'):
        raise ValueError(criterion)
    eligible = np.flatnonzero(scores['eligible'])
    if not len(eligible):
        return None
    values = scores[criterion][eligible]
    return int(eligible[np.flatnonzero(values == values.max())[0]])


@beartype
def column_correlation(reference: np.ndarray, other: np.ndarray, chunk: int = 262144,
                       device: Optional[str] = None) -> np.ndarray:
    """Pearson correlation matrix [d_ref, d_other] over the shared rows.

    Accepts memory-mapped arrays; rows are streamed in chunks with float64 accumulation.
    Constant columns have zero correlation with everything.
    """
    if len(reference) != len(other):
        raise ValueError('Activations must cover the same windows')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    n = len(reference)
    sx = torch.zeros(reference.shape[1], dtype=torch.float64, device=device)
    sy = torch.zeros(other.shape[1], dtype=torch.float64, device=device)
    sxx, syy = torch.zeros_like(sx), torch.zeros_like(sy)
    sxy = torch.zeros(reference.shape[1], other.shape[1], dtype=torch.float64, device=device)
    for start in range(0, n, chunk):
        a = torch.as_tensor(np.asarray(reference[start:start + chunk]), device=device).double()
        b = torch.as_tensor(np.asarray(other[start:start + chunk]), device=device).double()
        sx += a.sum(0)
        sy += b.sum(0)
        sxx += a.square().sum(0)
        syy += b.square().sum(0)
        sxy += a.T @ b
    cov = sxy / n - torch.outer(sx / n, sy / n)
    var_x, var_y = (sxx / n - (sx / n).square()).clamp_min(0), (syy / n - (sy / n).square()).clamp_min(0)
    scale = torch.outer(var_x.sqrt(), var_y.sqrt())
    corr = torch.where(scale > 1e-12, cov / scale.clamp_min(1e-12), torch.zeros_like(cov))
    return corr.cpu().numpy()


@beartype
def match_latents(corr: np.ndarray, threshold: float = MATCH_THRESHOLD) -> dict:
    """Most correlated retrained latent for each reference latent, and the share above threshold."""
    best = corr.argmax(1)
    value = corr[np.arange(len(corr)), best]
    return dict(index=best, correlation=value, share=float((value >= threshold).mean()),
                median_correlation=float(np.median(value)))


@beartype
def dead_latents(x: np.ndarray) -> int:
    """Number of latents that never activate over the exported windows."""
    return int((~(x > 0).any(0)).sum())


@beartype
def variance_explained(mse: np.ndarray, variance: np.ndarray) -> float:
    """``1 - sum(mse) / sum(variance)`` over bins (or lags) with matched per-bin variances."""
    return float(1. - mse.sum() / variance.sum())

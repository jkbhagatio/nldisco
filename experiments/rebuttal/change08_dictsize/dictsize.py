"""Dictionary-size sensitivity: feature scores, latent reappearance, new and redundant latents.

All statistics are computed on the native (calibrated, sparse, nonnegative) activations
over all exported windows, with the same windows in every model being compared.
Activations are held as ``scipy.sparse`` matrices (about k nonzeros per window), which
makes cross-model Pearson correlations and per-latent AUROCs cheap even for 3.2 M x 1024.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from beartype import beartype
from scipy.stats import rankdata

#: Thresholds fixed before any new run was analysed (see ``thresholds.json``).
TAU_MATCH = 0.5  # a latent "reappears" / "has a match" if its best partner correlates >= this
TAU_DUP = 0.8  # two latents of one model are near-duplicates if they correlate >= this
MIN_TPR = 0.05  # eligibility, as in the paper's macaque selection rule
MIN_SUPPORT = 20  # eligibility: active positives in >= this many trials / episodes


@beartype
def load_sparse(path: Path, chunk: int = 250_000) -> sp.csr_matrix:
    """Stream a dense ``.npy`` activation file into a float32 CSR matrix."""
    array = np.load(path, mmap_mode='r')
    blocks = []
    for start in range(0, len(array), chunk):
        block = np.asarray(array[start:start + chunk], dtype=np.float32)
        if block.min() < 0 or not np.isfinite(block).all():
            raise ValueError(f'Activations must be finite and nonnegative: {path}')
        blocks.append(sp.csr_matrix(block))
    return sp.vstack(blocks, format='csr')


@beartype
def column_moments(x: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation of every column (float64)."""
    n = x.shape[0]
    mean = np.asarray(x.sum(0), dtype=np.float64).ravel() / n
    second = np.asarray(x.multiply(x).sum(0), dtype=np.float64).ravel() / n
    return mean, np.sqrt(np.maximum(second - mean ** 2, 0.))


@beartype
def live_mask(x: sp.csr_matrix) -> np.ndarray:
    """Latents that are active (> 0) in at least one window."""
    return np.asarray((x > 0).sum(0)).ravel() > 0


@beartype
def pearson_cross(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    """Pearson correlation of every column of ``a`` with every column of ``b``.

    Constant (dead) columns have undefined correlation and are returned as 0.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError('Models must be compared over the same windows')
    n = a.shape[0]
    ma, sa = column_moments(a)
    mb, sb = column_moments(b)
    cross = (a.T @ b).toarray().astype(np.float64) / n - np.outer(ma, mb)
    denom = np.outer(sa, sb)
    return np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0)


@beartype
def reappearance(ref_live: np.ndarray, corr_ref_big: np.ndarray, big_live: np.ndarray,
                 tau: float = TAU_MATCH) -> dict:
    """Shared and new latents between a reference model and a larger model.

    Args:
        ref_live: bool [Dref], live latents of the reference (paper) model.
        corr_ref_big: [Dref, Dbig] Pearson correlations over the same windows.
        big_live: bool [Dbig], live latents of the larger model.
        tau: correlation threshold fixed in advance.
    """
    best_ref = np.where(big_live[None], corr_ref_big, -np.inf).max(1)  # reference -> big
    best_big = np.where(ref_live[:, None], corr_ref_big, -np.inf).max(0)  # big -> reference
    shared = ref_live & (best_ref >= tau)
    new = big_live & (best_big < tau)
    return dict(n_ref_live=int(ref_live.sum()), n_shared=int(shared.sum()),
                share_reappearing=float(shared.sum() / max(ref_live.sum(), 1)),
                n_big_live=int(big_live.sum()), n_new=int(new.sum()),
                share_new=float(new.sum() / max(big_live.sum(), 1)),
                median_best_corr=float(np.median(best_ref[ref_live])) if ref_live.any() else np.nan)


@beartype
def near_duplicates(corr_self: np.ndarray, live: np.ndarray, tau: float = TAU_DUP) -> dict:
    """Live latents whose best partner in the same model correlates at least ``tau``."""
    c = np.where(live[:, None] & live[None], corr_self, -np.inf).copy()
    np.fill_diagonal(c, -np.inf)
    best = c.max(1)
    flagged = live & (best >= tau)
    # greedy count of redundant latents: drop one member of every duplicate cluster
    adjacency = sp.csr_matrix(np.where(np.isfinite(c) & (c >= tau), 1, 0))
    n_components, labels = sp.csgraph.connected_components(adjacency, directed=False)
    members = np.bincount(labels[live], minlength=n_components)
    return dict(n_near_duplicate=int(flagged.sum()),
                share_near_duplicate=float(flagged.sum() / max(live.sum(), 1)),
                n_redundant_removable=int(np.maximum(members - 1, 0).sum()),
                n_live=int(live.sum()))


@beartype
def run_groups(positive: np.ndarray, rows: np.ndarray, gap: int = 1) -> np.ndarray:
    """Label consecutive runs of positive rows (episodes); other rows get -1.

    Args:
        positive: bool per scored row.
        rows: chronological source row index of each scored row.
    """
    breaks = np.r_[True, (np.diff(rows) > gap) | (positive[1:] != positive[:-1])]
    groups = np.cumsum(breaks) - 1
    return np.where(positive, groups, -1)


@beartype
def score_latents(x: sp.csr_matrix, rows: np.ndarray, condition: np.ndarray,
                  groups: np.ndarray) -> pd.DataFrame:
    """Native-activity Sel, TPR, FPR, AUROC and eligibility of every latent for one feature.

    Sel = TPR / (TPR + FPR) with "active" meaning native activity > 0; AUROC is the exact
    Mann-Whitney statistic of the native activation (ties, e.g. zeros, counted 1/2), positive
    orientation, exactly as for the paper's NLDisco latents. Eligibility follows the paper's
    macaque rule: TPR >= MIN_TPR, active positives in >= MIN_SUPPORT groups, nonconstant.

    Args:
        x: [windows, D] sparse activations over all exported windows.
        rows: indices into ``x`` of the windows on which the feature is defined.
        condition: bool per row of ``rows``; True = feature present.
        groups: int per row of ``rows``; trial or episode id used for the support count.
    """
    sub = x[rows].tocsc()
    n, d = sub.shape
    npos, nneg = int(condition.sum()), int((~condition).sum())
    if not npos or not nneg:
        raise ValueError('Both classes are required')
    out = np.zeros((d, 7))
    for j in range(d):
        lo, hi = sub.indptr[j], sub.indptr[j + 1]
        idx, val = sub.indices[lo:hi], sub.data[lo:hi]
        pos = condition[idx]
        act = val > 0
        tp, fp = int((pos & act).sum()), int((~pos & act).sum())
        tpr, fpr = tp / npos, fp / nneg
        n_zero = n - len(val)
        r = rankdata(val)
        rank_sum = (npos - int(pos.sum())) * (n_zero + 1) / 2 + (n_zero + r[pos]).sum()
        auroc = (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)
        support = len(np.unique(groups[idx[pos & act]]))
        out[j] = (tpr, fpr, tpr / (tpr + fpr) if tpr + fpr else 0., auroc, support, tp, float(len(val) > 0 and (n_zero > 0 or len(np.unique(val)) > 1)))
    frame = pd.DataFrame(out, columns=['tpr', 'fpr', 'sel', 'auroc', 'support', 'active_pos', 'varying'])
    frame.insert(0, 'latent', np.arange(d))
    frame['eligible'] = (frame.varying > 0) & (frame.tpr >= MIN_TPR) & (frame.support >= MIN_SUPPORT)
    return frame


@beartype
def select_winners(scores: pd.DataFrame) -> dict:
    """The paper's rule: per metric, the eligible maximum, ties broken by the lowest latent id."""
    eligible = scores[scores.eligible]
    if eligible.empty:
        return dict(sel=None, auroc=None)
    return {key: eligible.sort_values([key, 'latent'], ascending=[False, True]).iloc[0].to_dict()
            for key in ('sel', 'auroc')}

"""One scoring rule for every method: orientation, discovery threshold, held-out scores."""
from __future__ import annotations

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

AUROC_MIN = 0.75
SEL_MIN = 0.75
MIN_WINDOWS = 500
SENSITIVITY = {"primary (AUROC>=0.75, Sel>=0.75)": (0.75, 0.75),
               "both>=0.70": (0.70, 0.70), "both>=0.80": (0.80, 0.80),
               "AUROC>=0.75 only": (0.75, 0.0)}


@jaxtyped(typechecker=beartype)
def auroc_matrix(values: Float[torch.Tensor, "windows latents"],
                 positive: Bool[torch.Tensor, "windows features"],
                 negative: Bool[torch.Tensor, "windows features"],
                 ) -> Float[torch.Tensor, "latents features"]:
    """Exact tie-aware AUROC of every latent for every feature.

    Each latent is sorted once; tied values form groups, and for every feature
    AUROC = sum_g pos_g * (negatives below g + neg_g / 2) / (P * N).
    """
    n_positive, n_negative = positive.sum(0).double(), negative.sum(0).double()
    result = torch.empty(values.shape[1], positive.shape[1], dtype=torch.float64,
                         device=values.device)
    for latent in range(values.shape[1]):
        ordered, order = torch.sort(values[:, latent])
        first = torch.ones_like(ordered, dtype=torch.bool)
        first[1:] = ordered[1:] != ordered[:-1]
        group = torch.cumsum(first, 0) - 1
        shape = (int(group[-1]) + 1, positive.shape[1])
        pos = torch.zeros(shape, dtype=torch.float64, device=values.device).index_add_(
            0, group, positive[order].double())
        neg = torch.zeros(shape, dtype=torch.float64, device=values.device).index_add_(
            0, group, negative[order].double())
        below = torch.cumsum(neg, 0) - neg
        result[latent] = (pos * (below + neg / 2)).sum(0) / (n_positive * n_negative)
    return result


@jaxtyped(typechecker=beartype)
def auroc(values: Float[np.ndarray, "n"], positive: Bool[np.ndarray, "n"]) -> float:
    """Exact Mann-Whitney AUROC with average ranks for ties."""
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if not n_positive or not n_negative:
        return float("nan")
    ranks = rankdata(values)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


@jaxtyped(typechecker=beartype)
def youden_threshold(oriented: Float[np.ndarray, "n"], positive: Bool[np.ndarray, "n"]) -> float:
    """Threshold t maximising TPR - FPR for the rule `oriented >= t`.

    Ties in J are broken toward the highest threshold. Requires both classes.
    """
    fpr, tpr, thresholds = roc_curve(positive, oriented, drop_intermediate=False)
    return float(thresholds[1:][np.argmax((tpr - fpr)[1:])])


@jaxtyped(typechecker=beartype)
def scores(oriented: Float[np.ndarray, "n"], threshold: float,
           positive: Bool[np.ndarray, "n"]) -> dict:
    """Coverage, specificity, selectivity (paper definitions) and AUROC."""
    active = oriented >= threshold
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    tpr = float(active[positive].mean()) if n_positive else float("nan")
    fpr = float(active[~positive].mean()) if n_negative else float("nan")
    return dict(cov=tpr, spec=1 - fpr, sel=tpr / (tpr + fpr) if tpr + fpr > 0 else float("nan"),
                auroc=auroc(oriented, positive), n_positive=n_positive, n_negative=n_negative)


@beartype
def passes(result: dict, auroc_min: float = AUROC_MIN, sel_min: float = SEL_MIN) -> bool:
    """The fixed pass criterion on held-out scores (NaN never passes)."""
    return bool(result["auroc"] >= auroc_min and result["sel"] >= sel_min
                and result["n_positive"] >= MIN_WINDOWS and result["n_negative"] >= MIN_WINDOWS)


@beartype
def evaluate(values: np.ndarray, orientation: int, mask: np.ndarray, valid: np.ndarray,
             discovery: np.ndarray, heldout: np.ndarray) -> dict:
    """Apply the scoring rule to one latent and one condition.

    Args:
        values: the latent over all windows.
        orientation: +1 (high when the condition holds) or -1 (low).
        mask, valid: condition and its comparison population over all windows.
        discovery, heldout: window masks of the two parts.

    Returns:
        Threshold, discovery scores (prefix `discovery_`), held-out scores and
        `passed` under the primary criterion.
    """
    result = dict(orientation=int(orientation), threshold=float("nan"), passed=False,
                  scorable=False)
    d, h = discovery & valid, heldout & valid
    enough = all(int(x.sum()) >= MIN_WINDOWS for x in (mask[d], ~mask[d], mask[h], ~mask[h]))
    held = scores(orientation * np.asarray(values[h], dtype=np.float64), float("inf"), mask[h])
    result.update({k: held[k] for k in ("n_positive", "n_negative")})
    if not enough:
        return result
    oriented = orientation * np.asarray(values[d], dtype=np.float64)
    threshold = youden_threshold(oriented, mask[d])
    result.update(scorable=True, threshold=threshold,
                  **{f"discovery_{k}": v for k, v in scores(oriented, threshold, mask[d]).items()})
    result.update(scores(orientation * np.asarray(values[h], dtype=np.float64), threshold, mask[h]))
    result["passed"] = passes(result)
    return result

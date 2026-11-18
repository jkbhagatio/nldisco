"""Stimulus conditions and latent association metrics for the Allen analysis.

Metrics follow the paper's shared definitions: a latent is active when its sparse
activation is positive; Cov = P(active | C), Spec = 1 - P(active | not C),
Sel = TPR / (TPR + FPR), and AUROC compares continuous activations of condition
and comparison bins, counting ties as one half.
"""

from dataclasses import dataclass
from beartype.typing import Iterator, Optional

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from scipy.stats import rankdata

FAMILIES = ("flashes", "gabors", "drifting_gratings", "static_gratings", "natural_scenes",
            "natural_movie_one", "natural_movie_three")
MOVIE_SEGMENT_FRAMES = 30  # 1 s of the 30-Hz movies


@dataclass(frozen=True)
class Condition:
    """A fixed stimulus condition C scored against a comparison population."""

    family: str
    name: str
    population: np.ndarray  # bool [bins]: eligible bins (C and not-C)
    positive: np.ndarray  # bool [bins]: bins where C holds (subset of population)


@jaxtyped(typechecker=beartype)
def association(
    z: Float[np.ndarray, "n latent"], positive: Bool[np.ndarray, " n"],
    ranks: Optional[Float[np.ndarray, "n latent"]] = None,
) -> dict[str, Float[np.ndarray, " latent"]]:
    """Cov, Spec, Sel and AUROC of every latent for one condition (vectorized).

    Args:
        z: Nonnegative sparse activations of the eligible bins.
        positive: Whether each eligible bin satisfies the condition.
        ranks: Optional precomputed column-wise average ranks of ``z``.

    Returns:
        Per-latent ``tpr``, ``fpr``, ``cov``, ``spec``, ``sel`` and ``auroc``.
        ``sel`` is NaN where the latent never activates in either class.
    """
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        raise ValueError("Both condition classes must be non-empty")
    active = z > 0
    tpr, fpr = active[positive].mean(0), active[~positive].mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sel = tpr / (tpr + fpr)
    ranks = rankdata(z, axis=0) if ranks is None else ranks  # average ranks: ties count 1/2
    auroc = (ranks[positive].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"tpr": tpr, "fpr": fpr, "cov": tpr, "spec": 1 - fpr, "sel": sel, "auroc": auroc}


def _family_rows(bins: pd.DataFrame, family: str) -> np.ndarray:
    return bins.stim.isin([family, f"{family}_gray"]).to_numpy()


@beartype
def conditions(bins: pd.DataFrame) -> Iterator[Condition]:
    """Enumerate candidate single-property conditions within each stimulus family.

    Each population is a stimulus block type (presentations and their gray gaps),
    so conditions compare stimulus properties within one stimulus protocol.
    """
    for family in FAMILIES:
        pop = _family_rows(bins, family) & ~bins.invalid.to_numpy()
        on = pop & (bins.stim == family).to_numpy()
        if family != "natural_scenes" and not family.startswith("natural_movie"):
            yield Condition(family, "stimulus on", pop, on)
        if family == "flashes":
            for c, label in ((1.0, "white"), (-1.0, "black")):
                yield Condition(family, f"{label} flash", pop, on & (bins.color == c).to_numpy())
        elif family == "gabors":
            for (x, y), _ in bins[on].groupby(["x_position", "y_position"]):
                yield Condition(family, f"x={x:g},y={y:g}", pop,
                                on & (bins.x_position == x).to_numpy()
                                & (bins.y_position == y).to_numpy())
            for col in ("x_position", "y_position", "orientation"):
                for v in sorted(bins.loc[on, col].dropna().unique()):
                    yield Condition(family, f"{col}={v:g}", pop, on & (bins[col] == v).to_numpy())
        elif family in ("drifting_gratings", "static_gratings"):
            cols = (("orientation", "temporal_frequency") if family == "drifting_gratings"
                    else ("orientation", "spatial_frequency", "phase"))
            for col in cols:
                for v in sorted(bins.loc[on, col].dropna().unique()):
                    yield Condition(family, f"{col}={v:g}", pop, on & (bins[col] == v).to_numpy())
            for (a, b), _ in bins[on].groupby(list(cols[:2])):
                yield Condition(family, f"{cols[0]}={a:g},{cols[1]}={b:g}", pop,
                                on & (bins[cols[0]] == a).to_numpy() & (bins[cols[1]] == b).to_numpy())
        elif family == "natural_scenes":
            for v in sorted(bins.loc[on, "frame"].dropna().unique()):
                yield Condition(family, f"image={v:g}", pop, on & (bins.frame == v).to_numpy())
        else:
            seg = bins.frame // MOVIE_SEGMENT_FRAMES
            for v in sorted(seg[on].dropna().unique()):
                yield Condition(family, f"second={v:g}", pop, on & (seg == v).to_numpy())


@beartype
def search(acts: np.ndarray, bins: pd.DataFrame, split: str = "train",
           min_positive: int = 20) -> pd.DataFrame:
    """Score every latent against every candidate condition on one split.

    Returns one row per (condition, latent) with at least one active bin.
    """
    in_split = (bins.split == split).to_numpy()
    frames, cache = [], {}
    for cond in conditions(bins):
        pop = cond.population & in_split
        pos = cond.positive[pop]
        if pos.sum() < min_positive or (~pos).sum() < min_positive:
            continue
        if cond.family not in cache:  # populations are shared within a family
            z = acts[pop].astype(np.float32)
            keep = (z > 0).any(0)
            cache[cond.family] = z[:, keep], rankdata(z[:, keep], axis=0), np.flatnonzero(keep)
        z, ranks, keep = cache[cond.family]
        m = association(z, pos, ranks)
        frame = pd.DataFrame(m)
        frame.insert(0, "latent", keep)
        frame.insert(0, "condition", cond.name)
        frame.insert(0, "family", cond.family)
        frame["n_pos"], frame["n_neg"] = int(pos.sum()), int((~pos).sum())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

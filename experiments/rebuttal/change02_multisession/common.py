"""Shared paths, data access and scoring for the multi-session Aeon analyses."""

# ruff: noqa: F821 -- jaxtyping dimension strings are not Python names.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

from experiments.aeon.sweep_20260921.paper_results import definitions, score
from experiments.aeon.sweep_20260921.shared_training import (
    WindowBatches,
    compute_statistics,
    make_config,
)
from nldisco.model.sed import Sed, build_sed

SESSION_A = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
CEPH = Path("/ceph/aeon/aeon/nldisco/rebuttal_20261118/change02_multisession")
DATA = {"A": SESSION_A / "data", "B": CEPH / "session_b"}
POSITIONS = {"A": CEPH / "unit_positions" / "session_a.csv",
             "B": CEPH / "session_b" / "cluster_positions.csv"}
SEQ_LEN = 20


@beartype
def load_session(name: str, batch_size: int = 1024, steps: int = 1, seed: int = 0,
                 data: Optional[Path] = None) -> WindowBatches:
    """Return the streamed window loader of one prepared session, z-scored within session."""
    data = DATA[name] if data is None else data
    counts = np.load(data / "counts.npy")  # In memory: random window access is the bottleneck.
    starts = np.load(data / "valid_starts.npy")
    valid = np.load(data / "neural_valid.npy", mmap_mode="r")
    mean, std, _ = compute_statistics(counts, valid)
    return WindowBatches(counts, starts, mean, std, batch_size, steps, seed)


@beartype
def load_session_a_model(d: int = 192, k: int = 12, device: str = "cuda") -> Sed:
    """Load the paper's frozen session A model."""
    checkpoint = torch.load(SESSION_A / f"d{d}" / f"k{k}_seed0" / "checkpoint.pt",
                            map_location="cpu", weights_only=True)
    model = build_sed(make_config(d, k, 90))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval().requires_grad_(False)


@beartype
def load_single_session_model(run: Path, device: str = "cuda") -> Sed:
    """Load a model trained by ``shared_training.py`` from its run directory."""
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    d, n = state["decoder.weight"].shape[:2]
    k = max(checkpoint["architecture"]["dsed_topk_map"].values())
    model = build_sed(make_config(d, k, n))
    model.load_state_dict(state)
    return model.to(device).eval().requires_grad_(False)


@beartype
def behavior_conditions(name: str, endpoints: np.ndarray, data: Optional[Path] = None) -> dict:
    """Return the paper's feature conditions ``{name: (mask, valid)}`` on one session.

    Quantile-based definitions (area, speed) take their quantiles from the session
    being scored, exactly as the paper does for session A.
    """
    behavior = pd.read_parquet((DATA[name] if data is None else data) / "behavior.parquet")
    return definitions(behavior, endpoints)[0]


@jaxtyped(typechecker=beartype)
def score_latents(
    activations: Float[np.ndarray, "window latent"],
    condition: Bool[np.ndarray, "window"],
    valid: Bool[np.ndarray, "window"],
) -> pd.DataFrame:
    """Score every latent against one condition with the paper's metrics.

    Returns:
        One row per latent with coverage (TPR), specificity (1 - FPR), selectivity
        TPR / (TPR + FPR) at native activity (z > 0) and the exact AUROC.
    """
    positive, negative = condition & valid, ~condition & valid
    active = activations > 0
    tpr = active[positive].mean(axis=0)
    fpr = active[negative].mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        selectivity = tpr / (tpr + fpr)
    values = torch.from_numpy(np.ascontiguousarray(activations[valid]))
    labels = torch.from_numpy(condition[valid])
    auroc = np.empty(activations.shape[1])
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    for latent in range(activations.shape[1]):
        column = values[:, latent]
        # Exact Mann-Whitney AUROC with ties: zeros share one average rank.
        nonzero = column > 0
        order = torch.argsort(column[nonzero])
        ranks = torch.empty(int(nonzero.sum()), dtype=torch.float64)
        sorted_values = column[nonzero][order]
        _, inverse, counts = torch.unique_consecutive(sorted_values, return_inverse=True, return_counts=True)
        ends = torch.cumsum(counts, 0).double()
        average = ends - (counts.double() - 1) / 2
        ranks[order] = average[inverse]
        n_zero = int((~nonzero).sum())
        rank_sum = (ranks[labels[nonzero]] + n_zero).sum() + (n_zero + 1) / 2 * int((labels & ~nonzero).sum())
        auroc[latent] = float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return pd.DataFrame(dict(latent=np.arange(activations.shape[1]), coverage=tpr, specificity=1 - fpr,
                             selectivity=selectivity, auroc=auroc))


@beartype
def score_one(values: np.ndarray, condition: np.ndarray, valid: np.ndarray) -> dict:
    """Score one latent with the exact functions used for the paper's Aeon table."""
    result = score(values, condition, valid)
    return dict(coverage=result["tpr"], specificity=1 - result["fpr"],
                selectivity=result["selectivity"], auroc=result["auroc"])


@torch.inference_mode()
def encode_endpoints(model: Sed, loader: WindowBatches, device: str = "cuda",
                     batch_size: int = 8192, transform=None) -> np.ndarray:
    """Return native sparse endpoint activations [window, latent] for every valid window."""
    d = model.cfg.n_features
    out = np.empty((len(loader.starts), d), dtype=np.float32)
    for offset in range(0, len(loader.starts), batch_size):
        indices = np.arange(offset, min(len(loader.starts), offset + batch_size))
        batch = loader.batch(indices)
        values = batch.values.to(device)
        if transform is not None:
            values = transform(values)
        result = model(values, occurrence_mask=batch.occurrence_mask.to(device))
        out[indices] = result.sparse_acts[d][:, -1].cpu().numpy()
    return out

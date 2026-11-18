"""Behavioral tuning profiles of latents, to test whether latent identities carry over.

A latent's profile is its activity-rate difference (inside minus outside) across a
fixed panel of behavior conditions. If an interpretation transfers, the profile of
shared latent ``i`` in the new session resembles its own session A profile more
than the session A profile of any other latent. Behavior is used only here.
"""

# ruff: noqa: F722, F821 -- jaxtyping dimension strings are not Python names.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from experiments.aeon.sweep_20260921.shared_analysis import Condition, build_conditions
from experiments.rebuttal.change02_multisession.common import CEPH, DATA, SESSION_A
from experiments.rebuttal.change02_multisession.screen_features import extra_conditions


@beartype
def condition_panel(session: str, endpoints: np.ndarray, data: Path | None = None) -> list[Condition]:
    """The fixed panel of behavior conditions, evaluated on one session."""
    data = DATA[session] if data is None else data
    behavior = pd.read_parquet(data / "behavior.parquet")
    pellets = pd.read_csv(data / "pellet_times.csv")
    return build_conditions(behavior, endpoints) + extra_conditions(behavior, pellets, endpoints)


@jaxtyped(typechecker=beartype)
def tuning_profile(
    codes: Float[np.ndarray, "window latent"], conditions: list[Condition], device: str = "cuda",
) -> Float[np.ndarray, "latent condition"]:
    """Return TPR - FPR at native activity for every latent and condition."""
    masks = np.stack([m for c in conditions for m in (c.mask & c.valid, c.valid & ~c.mask)])
    totals = torch.zeros((len(masks), codes.shape[1]), dtype=torch.float64, device=device)
    for start in range(0, len(codes), 262144):
        active = torch.from_numpy(codes[start:start + 262144] > 0).to(device=device, dtype=torch.float32)
        chunk = torch.from_numpy(masks[:, start:start + 262144]).to(device=device, dtype=torch.float32)
        totals += (chunk @ active).double()
    rates = (totals / torch.from_numpy(masks.sum(axis=1)).to(device).double().clamp_min(1)[:, None]).cpu().numpy()
    return np.ascontiguousarray((rates[0::2] - rates[1::2]).T)


@beartype
def session_a_profile(d: int, device: str = "cuda") -> tuple[np.ndarray, list[str]]:
    """Profile of the paper's session A model on session A (cached on Ceph)."""
    cache = CEPH / "eval" / f"profile_a_d{d}.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        return saved["profile"], saved["names"].tolist()
    run = SESSION_A / f"d{d}" / "k12_seed0"
    endpoints = np.load(run / "endpoint_bins.npy")
    conditions = condition_panel("A", endpoints)
    profile = tuning_profile(np.load(run / "activations.npy"), conditions, device)
    names = [c.name for c in conditions]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, profile=profile, names=np.array(names))
    return profile, names


@jaxtyped(typechecker=beartype)
def identity_transfer(
    reference: Float[np.ndarray, "latent condition"], transferred: Float[np.ndarray, "latent condition"],
    minimum_tuning: float = .1,
) -> dict:
    """Compare each shared latent's profile across sessions.

    Latents whose reference profile never exceeds ``minimum_tuning`` are left out,
    since an untuned latent has no interpretation to transfer.

    Returns:
        Summary with the median self-correlation, the median correlation with
        other latents, and how often a latent's own profile is its best match.
    """
    tuned = np.flatnonzero(np.abs(reference).max(axis=1) >= minimum_tuning)
    a = reference - reference.mean(axis=1, keepdims=True)
    b = transferred - transferred.mean(axis=1, keepdims=True)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    similarity = a[tuned] @ b.T  # [tuned reference latent, transferred latent]
    own = similarity[np.arange(len(tuned)), tuned]
    rank = (similarity > own[:, None]).sum(axis=1) + 1
    off = similarity.copy()
    off[np.arange(len(tuned)), tuned] = np.nan
    return dict(n_tuned=int(len(tuned)), self_correlation_median=float(np.median(own)),
                self_correlation_mean=float(own.mean()),
                other_correlation_median=float(np.nanmedian(off)),
                top1=float((rank == 1).mean()), top5=float((rank <= 5).mean()),
                median_rank=float(np.median(rank)),
                self_correlation={int(i): float(v) for i, v in zip(tuned, own)},
                self_rank={int(i): int(v) for i, v in zip(tuned, rank)})

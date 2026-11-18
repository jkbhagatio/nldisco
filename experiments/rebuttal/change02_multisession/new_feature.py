"""Describe a food-delivery latent of the session-B-only model.

The session B model is trained on neural activity alone. Pellet-delivery triggers
are read afterwards to describe one latent that is not among the paper's features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.rebuttal.change02_multisession.common import DATA, score_one

LAGS = np.arange(-150, 301)  # 20 ms bins: -3 s to +6 s around each trigger.


def peri_event(values: np.ndarray, endpoints: np.ndarray, event_bins: np.ndarray, n_bins: int) -> np.ndarray:
    """Return latent activity [event, lag] (NaN where the window is not neural-valid)."""
    row_of_bin = np.full(n_bins, -1, dtype=np.int64)
    row_of_bin[endpoints] = np.arange(len(endpoints))
    index = event_bins[:, None] + LAGS[None, :]
    inside = (index >= 0) & (index < n_bins)
    rows = np.where(inside, row_of_bin[np.clip(index, 0, n_bins - 1)], -1)
    out = np.full(index.shape, np.nan)
    out[rows >= 0] = values[rows[rows >= 0]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--latents", type=int, nargs="+", required=True)
    parser.add_argument("--window", type=float, nargs=2, default=[0., 1.], help="seconds after trigger")
    parser.add_argument("--session", default="B")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = DATA[args.session] if args.data is None else args.data
    endpoints = np.load(args.run / "endpoint_bins.npy")
    codes = np.load(args.run / "activations.npy", mmap_mode="r")
    pellets = pd.read_csv(data / "pellet_times.csv")
    n_bins = len(np.load(data / "neural_valid.npy", mmap_mode="r"))
    event_bins = np.floor(pellets.relative_seconds.to_numpy() / .02).astype(np.int64)
    lo, hi = (int(round(value / .02)) for value in args.window)
    mask = np.zeros(n_bins, dtype=bool)
    for b in event_bins:
        mask[max(0, b + lo):min(n_bins, b + hi)] = True
    condition, valid = mask[endpoints], np.ones(len(endpoints), dtype=bool)
    results, saved = {}, dict(lags_s=LAGS * .02)
    for latent in args.latents:
        values = np.asarray(codes[:, latent], dtype=np.float32)
        raster = peri_event(values, endpoints, event_bins, n_bins)
        active = np.where(np.isnan(raster), np.nan, raster > 0)
        in_window = active[:, (LAGS >= lo) & (LAGS < hi)]
        results[str(latent)] = dict(
            **score_one(values, condition, valid), window_s=args.window,
            n_events=int(len(event_bins)),
            events_with_activity=int((np.nansum(in_window, axis=1) > 0).sum()),
            active_windows=int((values > 0).sum()),
            active_windows_within_2s_after_trigger=int(((values > 0) & np.isin(
                endpoints, (event_bins[:, None] + np.arange(0, 100)[None, :]).ravel())).sum()),
            n_condition=int(condition.sum()), activity_rate=float((values > 0).mean()),
        )
        saved[f"probability_{latent}"] = np.nanmean(active, axis=0)
        saved[f"raster_{latent}"] = active.astype(np.float32)
    np.savez_compressed(args.output / "pellet_latents.npz", **saved)
    (args.output / "pellet_latents.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

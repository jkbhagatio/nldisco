"""Screen a model's latents against behavior conditions to find candidate features.

Adds pellet-delivery and arena-position conditions to the conditions used for the
paper's session A sweep. The model is fitted without behavior; this screen only
describes its latents afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.aeon.sweep_20260921.shared_analysis import (
    Condition, build_conditions, complete_support, continuous_auc, screen,
)
from experiments.rebuttal.change02_multisession.common import DATA


def extra_conditions(behavior: pd.DataFrame, pellets: pd.DataFrame, endpoints: np.ndarray) -> list[Condition]:
    """Pellet-delivery epochs and a coarse arena-position grid."""
    conditions = []
    n = len(behavior)
    everything = np.ones(len(endpoints), dtype=bool)
    bins = np.floor(pellets.relative_seconds.to_numpy() / .02).astype(np.int64)
    for name, lo, hi in [("pellet_0_1s_after", 0, 50), ("pellet_1_3s_after", 50, 150),
                         ("pellet_3_6s_after", 150, 300), ("pellet_1s_before", -50, 0)]:
        mask = np.zeros(n, dtype=bool)
        for b in bins:
            mask[max(0, b + lo):max(0, min(n, b + hi))] = True
        conditions.append(Condition(name, "pellet", mask[endpoints], everything,
                                    f"{lo * .02:g} to {hi * .02:g} s relative to a pellet trigger"))
    camera = behavior.camera_analysis_valid.to_numpy(dtype=bool)
    x, y = behavior.x.to_numpy(dtype=float), behavior.y.to_numpy(dtype=float)
    valid = complete_support(camera & np.isfinite(x) & np.isfinite(y))[endpoints]
    xe, ye = x[endpoints], y[endpoints]
    x_edges = np.linspace(np.nanquantile(xe[valid], .001), np.nanquantile(xe[valid], .999), 7)
    y_edges = np.linspace(np.nanquantile(ye[valid], .001), np.nanquantile(ye[valid], .999), 7)
    for i in range(6):
        for j in range(6):
            mask = valid & (xe >= x_edges[i]) & (xe < x_edges[i + 1]) & (ye >= y_edges[j]) & (ye < y_edges[j + 1])
            if mask.sum() >= 2000:
                conditions.append(Condition(
                    f"place_x{i}_y{j}", "place", mask, valid,
                    f"{x_edges[i]:.0f}<=x<{x_edges[i + 1]:.0f}, {y_edges[j]:.0f}<=y<{y_edges[j + 1]:.0f} px"))
    speed = behavior.speed.to_numpy(dtype=float)
    still = complete_support(camera & np.isfinite(speed) & (speed < 5), 50)[endpoints]
    conditions.append(Condition("still_1s", "immobility", still, complete_support(camera & np.isfinite(speed))[endpoints],
                                "centroid speed <5 px/s for the preceding 1 s"))
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--session", default="B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = DATA[args.session]
    codes = np.load(args.run / "activations.npy")
    endpoints = np.load(args.run / "endpoint_bins.npy")
    behavior = pd.read_parquet(data / "behavior.parquet")
    pellets = pd.read_csv(data / "pellet_times.csv")
    conditions = build_conditions(behavior, endpoints) + extra_conditions(behavior, pellets, endpoints)
    table = screen(codes, conditions)
    table.to_csv(args.output / "screen.csv", index=False)
    by_name = {c.name: c for c in conditions}
    top = (table[(table.tpr >= .5) & (table.n_condition >= 2000)]
           .sort_values("selectivity", ascending=False).groupby("condition").head(2).head(120).copy())
    top["auroc"] = [continuous_auc(codes[by_name[c].valid, l].astype(float),
                                   by_name[c].mask[by_name[c].valid]) for l, c in zip(top.latent, top.condition)]
    top["definition"] = [by_name[c].definition for c in top.condition]
    top.to_csv(args.output / "top_candidates.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60)
    print(top[["latent", "condition", "n_condition", "tpr", "fpr", "selectivity", "auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Zero-shot check: feed session B through the frozen session A model via unit matching."""

from __future__ import annotations

import json
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

from experiments.rebuttal.change02_multisession.common import (
    CEPH,
    POSITIONS,
    behavior_conditions,
    encode_endpoints,
    load_session,
    load_session_a_model,
    score_latents,
    score_one,
)


def assignment(sigma: Optional[float]) -> np.ndarray:
    """Row-stochastic [A unit, B unit] map; nearest neighbour when sigma is None."""
    a, b = pd.read_csv(POSITIONS["A"]), pd.read_csv(POSITIONS["B"])
    distance = np.hypot(a.x_um.values[:, None] - b.x_um.values[None], a.y_um.values[:, None] - b.y_um.values[None])
    if sigma is None:
        weights = np.zeros_like(distance)
        weights[np.arange(len(a)), distance.argmin(axis=1)] = 1.
        return weights
    weights = np.exp(-.5 * (distance / sigma) ** 2)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def main() -> None:
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    model = load_session_a_model(d)
    loader = load_session("B")
    endpoints = loader.starts + 19
    conditions = behavior_conditions("B", endpoints)
    targets = {192: {"wheel": 52}, 256: {"area": 107, "speed": 109, "direction": 10, "wheel": None}}[d]
    rows = []
    for label, sigma in [("nearest", None), ("gauss5", 5.), ("gauss10", 10.), ("shuffled", "shuffle")]:
        weights = assignment(None if sigma == "shuffle" else sigma)
        if sigma == "shuffle":
            weights = weights[np.random.default_rng(0).permutation(len(weights))]
        matrix = torch.from_numpy(weights).float().cuda()
        z = encode_endpoints(model, loader, transform=lambda x: x @ matrix.T)
        for name, latent in targets.items():
            condition, valid = conditions[name]
            table = score_latents(z, condition, valid)
            best = table.sort_values("auroc", ascending=False).iloc[0]
            row = dict(mapping=label, feature=name, best_latent=int(best.latent), best_auroc=float(best.auroc),
                       best_sel=float(best.selectivity), mean_l0=float((z > 0).sum(1).mean()))
            if latent is not None:
                row.update(latent=latent, **score_one(z[:, latent], condition, valid),
                           auroc_rank=int((table.auroc > table.auroc[latent]).sum()) + 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(CEPH / f"zero_shot_d{d}.csv", index=False)


if __name__ == "__main__":
    main()

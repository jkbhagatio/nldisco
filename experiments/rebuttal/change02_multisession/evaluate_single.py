"""Score a single-session model on the paper's feature conditions (best-matching latents)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.rebuttal.change02_multisession.common import (
    DATA,
    behavior_conditions,
    score_latents,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--session", default="B")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.run if args.output is None else args.output
    output.mkdir(parents=True, exist_ok=True)
    codes = np.load(args.run / "activations.npy")
    endpoints = np.load(args.run / "endpoint_bins.npy")
    conditions = behavior_conditions(args.session, endpoints, DATA[args.session] if args.data is None else args.data)
    tables, best = [], {}
    for name, (condition, valid) in conditions.items():
        table = score_latents(codes, condition, valid).assign(feature=name)
        tables.append(table)
        usable = table[(table.coverage > .5)]
        best[name] = dict(
            by_auroc=table.sort_values("auroc", ascending=False).iloc[0].drop("feature").to_dict(),
            by_selectivity_cov50=(usable.sort_values("selectivity", ascending=False).iloc[0].drop("feature").to_dict()
                                  if len(usable) else None),
            n_condition=int((condition & valid).sum()), n_valid=int(valid.sum()),
        )
    pd.concat(tables, ignore_index=True).to_csv(output / "feature_latent_scores.csv", index=False)
    (output / "feature_best_latents.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()

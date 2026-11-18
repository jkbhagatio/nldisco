"""Search a trained run's latents for stimulus tuning on the training split.

Example:
    python -m experiments.rebuttal.change04_allen.search --name s1_raw
"""

import argparse

import numpy as np
import pandas as pd

from experiments.rebuttal.change04_allen.data import CACHE
from experiments.rebuttal.change04_allen.features import search
from experiments.rebuttal.change04_allen.train import RUNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    acts = np.load(RUNS / args.name / "acts.npy")
    bins = pd.read_parquet(CACHE / "bins.parquet")
    path = RUNS / args.name / "search_train.parquet"
    table = pd.read_parquet(path) if path.exists() else search(acts, bins, "train")
    table.to_parquet(path)
    # Joint score favouring discriminable, selective latents with useful coverage.
    table["score"] = table.auroc + 0.5 * table.sel.fillna(0) + 0.5 * np.minimum(table["cov"], 0.5)
    pd.set_option("display.width", 200)
    for family, frame in table.groupby("family"):
        best = frame.sort_values("score", ascending=False).drop_duplicates("latent")
        print(f"\n== {family}")
        print(best.head(args.top).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

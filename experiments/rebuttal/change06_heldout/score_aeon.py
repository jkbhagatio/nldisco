"""Score the retrained Aeon models: select on training blocks, evaluate on held-out blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype

from experiments.aeon.sweep_20260921.paper_results import definitions
from experiments.rebuttal.change06_heldout import metrics, scoring

ROOT = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
FEATURES = {"wheel": 192, "area": 256, "speed": 256, "direction": 256}
WHEEL_TARGET_FPR = 0.18  # the paper's display operating point for the wheel feature


@beartype
def load_conditions() -> tuple:
    """The paper's fixed feature definitions on every valid endpoint of the full recording.

    Quantile cutoffs (area 101.5 to 512.5 px^2, speed >= 9.837 px/s) are the values printed in
    the paper; they are constants of the description, applied identically to both parts.
    """
    behavior = pd.read_parquet(ROOT / "data/behavior.parquet")
    endpoints = np.load(ROOT / "data/valid_starts.npy") + 19
    conditions, _ = definitions(behavior, endpoints)
    return endpoints, conditions


@beartype
def episode_groups(endpoints: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Episode ids for positive rows (positives within one second share an episode)."""
    groups = np.full(len(labels), -1, dtype=np.int64)
    positive = np.flatnonzero(labels)
    groups[positive] = metrics.episode_ids(np.ones(len(positive), bool), endpoints[positive])
    return groups


@beartype
def score_run(folder: Path, name: str, endpoints: np.ndarray, conditions: dict) -> dict:
    """Evaluate one feature for one trained model."""
    eval_bins = np.load(folder / "eval_endpoint_bins.npy")
    part = np.load(folder / "eval_part.npy")
    position = np.searchsorted(endpoints, eval_bins)
    np.testing.assert_array_equal(endpoints[position], eval_bins)
    mask, valid = conditions[name]
    mask, valid = mask[position], valid[position]
    latents = np.load(folder / "activations.npy", mmap_mode="r")
    rows = np.flatnonzero(valid)
    labels = mask[rows]
    values = np.asarray(latents[rows])
    groups = np.full(len(rows), -1, dtype=np.int64)
    for p in (0, 1):  # episodes are defined within each part
        chosen = part[rows] == p
        groups[chosen] = episode_groups(eval_bins[rows][chosen], labels[chosen])
    return scoring.evaluate_feature(values, labels, part[rows], groups,
                                    operating_fpr=WHEEL_TARGET_FPR if name == "wheel" else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    endpoints, conditions = load_conditions()
    results = {}
    for seed in args.seeds:
        for name, d in FEATURES.items():
            folder = args.runs / f"aeon_d{d}_seed{seed}"
            if not (folder / "COMPLETE.json").exists():
                continue
            results.setdefault(f"seed{seed}", {})[name] = score_run(folder, name, endpoints, conditions)
            print(seed, name, flush=True)
            args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

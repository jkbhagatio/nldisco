"""Score the retrained macaque models: select on training trials, evaluate on held-out trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from beartype import beartype

from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.rebuttal.change06_heldout import scoring

MAIN = Path("/nfs/nhome/live/jbhagat/nldisco")
DATA = MAIN / "experiments/churchland/data/nitschke_20090812"
SWEEP = Path("/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1")
FEATURES = ("recent_braking", "fast_target_specific", "early_leftward", "late_target_specific")


@beartype
def revised_labels(rows: np.ndarray) -> dict:
    """The paper's current (rounded-cutoff) labels for deceleration and fast leftward reach."""
    with np.load(SWEEP / "shared/index.npz") as index:
        anchors = index["anchor_index"][rows]
    with np.load(DATA / "metadata.npz") as meta:
        times = np.arange(5) * .05
        times -= times.mean()
        slope = meta["speed"][anchors[:, None] + np.arange(-4, 1)] @ times / (times @ times)
        vx, vy = meta["vel_x"][anchors], meta["vel_y"][anchors]
        return dict(recent_braking=slope <= -75,
                    fast_target_specific=(meta["speed"][anchors] >= 500) & (vx < 0)
                    & (np.abs(vx) >= np.abs(vy)) & (meta["target_x"][anchors] == -144)
                    & (meta["target_y"][anchors] == -52))


@beartype
def load_features() -> dict:
    """Labels for the four fixed paper descriptions on all training and held-out rows."""
    labels = decoder.load_labels(SWEEP, DATA, (0, 1, 2))
    for name in ("recent_braking", "fast_target_specific"):
        labels[name]["labels"] = revised_labels(labels[name]["rows"])[name]
    return labels


@beartype
def score_run(activations: np.ndarray, features: dict) -> dict:
    """Evaluate every feature for one trained model; held-out = validation plus test trials."""
    result = {}
    for name in FEATURES:
        f = features[name]
        part = np.where(f["split"] == 0, 0, 1).astype(np.int8)
        result[name] = scoring.evaluate_feature(np.asarray(activations[f["rows"]]), f["labels"],
                                                part, f["trials"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="folder with macaque_seed* runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    features = load_features()
    results = {}
    for run in sorted(args.runs.glob("macaque_seed*/summary.json")):
        folder = run.parent
        activations = np.load(folder / "activations_level_192.npy", mmap_mode="r")
        results[folder.name] = score_run(activations, features)
        print(folder.name, flush=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""ST-NDT sweep on the macaque recording of Section 4.2 (neural activity only).

Twelve settings are trained on the shared 191-channel, ten-bin endpoint windows used by
NLDisco, CEBRA-Time and LangevinFlow. `--mode sweep` fits every window (as the other
methods do) and exports the representations; `--mode convergence` fits the training
trials only and reports held-out masked prediction against a mean-rate baseline.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from beartype import beartype

from experiments.rebuttal.change01_stndt.stndt import Hyperparameters, train_and_export

MAIN = Path("/nfs/nhome/live/jbhagat/nldisco")
DATA = MAIN / "experiments/churchland/data/nitschke_20090812"
SWEEP = Path("/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1")
OUTPUT = Path("/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt")


@beartype
def configurations() -> list[dict]:
    """Return the twelve settings in stable worker order."""
    return [dict(config_id=f"lam{str(lam).replace('.', 'p')}_lr{str(lr).replace('.', 'p')}"
                           f"_drop{str(drop).replace('.', 'p')}",
                 contrast_weight=lam, learning_rate=lr, dropout=drop)
            for lam, lr, drop in itertools.product((.01, .1, 1.), (.001, .003), (.1, .3))]


@beartype
def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load counts, the shared endpoint windows and each window's trial partition."""
    counts = np.load(DATA / "counts.npy")
    with np.load(DATA / "metadata.npz") as metadata:
        trial, split = metadata["trial_id"], metadata["split"]
    with np.load(SWEEP / "shared/index.npz") as index:
        anchors, indices = index["anchor_index"], index["source_indices"]
    expected = np.arange(9, len(counts))
    expected = expected[trial[expected] == trial[expected - 9]]
    assert np.array_equal(anchors, expected) and len(anchors) == 130507
    assert np.array_equal(indices, anchors[:, None] + np.arange(-9, 1))
    assert np.array_equal(counts, np.round(counts)) and counts.shape[1] == 191
    return counts.astype(np.int64), anchors, indices, split[anchors]


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sweep", "convergence"), required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", help="run a single config_id")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    counts, anchors, indices, split = load_inputs()
    masks = (None, None) if args.mode == "sweep" else (split == 0, split != 0)
    for config in configurations()[args.worker::args.workers]:
        if args.only and config["config_id"] != args.only:
            continue
        hp = Hyperparameters(epochs=args.epochs, batch_size=args.batch_size,
                             **{k: v for k, v in config.items() if k != "config_id"})
        run = args.output / args.mode / config["config_id"]
        try:
            summary = train_and_export(
                counts, indices, run, hp, *masks, anchors=anchors, provenance=dict(
                    config_id=config["config_id"], mode=args.mode,
                    preprocessing="Raw counts; independent within-trial [t-9,...,t] windows",
                    counts_path=str(DATA / "counts.npy"),
                    index_path=str(SWEEP / "shared/index.npz")))
        except RuntimeError as error:  # a diverged setting is recorded, not hidden
            if "Nonfinite" not in str(error):
                raise
            summary = dict(complete=False, error=str(error))
            (run / "failure.json").write_text(json.dumps(summary) + "\n")
        print(json.dumps(dict(config_id=config["config_id"], **summary)), flush=True)


if __name__ == "__main__":
    main()

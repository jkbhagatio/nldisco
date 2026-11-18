"""Retrain the paper's macaque NLDisco-192 model on the training trials only.

Settings are those of ``experiments/churchland/sweeps/s10_endpoint_v1/nldisco.py``
(two-level TW-SED, D=192, k=16, ten epochs, B=1024, Adam 0.005, unit z-score inputs).
Normalization statistics, windows, the dead-feature schedule and the activation
thresholds are computed on training trials (split 0) only; every window of the
recording is then exported with the training-fitted model.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import DataLoader

from experiments.churchland.scripts.train_nldisco import prepare_inputs
from experiments.churchland.sweeps.s10_endpoint_v1.nldisco import configuration, export
from nldisco.config import LossConfig, TrainConfig
from nldisco.data import SpikeWindowDataset
from nldisco.model import build_sed
from nldisco.train import train_model
from nldisco.util import set_seed

MAIN = Path("/nfs/nhome/live/jbhagat/nldisco")
DATA = MAIN / "experiments/churchland/data/nitschke_20090812"
SWEEP = Path("/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1")
DIMENSION, BUDGET, LEVELS, BATCH_SIZE = 192, 16, 2, 1024


@beartype
def build_datasets() -> tuple:
    """Training-trial dataset and all-window dataset sharing training-only normalization."""
    with np.load(DATA / "metadata.npz") as archive:
        trials, split = archive["trial_id"], archive["split"]
    counts = np.load(DATA / "counts.npy").astype(np.float32)
    train_rows = split == 0
    # Statistics from training rows only; the held-out trials never enter them.
    _, scale, mean = prepare_inputs(counts[train_rows], trials[train_rows], 0.0, "unit_zscore")
    inputs = torch.from_numpy(((counts - mean) / scale).astype(np.float32))
    train_trials = np.unique(trials[train_rows])
    train = SpikeWindowDataset(inputs, seq_len=10, stride=1, trial_ids=trials,
                               occurrence_support=(9, 0), allowed_trial_ids=set(train_trials.tolist()))
    everything = SpikeWindowDataset(inputs, seq_len=10, stride=1, trial_ids=trials,
                                    occurrence_support=(9, 0))
    anchors = np.asarray(everything.valid_starts, dtype=np.int64) + 9
    with np.load(SWEEP / "shared/index.npz") as shared:
        np.testing.assert_array_equal(anchors, shared["anchor_index"])
    train_anchors = np.asarray(train.valid_starts, dtype=np.int64) + 9
    if not np.all(split[train_anchors] == 0):
        raise AssertionError("A training window touches a held-out trial")
    return train, everything, anchors, scale, mean


@beartype
def run(seed: int, output: Path) -> None:
    """Train, calibrate on training windows, export every window, verify the reload."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train, everything, anchors, scale, mean = build_datasets()
    set_seed(seed)
    cfg = configuration(everything.spike_counts.shape[1], DIMENSION, BUDGET, LEVELS)
    loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    loss_cfg = LossConfig(timebin_weights=[1.] * 10, type="mse")
    training = TrainConfig(epochs=10, batch_size=BATCH_SIZE, learning_rate=.005,
                           log_frequency=max(1, len(loader) // 4),
                           dead_feature_window=max(1, len(loader) // 3))
    config = dict(seed=seed, sed=asdict(cfg), loss=asdict(loss_cfg), training=asdict(training),
                  train_windows=len(train), all_windows=len(everything),
                  protocol="fit and normalization on split==0 trials only; export of all windows")
    config["sed"]["dtype"] = str(cfg.dtype)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    np.savez(output / "preprocessing.npz", scale=scale, mean=mean)
    model = build_sed(cfg).cuda()
    history = train_model(model, loader, loss_cfg, training)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config}, output / "final_model.pt")
    (output / "history.json").write_text(json.dumps(asdict(history), indent=2))
    metrics = export(model, everything, BATCH_SIZE, anchors, output)
    restored = build_sed(cfg).cuda()
    restored.load_state_dict(torch.load(output / "final_model.pt", weights_only=True)["state_dict"])
    restored.eval()
    probe = next(iter(DataLoader(everything, batch_size=32)))
    with torch.inference_mode():
        a = model(probe.values.cuda(), occurrence_mask=probe.occurrence_mask.cuda())
        b = restored(probe.values.cuda(), occurrence_mask=probe.occurrence_mask.cuda())
        for level in metrics:
            torch.testing.assert_close(a.sparse_acts[int(level)], b.sparse_acts[int(level)])
    (output / "summary.json").write_text(json.dumps(dict(
        status="complete", seed=seed, levels=metrics, seconds=time.monotonic() - started), indent=2))
    print(json.dumps({"event": "complete", "seed": seed, "seconds": time.monotonic() - started}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.seed, args.output)


if __name__ == "__main__":
    main()

"""Train an NLDisco SED on the training presentations of Allen session 732592105.

No stimulus labels enter training: the stimulus table only selects which bins
belong to the training split. The default configuration follows the model
evaluation figure (three Matryoshka levels, single 50-ms bins, B=1024).

Example:
    python -m experiments.rebuttal.change04_allen.train --name s1_d320 --device cuda:2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from beartype import beartype
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from experiments.rebuttal.change04_allen.data import CACHE, load_counts
from nldisco import EncoderConfig, LossConfig, SedConfig, SpikeWindowDataset, TrainConfig, build_sed
from nldisco.data.preprocessing import fit_normalizer
from nldisco.train import train_model

RUNS = CACHE.parent / "runs"
LEVELS = {128: 12, 224: 24, 320: 36}


@beartype
def training_rows(bins: pd.DataFrame) -> np.ndarray:
    """Bins usable for training: training split, outside NWB invalid intervals."""
    return ((bins.split == "train") & ~bins.invalid).to_numpy()


@beartype
def run(name: str, seq_len: int = 1, normalization: str = "none", epochs: int = 30,
        seed: int = 0, device: str = "cpu", levels: dict = LEVELS) -> Path:
    """Train, calibrate inference thresholds, and save activations for every bin."""
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    t.manual_seed(seed)
    np.random.seed(seed)
    counts, _ = load_counts()
    bins = pd.read_parquet(CACHE / "bins.parquet")
    train = training_rows(bins)
    stats = None if normalization == "none" else fit_normalizer(counts[train], normalization)
    x = t.from_numpy(counts if stats is None else stats.transform(counts).astype(np.float32))
    cfg = SedConfig(n_neurons=x.shape[1], seq_len=seq_len, dsed_topk_map=dict(levels),
                    encoder=EncoderConfig(type="FlatWindow"))
    if normalization != "none":
        cfg.decoder.output_activation = "none"
    loss = LossConfig(timebin_weights=[1.0] * seq_len, type="mse")
    train_cfg = TrainConfig(batch_size=1024, epochs=epochs, learning_rate=5e-3)
    dataset = SpikeWindowDataset(x, seq_len, 1, allowed_rows=train)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True,
                        generator=t.Generator().manual_seed(seed))
    model = build_sed(cfg).to(device)
    history = train_model(model, loader, loss, train_cfg)
    # Encode every complete window (anchor = final bin) with calibrated thresholds.
    model.eval()
    acts = np.zeros((len(x), cfg.n_features), np.float16)
    recon = np.zeros(x.shape, np.float32)
    full = SpikeWindowDataset(x, seq_len, 1)
    with t.no_grad():
        for batch in DataLoader(full, batch_size=8192):
            output = model(batch.values.to(device))
            rows = batch.anchor_index.numpy()
            acts[rows] = output.sparse_acts[cfg.n_features].float().cpu().numpy()
            recon[rows] = output.reconstructions[cfg.n_features][:, -1].float().cpu().numpy()
    np.save(out / "acts.npy", acts)
    t.save({"state_dict": model.state_dict(), "cfg": cfg}, out / "model.pt")
    valid = ~bins.invalid.to_numpy() & (bins.stim != "none").to_numpy()
    valid[: seq_len - 1] = False
    xs = x.numpy()
    summary = {"name": name, "seq_len": seq_len, "normalization": normalization,
               "epochs": epochs, "seed": seed, "levels": {str(k): v for k, v in levels.items()},
               "n_train_bins": int(train.sum()), "n_train_windows": len(dataset)}
    for split in ("train", "test"):
        rows = valid & (bins.split == split).to_numpy()
        a = acts[rows].astype(np.float32)
        summary[split] = {
            "r2": float(r2_score(xs[rows], recon[rows], multioutput="variance_weighted")),
            "l0": float((a > 0).sum(1).mean()),
            "dead": int(((a > 0).sum(0) == 0).sum()),
            "dense_gt_50pct": int(((a > 0).mean(0) > 0.5).sum()),
        }
    summary["final_loss"] = list(history.loss.values())[-1]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--normalization", default="none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.name, args.seq_len, args.normalization, args.epochs, args.seed, args.device)


if __name__ == "__main__":
    main()

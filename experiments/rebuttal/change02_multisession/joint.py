"""Exploratory: train one multi-session model jointly on several Aeon sessions from scratch.

Every session has its own input mapping and decoder; the temporal core and the
latents are shared and trained on all sessions together, with neural activity
only. With ``--tie position`` the sessions' mappings start from one random
mapping carried across sessions by probe position; with ``--tie none`` they start
independently. Behavior is read afterwards to ask whether a single shared latent
tracks the same feature in every session.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.aeon.sweep_20260921.shared_training import make_config
from experiments.rebuttal.change02_multisession.adapt import (
    FEATURES, GpuWindowBatches, export_session, unit_positions,
)
from experiments.rebuttal.change02_multisession.common import (
    CEPH, DATA, SEQ_LEN, behavior_conditions, load_session, score_latents,
)
from nldisco.config import LossConfig, MultiSessionSedConfig, SessionConfig, TrainConfig
from nldisco.model.multisession import build_multisession_sed, position_correspondence
from nldisco.train import train_multisession_model

SESSIONS = {
    "A": (DATA["A"], CEPH / "unit_positions" / "session_a.csv"),
    "B": (DATA["B"], DATA["B"] / "cluster_positions.csv"),
    "C": (CEPH / "more_sessions/session_c", CEPH / "more_sessions/session_c/cluster_positions.csv"),
    "D": (CEPH / "more_sessions/session_e", CEPH / "more_sessions/session_e/cluster_positions.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--d", type=int, default=192)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--tie", choices=["none", "position"], default="none")
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    loaders = {name: GpuWindowBatches(load_session("B", args.batch_size, args.steps, args.seed + i,
                                                   data=SESSIONS[name][0]), args.device)
               for i, name in enumerate(args.sessions)}
    single = make_config(args.d, args.k, 90)
    config = MultiSessionSedConfig(
        seq_len=SEQ_LEN, dsed_topk_map=dict(single.dsed_topk_map),
        sessions={name: SessionConfig(n_neurons=int(loader.counts.shape[1])) for name, loader in loaders.items()},
        encoder=single.encoder, decoder=single.decoder, inference_sparsity=single.inference_sparsity,
        dtype=single.dtype,
    )
    model = build_multisession_sed(config).to(args.device)
    if args.tie == "position":
        first = args.sessions[0]
        reference = unit_positions("B", SESSIONS[first][1])
        for name in args.sessions[1:]:
            positions = unit_positions("B", SESSIONS[name][1])
            model.init_session_from(
                name, first, position_correspondence(reference, positions, args.sigma).to(args.device),
                position_correspondence(positions, reference, args.sigma).to(args.device))
            for level in config.dsed_topk_map:  # Untrained thresholds stay uncalibrated.
                model.session(name).sparsifier.inference_threshold(level).fill_(float("nan"))
    loss_config = LossConfig(timebin_weights=[1.] * SEQ_LEN, type="mse", dead_feature_loss_weight=1.)
    train_config = TrainConfig(batch_size=args.batch_size, epochs=1, learning_rate=args.lr,
                               use_lr_schedule=True, log_frequency=50)
    histories = train_multisession_model(model, loaders, loss_config, train_config)
    (args.output / "training_history.json").write_text(json.dumps(
        {name: dataclasses.asdict(history) for name, history in histories.items()}))
    torch.save({"state_dict": model.state_dict(), "args": {k: str(v) for k, v in vars(args).items()}},
               args.output / "checkpoint.pt")
    tables, mse = [], {}
    for name, loader in loaders.items():
        codes, mse[name] = export_session(model, loader, args.device, session=name)
        conditions = behavior_conditions("B", np.asarray(loader.starts) + SEQ_LEN - 1, SESSIONS[name][0])
        for feature in FEATURES:
            condition, valid = conditions[feature]
            tables.append(score_latents(codes, condition, valid).assign(feature=feature, session=name))
        del codes
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.output / "latent_scores.csv", index=False)
    summary = dict(args={k: str(v) for k, v in vars(args).items()}, mse=mse, features={},
                   elapsed_seconds=time.monotonic() - started)
    for feature in FEATURES:
        wide = table[table.feature == feature].pivot(index="latent", columns="session", values="auroc")
        common = int(wide.min(axis=1).idxmax())
        summary["features"][feature] = dict(
            best_common_latent=common, auroc={name: float(wide.loc[common, name]) for name in wide.columns},
            best_per_session={name: (int(wide[name].idxmax()), float(wide[name].max())) for name in wide.columns},
            rank_of_common={name: int((wide[name] > wide.loc[common, name]).sum()) + 1 for name in wide.columns},
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

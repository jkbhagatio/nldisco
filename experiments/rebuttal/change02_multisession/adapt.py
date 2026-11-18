"""Adapt the paper's frozen session A model to a new session without behavior labels.

Session A's temporal core, shared latents, input mapping and decoder stay frozen.
The new session gets its own input mapping, decoder and a few session-specific
latents, trained only to reconstruct the new session's neural activity. Behavior
is read after training, for scoring alone.

Run one configuration from flags, or many in one process with ``--grid`` (a JSON
list of flag overrides), which loads the data once and keeps it on the GPU.
"""

# ruff: noqa: F821 -- jaxtyping dimension strings are not Python names.

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from beartype import beartype

from experiments.aeon.sweep_20260921.shared_training import WindowBatches
from experiments.rebuttal.change02_multisession.common import (
    DATA, POSITIONS, SEQ_LEN, behavior_conditions, load_session, load_session_a_model,
    score_latents, score_one,
)
from experiments.rebuttal.change02_multisession.profiles import (
    condition_panel, identity_transfer, session_a_profile, tuning_profile,
)
from nldisco.config import LossConfig, SessionConfig, TrainConfig
from nldisco.data.window import WindowSample
from nldisco.model.multisession import MultiSessionSed, position_correspondence
from nldisco.train import train_multisession_model

TARGETS = {192: {"wheel": 52}, 256: {"area": 107, "speed": 109, "direction": 10}}
MIN_CONDITION_WINDOWS = 2000  # Conditions rarer than this in either session are left out.
PAPER_WHEEL_THRESHOLD = 3.306849718093872  # Activation threshold of the paper's wheel scores.
FEATURES = ("wheel", "area", "speed", "direction")


class GpuWindowBatches(WindowBatches):
    """``WindowBatches`` with the recording held on the GPU; identical batches, no host I/O."""

    def __init__(self, loader: WindowBatches, device: str):
        super().__init__(loader.counts, loader.starts, loader.mean, loader.std,
                         loader.batch_size, loader.steps, loader.seed)
        self.device = device
        self.gpu_counts = torch.from_numpy(np.asarray(loader.counts).astype(np.int16)).to(device)
        self.gpu_starts = torch.from_numpy(np.asarray(loader.starts)).to(device)
        self.gpu_mean = torch.from_numpy(loader.mean).to(device)
        self.gpu_std = torch.from_numpy(loader.std).to(device)
        self.offsets = torch.arange(SEQ_LEN, device=device)

    def fresh(self, steps: int, seed: int, batch_size: int) -> "GpuWindowBatches":
        """Reset the sampler for a new run without reloading the recording."""
        self.steps, self.seed, self.batch_size = steps, seed, batch_size
        self.iterations = 0
        self.seen[:] = 0
        return self

    def batch(self, indices: np.ndarray) -> WindowSample:
        identity = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        source = self.gpu_starts[identity, None] + self.offsets
        values = (self.gpu_counts[source].float() - self.gpu_mean) / self.gpu_std
        occurrence = torch.zeros(source.shape, dtype=torch.bool, device=self.device)
        occurrence[:, -1] = True
        missing = torch.full_like(identity, -1)
        return WindowSample(values, identity, source, occurrence, source[:, -1], missing, missing)


@beartype
def unit_positions(name: str, path: Path | None = None) -> torch.Tensor:
    """Probe coordinates [unit, 2] in um, in the column order of ``counts.npy``."""
    frame = pd.read_csv(POSITIONS[name] if path is None else path)
    return torch.from_numpy(frame[["x_um", "y_um"]].to_numpy(dtype=np.float32))


@beartype
def build_adapted_model(
    d: int, n_units: int, n_session_latents: int, session_topk: int, init: str, sigma: float,
    positions: torch.Tensor | None = None, device: str = "cuda", seed: int = 0,
) -> MultiSessionSed:
    """Return the frozen session A model with a trainable session B attached."""
    torch.manual_seed(seed)
    model = MultiSessionSed.from_sed(load_session_a_model(d, device=device), "A")
    model.add_session("B", SessionConfig(n_neurons=n_units, n_session_features=n_session_latents,
                                         session_topk=session_topk))
    if init in ("position", "shuffled"):
        positions = unit_positions("B") if positions is None else positions
        reference = unit_positions("A")
        if init == "shuffled":  # Same weights with unit identities scrambled.
            positions = positions[torch.from_numpy(np.random.default_rng(seed).permutation(len(positions)))]
        model.init_session_from(
            "B", "A", position_correspondence(reference, positions, sigma).to(device),
            position_correspondence(positions, reference, sigma).to(device),
        )
    elif init != "random":
        raise ValueError(f"Unknown initialisation: {init}")
    model.freeze_shared(["A"])
    return model


@torch.inference_mode()
def export_session(model: MultiSessionSed, loader: WindowBatches, device: str = "cuda",
                   session: str = "B") -> tuple[np.ndarray, dict]:
    """Encode every valid window; return native sparse endpoint codes and reconstruction MSE."""
    sed = model.session(session).eval()
    d = sed.cfg.n_features
    n = len(loader.starts)
    codes = np.empty((n, d), dtype=np.float32)
    levels = sorted(sed.cfg.dsed_topk_map)
    error = {level: 0. for level in levels}
    for offset in range(0, n, 8192):
        indices = np.arange(offset, min(n, offset + 8192))
        batch = loader.batch(indices)
        target = batch.values.to(device)
        result = sed(target, occurrence_mask=batch.occurrence_mask.to(device))
        codes[indices] = result.sparse_acts[d][:, -1].cpu().numpy()
        for level in levels:
            error[level] += float((target - result.reconstructions[level]).double().square().sum())
    size = n * SEQ_LEN * sed.cfg.n_neurons
    return codes, {str(level): error[level] / size for level in levels}


@beartype
def score_run(codes: np.ndarray, conditions: dict, d: int, n_shared: int) -> tuple[pd.DataFrame, list[dict]]:
    """Score all latents on the paper's feature conditions; summarise the paper's latents."""
    tables, rows = [], []
    for name in FEATURES:
        latent = TARGETS[d].get(name)
        if latent is None and name != "wheel":
            continue
        condition, valid = conditions[name]
        table = score_latents(codes, condition, valid).assign(feature=name)
        tables.append(table)
        shared = table.iloc[:n_shared]
        best = table.sort_values("auroc", ascending=False).iloc[0]
        row = dict(feature=name, best_latent=int(best.latent), best_latent_auroc=float(best.auroc),
                   best_latent_selectivity=float(best.selectivity))
        if latent is not None:
            others = shared.drop(index=latent)
            row.update(
                latent=latent, **score_one(codes[:, latent], condition, valid),
                auroc_rank_among_shared=int((shared.auroc > shared.auroc[latent]).sum()) + 1,
                other_shared_auroc_median=float(others.auroc.median()),
                other_shared_auroc_p95=float(others.auroc.quantile(.95)),
                other_shared_auroc_max=float(others.auroc.max()),
                other_shared_selectivity_median=float(others.selectivity.median()),
                other_shared_selectivity_p95=float(others.selectivity.quantile(.95)),
                activity_rate=float((codes[:, latent] > 0).mean()),
            )
            if name == "wheel":
                thresholded = np.where(codes[:, latent] >= PAPER_WHEEL_THRESHOLD, codes[:, latent], 0.)
                at = score_one(thresholded, condition, valid)
                row.update({f"{key}_at_paper_threshold": at[key]
                            for key in ("coverage", "specificity", "selectivity")})
        rows.append(row)
    return pd.concat(tables, ignore_index=True), rows


DEFAULTS = dict(d=192, k=12, init="position", sigma=1., n_session_latents=16, session_topk=2,
                steps=5000, batch_size=1024, lr=.005, seed=0, save_activations=False)


@beartype
def adapt(config: dict, output: Path, loader: WindowBatches, conditions: dict,
          positions: torch.Tensor | None = None, device: str = "cuda",
          panel: list | None = None) -> dict:
    """Run one adaptation, write its checkpoint and scores, and return the summary."""
    cfg = {**DEFAULTS, **config}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    np.random.seed(cfg["seed"])
    model = build_adapted_model(cfg["d"], loader.counts.shape[1], cfg["n_session_latents"],
                                cfg["session_topk"], cfg["init"], cfg["sigma"], positions,
                                device, cfg["seed"])
    reference = {name: value.clone() for name, value in model.state_dict().items()
                 if not name.startswith("sessions.B.")}
    n_shared = model.cfg.n_shared_features
    sed = model.session("B")
    summary = dict(config=cfg, n_units=int(loader.counts.shape[1]), n_shared_latents=n_shared,
                   levels={str(key): value for key, value in sed.cfg.dsed_topk_map.items()})
    if cfg["steps"]:
        loss_config = LossConfig(timebin_weights=[1.] * SEQ_LEN, type="mse", dead_feature_loss_weight=1.)
        train_config = TrainConfig(batch_size=cfg["batch_size"], epochs=1, learning_rate=cfg["lr"],
                                   use_lr_schedule=True, log_frequency=50)
        history = train_multisession_model(model, {"B": loader}, loss_config, train_config)["B"]
        (output / "training_history.json").write_text(json.dumps(dataclasses.asdict(history)))
    elif sed.cfg.n_features != n_shared:
        # No training: keep session A's threshold and silence the session-specific latents.
        with torch.no_grad():
            sed.encoder.session_projection.weight.zero_()
            sed.encoder.session_projection.bias.fill_(-1.)
            sed.sparsifier.inference_threshold(sed.cfg.n_features).copy_(
                sed.sparsifier.inference_threshold(n_shared))
    state = model.state_dict()
    if not all(torch.equal(value, state[name]) for name, value in reference.items()):
        raise AssertionError("Frozen session A parameters changed during adaptation")
    torch.save({"state_dict": state, "config": cfg}, output / "checkpoint.pt")
    codes, mse = export_session(model, loader, device)
    if cfg["save_activations"]:
        np.save(output / "activations.npy", codes)
        np.save(output / "endpoint_bins.npy", np.asarray(loader.starts) + SEQ_LEN - 1)
    table, rows = score_run(codes, conditions, cfg["d"], n_shared)
    table.to_csv(output / "latent_scores.csv", index=False)
    if panel is not None:
        profile, sizes = tuning_profile(codes, panel, device)
        np.save(output / "tuning_profile.npy", profile)
        reference, reference_sizes = session_a_profile(cfg["d"], device)
        keep = (sizes >= MIN_CONDITION_WINDOWS) & (reference_sizes >= MIN_CONDITION_WINDOWS)
        transfer = identity_transfer(np.ascontiguousarray(reference[:, keep]),
                                     np.ascontiguousarray(profile[:n_shared, keep]))
        transfer["n_conditions"] = int(keep.sum())
        (output / "identity_transfer.json").write_text(json.dumps(transfer, indent=2))
        summary["identity_transfer"] = {key: value for key, value in transfer.items()
                                        if not isinstance(value, dict)}
    active = codes > 0
    summary.update(
        mse=mse, mean_l0=float(active.sum(axis=1).mean()),
        mean_l0_shared=float(active[:, :n_shared].sum(axis=1).mean()),
        dead_shared_latents=int((active[:, :n_shared].sum(axis=0) == 0).sum()),
        thresholds={str(level): float(sed.sparsifier.inference_threshold(level))
                    for level in sed.cfg.dsed_topk_map},
        features=rows, elapsed_seconds=time.monotonic() - started,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="run directory, or parent directory of the grid's runs")
    parser.add_argument("--grid", type=Path, default=None,
                        help="JSON list of {name, ...overrides}; runs sequentially in one process")
    parser.add_argument("--data", type=Path, default=DATA["B"])
    parser.add_argument("--positions", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    for key, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()
    torch.set_num_threads(2)
    base = {key: getattr(args, key) for key in DEFAULTS}
    loader = GpuWindowBatches(load_session("B", data=args.data), args.device)
    conditions = behavior_conditions("B", np.asarray(loader.starts) + SEQ_LEN - 1, args.data)
    panel = condition_panel("B", np.asarray(loader.starts) + SEQ_LEN - 1, args.data)
    positions = None if args.positions is None else unit_positions("B", args.positions)
    runs = [dict(name=None)] if args.grid is None else json.loads(args.grid.read_text())
    for run in runs:
        run = dict(run)
        name = run.pop("name")
        output = args.output if name is None else args.output / name
        if name is not None and (output / "summary.json").exists():
            continue
        cfg = {**base, **run}
        loader.fresh(max(cfg["steps"], 1), cfg["seed"], cfg["batch_size"])
        summary = adapt(cfg, output, loader, conditions, positions, args.device, panel)
        line = dict(name=name, **{key: cfg[key] for key in ("d", "init", "sigma", "n_session_latents",
                                                             "session_topk", "steps", "lr", "seed")},
                    mse=summary["mse"], mean_l0_shared=summary["mean_l0_shared"],
                    dead_shared=summary["dead_shared_latents"], features=summary["features"],
                    identity=summary.get("identity_transfer"))
        print("RESULT " + json.dumps(line), flush=True)


if __name__ == "__main__":
    main()

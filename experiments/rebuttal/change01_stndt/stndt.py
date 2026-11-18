"""Dataset-agnostic training and export for the pinned, unmodified ST-NDT model.

The model (`src/model.py`) and its masking (`src/mask.py`) are imported unchanged from
the vendored upstream source. This adapter only supplies windows, the optimisation
loop (one update per minibatch) and the export of the upstream "features": the output
of the final transformer layer, taken at the last bin of each window.

Usage (any binned recording; see `--help`):

    $STNDT_PY -m experiments.rebuttal.change01_stndt.stndt \
        --counts counts.npy --index index.npz --output NEW_DIRECTORY \
        [--train-mask train.npy] [--validation-mask heldout.npy] [hyperparameters]

`counts.npy` is a (time bins, units) array of spike counts. `index.npz` holds
`source_indices`, a (windows, bins per window) integer array of rows of `counts`;
each row is one window in time order whose last entry is the bin the exported vector
is aligned to. Optional masks are boolean arrays over windows: the model is trained on
`train-mask` windows only (default all), held-out masked prediction is reported on
`validation-mask` windows, and representations are exported for every window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from beartype import beartype
from einops import reduce
from jaxtyping import Bool, Float, Int, jaxtyped

from experiments.rebuttal.change01_stndt.environments.setup_stndt import (
    REVISION,
    SOURCE,
    source_digests,
    verify_source,
)

sys.path.insert(0, str(SOURCE))
from src.config.default import get_cfg_defaults  # noqa: E402
from src.mask import UNMASKED_LABEL, Masker  # noqa: E402
from src.model import STNDT  # noqa: E402


@dataclass(frozen=True)
class Hyperparameters:
    """Settings layered on the upstream MC_Maze configuration (`configs/mc_maze.yaml`)."""

    layers: int = 4
    heads: int = 1
    hidden: int = 128
    dropout: float = .1
    contrast_weight: float = .1
    learning_rate: float = 1e-3
    warmup_fraction: float = .1
    epochs: int = 10
    batch_size: int = 256
    seed: int = 0


@beartype
def make_config(hp: Hyperparameters):
    """Return the upstream MC_Maze configuration with this run's overrides."""
    config = get_cfg_defaults()
    config.merge_from_file(str(SOURCE / "configs/mc_maze.yaml"))
    config.merge_from_list([
        "MODEL.NUM_LAYERS", hp.layers, "MODEL.NUM_HEADS", hp.heads,
        "MODEL.HIDDEN_SIZE", hp.hidden, "MODEL.DROPOUT", hp.dropout,
        "MODEL.DROPOUT_RATES", hp.dropout, "MODEL.DROPOUT_EMBEDDING", hp.dropout,
        "MODEL.LAMBDA", hp.contrast_weight, "MODEL.FULL_CONTEXT", True,
        "TRAIN.LR.INIT", hp.learning_rate, "TRAIN.BATCH_SIZE", hp.batch_size,
        "SEED", hp.seed])
    config.freeze()
    return config


@beartype
def make_model(units: int, bins: int, hp: Hyperparameters, device: str,
               max_spikes: int) -> STNDT:
    """Instantiate the unmodified upstream model; its width equals the unit count."""
    if units % hp.heads or bins % hp.heads:
        raise ValueError(f"heads={hp.heads} must divide units={units} and bins={bins}: "
                         "upstream attention widths are the unit and bin counts.")
    return STNDT(make_config(hp).MODEL, bins, units, torch.device(device),
                 max_spikes=max_spikes).to(device)


@jaxtyped(typechecker=beartype)
def masked(masker: Masker, spikes: Int[torch.Tensor, "batch bins units"], contrast: bool,
           max_spikes: int) -> tuple[Int[torch.Tensor, "batch bins units"],
                                     Int[torch.Tensor, "batch bins units"]]:
    """Apply upstream masking; empty held-out and forward parts keep its code path."""
    batch, bins, units = spikes.shape
    result = masker.mask_batch(
        spikes, contrast_mode=contrast, max_spikes=max_spikes, should_mask=True,
        expand_prob=0., heldout_spikes=spikes.new_zeros(batch, bins, 0),
        forward_spikes=spikes.new_zeros(batch, 0, units))
    return result[0], result[1]


@jaxtyped(typechecker=beartype)
def objective(model: STNDT, maskers: tuple[Masker, Masker],
              spikes: Int[torch.Tensor, "batch bins units"], max_spikes: int
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream training objective: masked Poisson NLL plus weighted contrastive loss."""
    inputs, labels = masked(maskers[0], spikes, False, max_spikes)
    views = [masked(maskers[1], spikes, True, max_spikes)[0] for _ in range(2)]
    loss, nll, contrast, _ = model(inputs, labels, contrast_src1=views[0],
                                   contrast_src2=views[1], val_phase=False, rates=None,
                                   return_outputs=True, return_weights=False)
    return loss.mean(), nll.mean(), contrast.mean()


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def extract(model: STNDT, counts: Int[torch.Tensor, "time units"],
            indices: Int[np.ndarray, "windows bins"], batch_size: int
            ) -> Float[np.ndarray, "windows units"]:
    """Export the final transformer layer's output at the last bin of each window."""
    model.eval()
    result = np.empty((len(indices), counts.shape[1]), np.float32)
    for start in range(0, len(indices), batch_size):
        spikes = counts[torch.as_tensor(indices[start:start + batch_size],
                                        device=counts.device)]
        layers = model(spikes, spikes, contrast_src1=None, contrast_src2=None,
                       val_phase=True, passthrough=True, return_outputs=True,
                       return_weights=True)[5]  # batch, bins, units, layers
        result[start:start + len(spikes)] = layers[:, -1, :, -1].cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite exported representation")
    return result


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def masked_prediction(model: STNDT, masker: Masker,
                      counts: Int[torch.Tensor, "time units"],
                      indices: Int[np.ndarray, "windows bins"],
                      log_mean_rate: Float[torch.Tensor, " units"], batch_size: int,
                      max_spikes: int, seed: int = 0) -> dict:
    """Score masked-bin prediction against each unit's mean training rate.

    Returns the Poisson NLL per masked entry for the model and for the mean-rate
    baseline, and their difference in bits per masked spike (positive: model better).
    """
    model.eval()
    cuda = counts.is_cuda
    state = torch.random.get_rng_state(), torch.cuda.get_rng_state_all() if cuda else None
    torch.manual_seed(seed)
    sums = torch.zeros(4, dtype=torch.float64, device=counts.device)
    for start in range(0, len(indices), batch_size):
        spikes = counts[torch.as_tensor(indices[start:start + batch_size],
                                        device=counts.device)]
        inputs, labels = masked(masker, spikes, False, max_spikes)
        log_rate = model(inputs, labels, contrast_src1=None, contrast_src2=None,
                         val_phase=True, return_outputs=False, return_weights=False)[3]
        chosen = labels != UNMASKED_LABEL
        target = spikes[chosen].double()
        baseline = log_mean_rate.expand_as(log_rate)[chosen].double()
        predicted = log_rate[chosen].double()
        sums += torch.stack([(predicted.exp() - target * predicted).sum(),
                             (baseline.exp() - target * baseline).sum(),
                             target.sum(), chosen.sum().double()])
    torch.random.set_rng_state(state[0])
    if cuda:
        torch.cuda.set_rng_state_all(state[1])
    model_nll, baseline_nll, spikes_sum, entries = sums.tolist()
    return dict(masked_nll=model_nll / entries, mean_rate_nll=baseline_nll / entries,
                bits_per_spike=(baseline_nll - model_nll) / spikes_sum / math.log(2),
                masked_entries=int(entries))


@beartype
def warmup_cosine(step: int, warmup: int, total: int) -> float:
    """Learning-rate factor of the upstream `WarmupCosineSchedule` (half a cosine)."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0., .5 * (1. + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


@beartype
def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


@jaxtyped(typechecker=beartype)
def train_and_export(counts: Int[np.ndarray, "time units"],
                     indices: Int[np.ndarray, "windows bins"], output: Path,
                     hp: Hyperparameters,
                     train_mask: Optional[Bool[np.ndarray, " windows"]] = None,
                     validation_mask: Optional[Bool[np.ndarray, " windows"]] = None,
                     device: str = "cuda", anchors: Optional[np.ndarray] = None,
                     provenance: Optional[dict] = None) -> dict:
    """Train on the training windows and export one vector per window.

    Args:
        counts: Spike counts, bins by units; no behavior enters this function.
        indices: Rows of `counts` forming each window, in time order.
        output: New directory for the checkpoint, representation and records.
        hp: Model and optimisation settings.
        train_mask: Windows used for training (default: all).
        validation_mask: Windows scored for held-out masked prediction each epoch.
        device: Torch device.
        anchors: Optional row of `counts` each window is aligned to (default: last bin).
        provenance: Extra entries recorded in `config.json`.

    Returns:
        The summary written to `summary.json`.
    """
    verify_source()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    train = indices if train_mask is None else indices[train_mask]
    held_out = None if validation_mask is None else indices[validation_mask]
    if train_mask is not None and validation_mask is not None and (
            train_mask & validation_mask).any():
        raise ValueError("Training and validation windows overlap")
    train_rows = np.unique(train)
    max_spikes = int(counts[train_rows].max())
    values = torch.as_tensor(counts, device=device).clamp(max=max_spikes).long()
    log_mean_rate = torch.as_tensor(
        reduce(counts[train_rows].astype(np.float64), "time units -> units", "mean"),
        device=device).clamp_min(1e-7).log().float()
    units, bins = counts.shape[1], indices.shape[1]
    model = make_model(units, bins, hp, device, max_spikes)
    config = make_config(hp)
    maskers = Masker(config.TRAIN, torch.device(device)), Masker(config.TRAIN,
                                                               torch.device(device))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=hp.learning_rate,
                                  weight_decay=config.TRAIN.WEIGHT_DECAY,
                                  eps=config.TRAIN.EPS)
    steps_per_epoch = math.ceil(len(train) / hp.batch_size)
    total = hp.epochs * steps_per_epoch
    warmup = round(hp.warmup_fraction * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, warmup, total))
    record = dict(asdict(hp), source_revision=REVISION, source_sha256=source_digests(),
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  upstream_config="configs/mc_maze.yaml", units=units, bins=bins,
                  dimensions=units, windows=len(indices), train_windows=len(train),
                  validation_windows=0 if held_out is None else len(held_out),
                  max_spikes=max_spikes, total_updates=total, warmup_updates=warmup,
                  parameters=sum(p.numel() for p in parameters),
                  extraction="Final transformer layer output at the last bin of each window",
                  gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                  torch=torch.__version__, **(provenance or {}))
    write_json(output / "config.json", record)
    history = []
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    start = time.monotonic()
    for epoch in range(hp.epochs):
        model.train()
        order = np.random.permutation(len(train))
        sums = np.zeros(3)
        epoch_start = time.monotonic()
        for offset in range(0, len(order), hp.batch_size):
            spikes = values[torch.as_tensor(train[order[offset:offset + hp.batch_size]],
                                            device=device)]
            loss, nll, contrast = objective(model, maskers, spikes, max_spikes)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite objective at epoch {epoch}, batch {offset}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.TRAIN.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            sums += np.array([loss.item(), nll.item(), contrast.item()]) * len(spikes)
        if cuda:
            torch.cuda.synchronize()
        means = sums / len(train)
        entry = dict(epoch=epoch, objective=means[0], masked_nll=means[1],
                     contrast=means[2], learning_rate=scheduler.get_last_lr()[0],
                     epoch_seconds=time.monotonic() - epoch_start)
        if held_out is not None:
            entry["validation"] = masked_prediction(
                model, maskers[0], values, held_out, log_mean_rate, hp.batch_size,
                max_spikes)
        history.append(entry)
        write_json(output / "progress.json", dict(epochs_completed=epoch + 1,
                                                  history=history))
        print(json.dumps(dict(output=output.name, **entry)), flush=True)
    if cuda:
        torch.cuda.synchronize()
    train_seconds = sum(entry["epoch_seconds"] for entry in history)  # excludes validation
    torch.save(dict(state_dict=model.state_dict(), config=record), output / "model_final.pt")
    extraction_start = time.monotonic()
    activation = extract(model, values, indices, hp.batch_size)
    np.save(output / "activations.npy", activation)
    np.savez(output / "index.npz", source_indices=indices,
             anchor_index=indices[:, -1] if anchors is None else anchors,
             train_mask=np.ones(len(indices), bool) if train_mask is None else train_mask,
             validation_mask=(np.zeros(len(indices), bool) if validation_mask is None
                              else validation_mask))
    restored = make_model(units, bins, hp, device, max_spikes)
    restored.load_state_dict(torch.load(output / "model_final.pt", weights_only=False,
                                        map_location=device)["state_dict"])
    actual = extract(restored, values, indices[:hp.batch_size], hp.batch_size)
    np.testing.assert_allclose(actual, activation[:hp.batch_size], rtol=1e-5, atol=1e-5)
    summary = dict(complete=True, epochs_completed=hp.epochs, windows=len(indices),
                   dimensions=int(activation.shape[1]),
                   seconds=time.monotonic() - start, train_seconds=train_seconds,
                   extraction_and_reload_seconds=time.monotonic() - extraction_start,
                   final_masked_nll=history[-1]["masked_nll"],
                   final_contrast=history[-1]["contrast"],
                   validation=history[-1].get("validation"),
                   reload_max_abs=float(np.abs(actual - activation[:hp.batch_size]).max()),
                   allocated_peak_bytes=torch.cuda.max_memory_allocated() if cuda else None)
    write_json(output / "summary.json", summary)
    return summary


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=Path, required=True,
                        help="(time bins, units) spike-count .npy")
    parser.add_argument("--index", type=Path, required=True,
                        help=".npz with source_indices (windows, bins per window)")
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--train-mask", type=Path, help="boolean .npy over windows")
    parser.add_argument("--validation-mask", type=Path, help="boolean .npy over windows")
    parser.add_argument("--device", default="cuda")
    for name, value in asdict(Hyperparameters()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    hp = Hyperparameters(**{name: getattr(args, name) for name in asdict(Hyperparameters())})
    counts = np.load(args.counts)
    if not np.array_equal(counts, np.round(counts)):
        raise ValueError("Counts must be integers")
    with np.load(args.index) as index:
        indices = index["source_indices"].astype(np.int64)
        anchors = index["anchor_index"] if "anchor_index" in index.files else None
    load = lambda path: None if path is None else np.load(path).astype(bool)  # noqa: E731
    summary = train_and_export(
        counts.astype(np.int64), indices, args.output, hp, load(args.train_mask),
        load(args.validation_mask), args.device, anchors,
        dict(counts_path=str(args.counts), index_path=str(args.index),
             train_mask_path=None if args.train_mask is None else str(args.train_mask),
             validation_mask_path=(None if args.validation_mask is None
                                   else str(args.validation_mask))))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

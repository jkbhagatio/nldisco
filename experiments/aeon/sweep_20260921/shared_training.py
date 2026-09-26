"""Matched-budget, streamed TW-SED training for the strict Aeon 18 h sweep."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from einops import reduce
from jaxtyping import Float, Integer, jaxtyped

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig, TrainConfig
from nldisco.data.window import WindowSample
from nldisco.model.sed import build_sed
from nldisco.train import train_model


@beartype
def write_json(path: Path, value: dict) -> None:
    """Write strict JSON, rejecting silent nonfinite scalar exports."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@beartype
def make_config(d: int, k: int, n_neurons: int = 90) -> SedConfig:
    """Construct the shared one-layer endpoint-code architecture."""
    if d % 2 or k % 2:
        raise ValueError("Matryoshka dimensions and K must be even")
    return SedConfig(
        n_neurons=n_neurons, seq_len=20, dsed_topk_map={d // 2: k // 2, d: k},
        encoder=EncoderConfig(
            type="TransformerWindow", shift_equivariant=True, d_model=128,
            n_heads=4, n_layers=1, d_feedforward=512, dropout=0., causal=True,
            attention_radius=19,
        ),
        decoder=DecoderConfig(
            temporal_kernel_len=20, temporal_alignment="causal", output_activation="none",
        ),
        inference_sparsity="training_threshold", dtype=torch.float32,
    )


@jaxtyped(typechecker=beartype)
def compute_statistics(
    counts: Integer[np.ndarray, "time neuron"], valid: np.ndarray,
) -> tuple[Float[np.ndarray, "neuron"], Float[np.ndarray, "neuron"], int]:  # noqa: F821
    """Compute population mean/std over every valid neural bin, without smoothing."""
    total = np.zeros(counts.shape[1], np.float64)
    squared = np.zeros_like(total)
    n = 0
    for start in range(0, len(counts), 100_000):
        values = np.asarray(counts[start:start + 100_000][valid[start:start + 100_000]],
                            dtype=np.float64)
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        n += len(values)
    if not n:
        raise ValueError("No valid neural bins")
    mean = total / n
    std = np.sqrt(np.maximum(squared / n - mean ** 2, 0.))
    std[std == 0.] = 1.
    return mean.astype(np.float32), std.astype(np.float32), n


class WindowDatasetView:
    """Expose individual windows for validation without advancing the batch sampler."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader.starts)

    def __getitem__(self, index: int) -> WindowSample:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        batch = self.loader.batch(np.asarray([index % len(self)], dtype=np.int64))
        return WindowSample(*(field[0] for field in batch))


@beartype
class WindowBatches:
    """Vectorized memmap batches; one train pass then one all-window calibration pass.

    Training walks independently shuffled complete permutations, never discarding
    the end of a permutation. Calibration visits each eligible window exactly once
    in a separately seeded shuffle. Only one [B,20,N] float tensor is materialized.
    """

    def __init__(self, counts: np.ndarray, starts: np.ndarray, mean: np.ndarray,
                 std: np.ndarray, batch_size: int, steps: int, seed: int):
        self.counts, self.starts = counts, starts
        self.mean, self.std = mean, std
        self.batch_size, self.steps, self.seed = batch_size, steps, seed
        self.dataset = WindowDatasetView(self)
        self.iterations = 0
        self.seen = np.zeros(len(starts), dtype=np.uint16)

    def __len__(self) -> int:
        return self.steps

    def batch(self, indices: np.ndarray) -> WindowSample:
        source = self.starts[indices, None] + np.arange(20)[None, :]
        values = (self.counts[source].astype(np.float32) - self.mean) / self.std
        occurrence = np.zeros((len(indices), 20), dtype=bool)
        occurrence[:, -1] = True
        identity = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        source_tensor = torch.from_numpy(source)
        return WindowSample(
            torch.from_numpy(values), identity, source_tensor,
            torch.from_numpy(occurrence), source_tensor[:, -1],
            torch.full_like(identity, -1), torch.full_like(identity, -1),
        )

    def __iter__(self):
        self.iterations += 1
        rng = np.random.default_rng(self.seed + (1 if self.iterations > 1 else 0))
        order = rng.permutation(len(self.starts))
        if self.iterations > 1:
            for offset in range(0, len(order), self.batch_size):
                yield self.batch(order[offset:offset + self.batch_size])
            return
        cursor = 0
        for _ in range(self.steps):
            pieces, remaining = [], self.batch_size
            while remaining:
                take = min(remaining, len(order) - cursor)
                pieces.append(order[cursor:cursor + take])
                cursor += take
                remaining -= take
                if cursor == len(order):
                    order, cursor = rng.permutation(len(self.starts)), 0
            indices = np.concatenate(pieces)
            np.add.at(self.seen, indices, 1)
            yield self.batch(indices)


@beartype
def export(model: torch.nn.Module, loader: WindowBatches, output: Path) -> dict:
    """Stream all native sparse and prethreshold codes plus reconstruction diagnostics."""
    model.eval()
    d = model.cfg.n_features
    n = len(loader.starts)
    sparse = np.lib.format.open_memmap(output / "activations.npy", mode="w+",
                                      dtype=np.float32, shape=(n, d))
    dense = np.lib.format.open_memmap(output / "prethreshold_activations.npy", mode="w+",
                                     dtype=np.float32, shape=(n, d))
    endpoints = loader.starts + 19
    np.save(output / "endpoint_bins.npy", endpoints)
    np.save(output / "decoder_kernels.npy", model.decoder.weight.detach().cpu().numpy())
    levels = sorted(model.cfg.dsed_topk_map)
    error_sum = {level: np.zeros(20) for level in levels}
    target_sum = np.zeros((20, model.cfg.n_neurons))
    target_square = np.zeros_like(target_sum)
    active_count = np.zeros(d, dtype=np.int64)
    device = next(model.parameters()).device
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, n, loader.batch_size):
            indices = np.arange(offset, min(n, offset + loader.batch_size))
            batch = loader.batch(indices)
            target = batch.values.to(device)
            result = model(target, occurrence_mask=batch.occurrence_mask.to(device))
            native = result.sparse_acts[d][:, -1].cpu().numpy()
            raw = result.acts[:, -1].cpu().numpy()
            if not np.isfinite(native).all() or not np.isfinite(raw).all():
                raise ValueError("Nonfinite activation export")
            sparse[indices], dense[indices] = native, raw
            active_count += (native > 0).sum(axis=0)
            values = target.double()
            target_sum += values.sum(0).cpu().numpy()
            target_square += values.square().sum(0).cpu().numpy()
            for level in levels:
                residual = (target - result.reconstructions[level]).double()
                if not torch.isfinite(residual).all():
                    raise ValueError("Nonfinite reconstruction")
                error_sum[level] += reduce(residual.square(), "b s n -> s", "sum").cpu().numpy()
            if offset % (100 * loader.batch_size) == 0:
                print(f"export {offset}/{n} windows; elapsed {time.monotonic()-started:.1f}s",
                      flush=True)
    sparse.flush()
    dense.flush()
    sst = np.maximum(target_square - target_sum ** 2 / n, 0.).sum(axis=1)
    metrics = []
    for level in levels:
        for lag in range(20):
            metrics.append({"level": level, "lag": lag - 19,
                            "mse": error_sum[level][lag] / (n * model.cfg.n_neurons),
                            "variance_weighted_r2": 1. - error_sum[level][lag] / max(sst[lag], 1e-30)})
    pd.DataFrame(metrics).to_csv(output / "reconstruction_by_lag.csv", index=False)
    return {
        "n_windows": n, "activity_counts": active_count.tolist(),
        "dead_latents": np.flatnonzero(active_count == 0).tolist(),
        "mean_l0": float(active_count.sum() / n),
        "thresholds": {str(level): float(model.sparsifier.inference_threshold(level))
                       for level in levels},
        "mse": {str(level): float(error_sum[level].sum() / (n * 20 * model.cfg.n_neurons))
                for level in levels},
        "activation_semantics": "activations.npy = native calibrated sparse amplitudes; primary AUROC uses this continuous-valued output",
        "prethreshold_semantics": "prethreshold_activations.npy = ReLU encoder endpoint before native threshold; sensitivity only",
        "row_alignment": "row i endpoint = valid_starts[i] + 19; chronological all-valid windows",
    }


@beartype
def run(args: argparse.Namespace) -> None:
    """Train, calibrate, verify reload, and export one independently initialized run."""
    data, output = Path(args.data), Path(args.output)
    if not (data / "READY.json").is_file():
        raise RuntimeError("Root QC release READY.json is required before training")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly the assigned GPU UUID")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.gpu_memory_gb * 2**30 / total, 1.), 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    counts = np.load(data / "counts.npy", mmap_mode="r")
    starts = np.load(data / "valid_starts.npy", mmap_mode="r")
    valid = np.load(data / "neural_valid.npy", mmap_mode="r")
    if not len(starts) or np.any(np.diff(starts) <= 0):
        raise ValueError("Valid starts must be nonempty and strictly chronological")
    mean, std, n_valid = compute_statistics(counts, valid)
    np.savez(output / "normalization.npz", mean=mean, std=std, n_valid=n_valid)
    config = make_config(args.d, args.k, counts.shape[1])
    loss_config = LossConfig(timebin_weights=[1.] * 20, type="mse",
                             dsed_loss_weight_map={args.d // 2: 1., args.d: 1.},
                             dead_feature_loss_weight=1.)
    train_config = TrainConfig(batch_size=args.batch_size, epochs=1, learning_rate=.005,
                               use_lr_schedule=True, log_frequency=50)
    loader = WindowBatches(counts, starts, mean, std, args.batch_size, args.steps, args.seed)
    architecture = dataclasses.asdict(config)
    architecture["dtype"] = str(architecture["dtype"])
    config_record = {"architecture": architecture, "loss": dataclasses.asdict(loss_config),
                     "training": dataclasses.asdict(train_config), "args": vars(args),
                     "data_ready": json.loads((data / "READY.json").read_text()),
                     "sample_policy": "seeded complete permutations; 5000 exact updates; calibration full independent permutation",
                     "analysis_scope": "full-dataset descriptive discovery, no held-out/decoding claim",
                     "clusters": "all 90 raw KS4 clusters (23 good, 67 mua); anatomy unknown",
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    write_json(output / "config.json", config_record)
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    sources = [Path(__file__)]
    repo = Path(__file__).resolve().parents[3]
    sources += list((repo / "src" / "nldisco" / "model").glob("*.py"))
    sources += [repo / "src" / "nldisco" / "train.py", repo / "src" / "nldisco" / "config.py"]
    hashes = {}
    for source in sources:
        shutil.copy2(source, source_dir / source.name)
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "source_hashes.json", hashes)
    model = build_sed(config).to(device)
    result = train_model(model, loader, loss_config, train_config)
    write_json(output / "training_history.json", dataclasses.asdict(result))
    torch.save({"state_dict": model.state_dict(), "architecture": architecture,
                "seed": args.seed, "steps": args.steps}, output / "checkpoint.pt")
    exposure = {"presentations": int(loader.seen.sum()), "unique_windows": int((loader.seen > 0).sum()),
                "eligible_windows": len(starts), "coverage": float((loader.seen > 0).mean()),
                "minimum_visits": int(loader.seen.min()), "maximum_visits": int(loader.seen.max()),
                "calibration_windows": len(starts), "calibration_batches": math.ceil(len(starts) / args.batch_size)}
    np.save(output / "training_window_visits.npy", loader.seen)
    probe = loader.batch(np.arange(min(32, len(starts))))
    model.eval()
    with torch.inference_mode():
        expected = model(probe.values.to(device)).sparse_acts[args.d].cpu()
    reloaded = build_sed(config)
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    reloaded.load_state_dict(checkpoint["state_dict"])
    reloaded = reloaded.to(device).eval()
    with torch.inference_mode():
        observed = reloaded(probe.values.to(device)).sparse_acts[args.d].cpu()
    if not torch.equal(expected, observed):
        raise AssertionError("Checkpoint reload did not reproduce native sparse activations")
    del model
    metrics = export(reloaded, loader, output)
    metrics.update({"sampler_exposure": exposure, "checkpoint_reload_exact": True,
                    "elapsed_seconds": time.monotonic() - started,
                    "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.,
                    "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0.})
    write_json(output / "metrics.json", metrics)
    write_json(output / "COMPLETE.json", {"d": args.d, "k": args.k, "seed": args.seed,
                                          "elapsed_seconds": metrics["elapsed_seconds"]})
    print(json.dumps(metrics, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-gb", type=float, default=8.)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""PCA baseline: principal components of the z-scored, flattened 20-bin window."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped

from experiments.aeon.sweep_20260921 import shared_training as shared
from experiments.rebuttal.change03_discovery import common, data


@jaxtyped(typechecker=beartype)
def flat_windows(values: Float[torch.Tensor, "time neurons"],
                 starts: Int[torch.Tensor, "batch"]) -> Float[torch.Tensor, "batch features"]:
    """Materialize [t-19, ..., t] for each window and flatten time and neurons."""
    index = starts[:, None] + torch.arange(data.SEQ_LEN, device=values.device)
    return rearrange(values[index], "b t n -> b (t n)")


@beartype
def run(args: argparse.Namespace) -> None:
    """Fit on fit windows (exact covariance), project every window."""
    output = common.new_output(Path(args.output))
    started = time.monotonic()
    device = torch.device(args.device)
    common.cap_gpu_memory(args.gpu_memory_gb)
    session = data.load_neural(Path(args.data))
    mean, std, _ = shared.compute_statistics(session.counts, data.fit_bins(session))
    values = torch.as_tensor((np.asarray(session.counts, dtype=np.float32) - mean) / std,
                             device=device)
    fit = torch.as_tensor(session.starts[session.part == data.FIT], device=device)
    features = data.SEQ_LEN * values.shape[1]
    total = torch.zeros(features, dtype=torch.float64, device=device)
    gram = torch.zeros(features, features, dtype=torch.float64, device=device)
    for offset in range(0, len(fit), args.batch_size):
        x = flat_windows(values, fit[offset:offset + args.batch_size]).double()
        total += x.sum(0)
        gram += x.T @ x
    center = total / len(fit)
    covariance = gram / len(fit) - torch.outer(center, center)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    width = min(args.d, features)
    order = torch.argsort(eigenvalues, descending=True)[:width]
    components = eigenvectors[:, order]
    # Fix each component's sign so that its largest-magnitude loading is positive.
    components = components * torch.sign(
        components[components.abs().argmax(0), torch.arange(width, device=device)])
    representation = common.open_representation(output, len(session.starts), width)
    everything = torch.as_tensor(session.starts, device=device)
    for offset in range(0, len(everything), args.batch_size):
        x = flat_windows(values, everything[offset:offset + args.batch_size]).double()
        representation[offset:offset + len(x)] = ((x - center) @ components).float().cpu().numpy()
    representation.flush()
    explained = (eigenvalues[order] / eigenvalues.clamp_min(0).sum()).cpu().numpy()
    np.savez(output / "pca.npz", components=components.cpu().numpy(),
             center=center.cpu().numpy(), explained_variance_ratio=explained,
             mean=mean, std=std)
    common.write_json(output / "summary.json", dict(
        method="pca", width=width, fit_windows=int(len(fit)), all_windows=len(session.starts),
        explained_variance=float(explained.sum()), elapsed_seconds=time.monotonic() - started,
        preprocessing="z-scored counts (fit-portion statistics), 20 bins x units flattened"))
    common.write_json(output / "COMPLETE.json", {"elapsed_seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d", type=int, default=common.LATENTS)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--gpu-memory-gb", type=float, default=8.)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

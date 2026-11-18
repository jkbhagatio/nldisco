"""CEBRA-Time on Aeon windows: the Section 4.2 recipe with a 20-bin receptive field."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped
from torch import nn

from experiments.rebuttal.change03_discovery import common, data

CONFIGURATIONS = {  # learning rates compared; temperature and lag as selected in Section 4.2
    "lr1e-4": dict(learning_rate=1e-4), "lr3e-4": dict(learning_rate=3e-4),
    "lr1e-3": dict(learning_rate=1e-3)}
TEMPERATURE, LAG, HIDDEN = 1., 1, 128


@beartype
def encoder(neurons: int, dimension: int) -> nn.Module:
    """CEBRA offset encoder with eight residual blocks: a 20-bin receptive field.

    Same layers as the official `offset10-model` (which has three residual blocks
    and a 10-bin receptive field): 2 + 8 * 2 + 2 = 20 bins.
    """
    import cebra.data
    from cebra.models import layers as cebra_layers
    from cebra.models.model import ConvolutionalModelMixin, _OffsetModel

    class Offset20Model(_OffsetModel, ConvolutionalModelMixin):
        def __init__(self):
            super().__init__(
                nn.Conv1d(neurons, HIDDEN, 2), nn.GELU(),
                *[cebra_layers._Skip(nn.Conv1d(HIDDEN, HIDDEN, 3), nn.GELU()) for _ in range(8)],
                nn.Conv1d(HIDDEN, dimension, 3),
                num_input=neurons, num_output=dimension, normalize=True)

        def get_offset(self):
            return cebra.data.Offset(10, 10)

    return Offset20Model()


@jaxtyped(typechecker=beartype)
def windows(values: Float[torch.Tensor, "time neurons"], starts: Int[torch.Tensor, "batch"],
            ) -> Float[torch.Tensor, "batch neurons 20"]:
    """Materialize exactly [t-19, ..., t] in encoder channel-first order."""
    index = starts[:, None] + torch.arange(data.SEQ_LEN, device=values.device)
    return rearrange(values[index], "b t c -> b c t").contiguous()


@beartype
def transform(model: nn.Module, values: torch.Tensor, starts: np.ndarray,
              out: np.ndarray, batch_size: int = 8192) -> None:
    """Write one unit-norm embedding per window, in window order."""
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            index = torch.as_tensor(starts[offset:offset + batch_size], device=values.device)
            out[offset:offset + len(index)] = model(windows(values, index)).cpu().numpy()


@beartype
def run(args: argparse.Namespace) -> None:
    """Train one configuration on fit windows, score validation windows, export all."""
    from cebra.models import FixedCosineInfoNCE
    output = common.new_output(Path(args.output))
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    common.cap_gpu_memory(args.gpu_memory_gb)
    session = data.load_neural(Path(args.data))
    root = np.sqrt(np.asarray(session.counts, dtype=np.float32))
    fit_bins = data.fit_bins(session)
    mean, std = root[fit_bins].mean(0), root[fit_bins].std(0)
    std[std == 0] = 1.
    values = torch.as_tensor((root - mean) / std, device=device)
    del root
    starts = torch.as_tensor(session.starts, device=device)
    config = dict(CONFIGURATIONS[args.config], config_id=args.config, dimension=args.d,
                  temperature=TEMPERATURE, lag=LAG, hidden_units=HIDDEN, seed=args.seed,
                  steps=args.steps, batch_size=args.batch_size, optimizer="Adam",
                  architecture="CEBRA offset encoder, 8 residual blocks, 20-bin receptive field",
                  objective="FixedCosineInfoNCE",
                  preprocessing="sqrt counts; unit-wise standardization, fit-portion statistics",
                  sampling="uniform fit references; positive = window LAG bins later in the "
                           "same fit segment; uniform fit negatives")
    common.write_json(output / "config.json", config)
    pools = {}
    for name, part in (("fit", data.FIT), ("validation", data.VALIDATION)):
        member = session.part == part
        pools[name] = (torch.as_tensor(data.lagged_pairs(session.starts, member, LAG), device=device),
                       torch.as_tensor(np.flatnonzero(member), device=device))
    model = encoder(values.shape[1], args.d).to(device)
    objective = FixedCosineInfoNCE(temperature=TEMPERATURE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    def loss_on(pool: str, generator: torch.Generator | None = None) -> torch.Tensor:
        references, members = pools[pool]
        pick = lambda n: torch.randint(n, (args.batch_size,), device=device, generator=generator)
        reference = references[pick(len(references))]
        index = torch.cat((reference, reference + LAG, members[pick(len(members))]))
        embedded = model(windows(values, starts[index]))
        return objective(*embedded.chunk(3))[0]

    def validation_loss() -> float:
        generator = torch.Generator(device=device).manual_seed(1234)
        model.eval()
        with torch.inference_mode():
            result = float(torch.stack([loss_on("validation", generator)
                                        for _ in range(args.validation_batches)]).mean())
        model.train()
        return result

    history, validation = [], []
    model.train()
    train_started = time.monotonic()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on("fit")
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if (step + 1) % args.validation_every == 0 or step == args.steps - 1:
            validation.append(dict(step=step + 1, loss=validation_loss()))
            print(args.config, step + 1, np.mean(history[-100:]), validation[-1], flush=True)
    train_seconds = time.monotonic() - train_started
    torch.save(dict(state_dict=model.state_dict(), config=config, mean=mean, std=std),
               output / "model.pt")
    representation = common.open_representation(output, len(session.starts), args.d)
    transform(model, values, session.starts, representation)
    representation.flush()
    norms = np.linalg.norm(representation[:: max(len(representation) // 10000, 1)], axis=1)
    if not np.allclose(norms, 1, atol=1e-4):
        raise ValueError("CEBRA outputs must be unit normalized")
    common.write_json(output / "summary.json", dict(
        method="cebra", config_id=args.config, validation_loss=validation[-1]["loss"],
        validation_history=validation, train_loss_first_100=float(np.mean(history[:100])),
        train_loss_last_100=float(np.mean(history[-100:])), train_seconds=train_seconds,
        elapsed_seconds=time.monotonic() - started, fit_references=int(len(pools["fit"][0])),
        validation_references=int(len(pools["validation"][0]))))
    np.save(output / "objective_history.npy", np.asarray(history, dtype=np.float32))
    common.write_json(output / "COMPLETE.json", {"elapsed_seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", choices=tuple(CONFIGURATIONS), required=True)
    parser.add_argument("--d", type=int, default=common.LATENTS)
    parser.add_argument("--steps", type=int, default=common.STEPS)
    parser.add_argument("--batch-size", type=int, default=common.BATCH_SIZE)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-gb", type=float, default=8.)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

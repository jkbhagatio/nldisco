"""LangevinFlow on Aeon windows: the Section 4.2 adapter extended to 20 bins.

Run with the LangevinFlow overlay interpreter
(`experiments/churchland/environments/langevinflow-venv/bin/python` of the main checkout).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from einops import rearrange

from experiments.churchland.sweeps.s10_endpoint_v1.langevinflow import EndpointLangevin
from experiments.rebuttal.change03_discovery import common, data

CONFIGURATIONS = {f"lr{str(lr).replace('.', 'p')}_ramp{ramp}": dict(learning_rate=lr, ramp=ramp)
                  for lr in (.001, .003) for ramp in (5, 500)}
HIDDEN = 64  # position, momentum and GRU state of 64 dimensions each: 192 latents
PSEUDO_EPOCHS = 10  # the Section 4.2 schedules are defined over ten epochs


class WindowLangevin(EndpointLangevin):
    """Upstream equations over a window of any length; every bin is consumed once."""

    def forward(self, observ: torch.Tensor, use_logrates: bool = True):
        hidden = self.dropout(self.encoder(observ[:, 0]))
        z_mu, z_logvar = self.linear_z_means(hidden), self.linear_z_logvar(hidden)
        v_mu, v_logvar = self.linear_v_means(hidden), self.linear_v_logvar(hidden)
        z, v = self.reparameterize(z_mu, z_logvar), self.reparameterize(v_mu, v_logvar)
        latents = [torch.cat((z, v, hidden), dim=1)]
        kl = self.kl_gauss(z_mu, z_logvar) + self.kl_gauss(v_mu, v_logvar)
        for j in range(1, observ.shape[1]):
            hidden = self.dropout(self.encoder(observ[:, j], hidden))
            z = z.clone().requires_grad_()
            force = torch.autograd.grad(self.potential(z, observ[:, j]).sum(), z,
                                        create_graph=True)[0]
            z, v = z + self.step * v, v - self.step * force
            v = self.reparameterize((1 - self.gamma) * v,
                                   math.sqrt(2 * self.gamma) * torch.ones_like(v), False)
            latents.append(torch.cat((z, v, hidden), dim=1))
            kl = kl + self.kl_gauss(v, 2 * self.gamma * torch.ones_like(v), var2=.1)
        latent = torch.stack(latents, dim=1)
        prediction = rearrange(self.readout(self.decoder(rearrange(latent, "b t h -> t b h"))),
                               "t b n -> b t n")
        return (prediction if use_logrates else prediction.exp()), latent, kl


@beartype
def make_model(neurons: int, learning_rate: float, device: str,
               hidden: int = HIDDEN) -> WindowLangevin:
    """Upstream architecture with the Section 4.2 hyperparameters."""
    return WindowLangevin(input_size=neurons, output_size=neurons, hidden_size=hidden,
                          fwd_steps=0, learning_rate=learning_rate, weight_decay=2e-5,
                          dropout=.05, gamma=.55, cd_rate=.5).to(device)


@beartype
def masked_nll(model: WindowLangevin, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordinated-dropout Poisson NLL (the method's own data term) and KL."""
    masked, mask = model.cd.process_batch(values)
    prediction, _, kl = model(masked)
    nll = torch.nn.functional.poisson_nll_loss(prediction, values, reduction="none")
    return model.cd.process_losses(nll, mask).mean(), kl


@beartype
def extract(model: WindowLangevin, counts: torch.Tensor, starts: np.ndarray,
            out: np.ndarray, batch_size: int = 4096) -> None:
    """Write the deterministic endpoint [q, p, h] for every window."""
    model.eval()
    model.posterior_mean = True
    steps = torch.arange(data.SEQ_LEN, device=counts.device)
    for offset in range(0, len(starts), batch_size):
        index = torch.as_tensor(starts[offset:offset + batch_size], device=counts.device)
        with torch.enable_grad():
            _, latent, _ = model(counts[index[:, None] + steps])
        out[offset:offset + len(index)] = latent[:, -1].detach().cpu().numpy()
    model.posterior_mean = False


@beartype
def run(args: argparse.Namespace) -> None:
    """Train one configuration on fit windows, score validation windows, export all."""
    output = common.new_output(Path(args.output))
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    common.cap_gpu_memory(args.gpu_memory_gb)
    session = data.load_neural(Path(args.data))
    counts = torch.as_tensor(np.asarray(session.counts, dtype=np.float32), device=device)
    steps_in_window = torch.arange(data.SEQ_LEN, device=device)
    fit = session.starts[session.part == data.FIT]
    validation = session.starts[session.part == data.VALIDATION]
    config = dict(CONFIGURATIONS[args.config], config_id=args.config, hidden_size=HIDDEN,
                  seed=args.seed, steps=args.steps, batch_size=args.batch_size,
                  pseudo_epochs=PSEUDO_EPOCHS, gamma=.55, step=.01, dropout=.05, cd_rate=.5,
                  target_kl=.1, target_weight_decay=2e-5,
                  preprocessing="raw counts; independent [t-19, ..., t] windows",
                  extraction="endpoint q, p, h with zero Gaussian innovations")
    common.write_json(output / "config.json", config)
    model = make_model(counts.shape[1], config["learning_rate"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.95, patience=10, threshold=0., min_lr=1e-5)
    per_epoch = math.ceil(args.steps / PSEUDO_EPOCHS)
    rng = np.random.default_rng(args.seed)
    order, cursor = rng.permutation(len(fit)), 0

    def validation_loss() -> float:
        model.eval()
        generator_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(1234)
        pick = np.random.default_rng(1234).choice(
            len(validation), size=min(len(validation), args.validation_windows), replace=False)
        total, n = 0., 0
        for offset in range(0, len(pick), 4096):
            index = torch.as_tensor(validation[pick[offset:offset + 4096]], device=device)
            with torch.enable_grad():
                nll, _ = masked_nll(model, counts[index[:, None] + steps_in_window])
            total, n = total + float(nll.detach()) * len(index), n + len(index)
        torch.random.set_rng_state(generator_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train()
        return total / n

    history, validation_history = [], []
    train_started = time.monotonic()
    model.train()
    for step in range(args.steps):
        epoch = step // per_epoch
        ramp = min(epoch / config["ramp"], 1.)
        optimizer.param_groups[0]["weight_decay"] = 2e-5 * ramp
        if cursor + args.batch_size > len(order):
            order, cursor = rng.permutation(len(fit)), 0
        index = torch.as_tensor(fit[order[cursor:cursor + args.batch_size]], device=device)
        cursor += args.batch_size
        optimizer.zero_grad(set_to_none=True)
        nll, kl = masked_nll(model, counts[index[:, None] + steps_in_window])
        loss = nll + .1 * ramp * kl
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite objective at step {step}")
        loss.backward()
        optimizer.step()
        history.append((float(loss.detach()), float(nll.detach()), float(kl.detach())))
        if (step + 1) % per_epoch == 0 or step == args.steps - 1:
            scheduler.step(float(np.mean([h[1] for h in history[-per_epoch:]])))
            validation_history.append(dict(step=step + 1, masked_poisson_nll=validation_loss()))
            print(args.config, step + 1, np.mean(history[-per_epoch:], axis=0),
                  validation_history[-1], f"{time.monotonic() - train_started:.0f}s", flush=True)
    train_seconds = time.monotonic() - train_started
    torch.save(dict(state_dict=model.state_dict(), config=config), output / "model.pt")
    representation = common.open_representation(output, len(session.starts), 3 * HIDDEN)
    extract(model, counts, session.starts, representation)
    representation.flush()
    if not np.isfinite(representation[::1000]).all():
        raise ValueError("Nonfinite endpoint means")
    np.save(output / "objective_history.npy", np.asarray(history, dtype=np.float32))
    common.write_json(output / "summary.json", dict(
        method="langevinflow", config_id=args.config,
        validation_loss=validation_history[-1]["masked_poisson_nll"],
        validation_history=validation_history,
        train_nll_first_100=float(np.mean([h[1] for h in history[:100]])),
        train_nll_last_100=float(np.mean([h[1] for h in history[-100:]])),
        train_seconds=train_seconds, elapsed_seconds=time.monotonic() - started))
    common.write_json(output / "COMPLETE.json", {"elapsed_seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", choices=tuple(CONFIGURATIONS), required=True)
    parser.add_argument("--steps", type=int, default=common.STEPS)
    parser.add_argument("--batch-size", type=int, default=common.BATCH_SIZE)
    parser.add_argument("--validation-windows", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-gb", type=float, default=12.)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

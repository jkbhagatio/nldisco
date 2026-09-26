"""Pinned LangevinFlow with observation-aligned, independent ten-bin windows."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "environments/LangevinFlow_CCN"
REVISION = "68dcbfaa78b2371851842333ae933eea1f961832"
PROCESS_PEAK_BYTES = 0
sys.path.insert(0, str(SOURCE))
from nlb_lightning.models import LangevinAutoencoder  # noqa: E402


@beartype
def configurations() -> list[dict]:
    """Return the twelve approved settings in stable worker order."""
    return [dict(config_id=f"h{h}_lr{str(lr).replace('.', 'p')}_ramp{r}",
                 hidden_size=h, learning_rate=lr, ramp=r)
            for h, lr, r in itertools.product((32, 64, 128), (.001, .003), (5, 500))]


class EndpointLangevin(LangevinAutoencoder):
    """Preserve upstream equations while consuming each observation exactly once."""

    posterior_mean = False

    def reparameterize(self, mu, log_var, logvar=True):
        return mu if self.posterior_mean else super().reparameterize(mu, log_var, logvar)

    @jaxtyped(typechecker=beartype)
    def forward(self, observ: Float[torch.Tensor, "batch 10 neurons"],
                use_logrates: bool = True):
        hidden = self.dropout(self.encoder(observ[:, 0]))
        z_mu, z_logvar = self.linear_z_means(hidden), self.linear_z_logvar(hidden)
        v_mu, v_logvar = self.linear_v_means(hidden), self.linear_v_logvar(hidden)
        z, v = self.reparameterize(z_mu, z_logvar), self.reparameterize(v_mu, v_logvar)
        latents = [torch.cat((z, v, hidden), dim=1)]
        kl = self.kl_gauss(z_mu, z_logvar) + self.kl_gauss(v_mu, v_logvar)
        for j in range(1, 10):
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
def make_model(neurons: int, config: dict, device: str) -> EndpointLangevin:
    """Instantiate upstream architecture and the approved indexing adaptation."""
    return EndpointLangevin(input_size=neurons, output_size=neurons,
                           hidden_size=config["hidden_size"], fwd_steps=0,
                           learning_rate=config["learning_rate"], weight_decay=2e-5,
                           dropout=.05, gamma=.55, cd_rate=.5).to(device)


@beartype
def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


@beartype
def memory_check() -> dict:
    """Check total device occupancy and this process's allocator peak."""
    global PROCESS_PEAK_BYTES
    free, total = torch.cuda.mem_get_info()
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True)
    process_bytes = sum(int(row.split(",")[1]) * 1024**2
                        for row in processes.splitlines()
                        if int(row.split(",")[0]) == os.getpid())
    PROCESS_PEAK_BYTES = max(PROCESS_PEAK_BYTES, process_bytes)
    result = dict(device_used_bytes=total-free,
                  process_gpu_bytes=process_bytes, process_gpu_sampled_peak_bytes=PROCESS_PEAK_BYTES,
                  allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                  reserved_peak_bytes=torch.cuda.max_memory_reserved())
    if process_bytes >= 40_000_000_000:
        raise RuntimeError(f"GPU memory ceiling exceeded: {result}")
    return result


@beartype
def load_inputs(sweep: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only counts and trial identifiers; verify every shared window."""
    counts = np.load(ROOT / "data/nitschke_20090812/counts.npy")
    with np.load(ROOT / "data/nitschke_20090812/metadata.npz") as metadata:
        trial = metadata["trial_id"]
    expected = np.arange(9, len(counts))
    expected = expected[trial[expected] == trial[expected-9]]
    with np.load(sweep / "shared/index.npz") as index:
        anchors = index["anchor_index"]
        indices = index["source_indices"]
        assert np.array_equal(anchors, expected) and len(anchors) == 130507
        assert np.array_equal(indices, anchors[:, None] + np.arange(-9, 1))
        assert np.array_equal(index["trial_id"], trial[anchors])
    return counts.astype(np.float32), anchors, indices


@beartype
def objective(model: EndpointLangevin, values: torch.Tensor,
              kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked, mask = model.cd.process_batch(values)
    prediction, _, kl = model(masked)
    nll = torch.nn.functional.poisson_nll_loss(prediction, values, reduction="none")
    nll = model.cd.process_losses(nll, mask).mean()
    return nll + kl_weight * kl, nll, kl


@beartype
def extract(model: EndpointLangevin, counts: torch.Tensor,
            indices: np.ndarray, batch_size: int) -> np.ndarray:
    """Export the deterministic final predecoder q,p,h mean from each window."""
    model.eval()
    model.posterior_mean = True
    result = np.empty((len(indices), 3 * model.hparams.hidden_size), np.float32)
    for start in range(0, len(indices), batch_size):
        values = counts[torch.as_tensor(indices[start:start+batch_size], device=counts.device)]
        with torch.enable_grad():
            _, latent, _ = model(values)
        result[start:start+len(values)] = latent[:, 9].detach().cpu().numpy()
    model.posterior_mean = False
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite endpoint means")
    return result


@beartype
def benchmark(sweep: Path, counts: torch.Tensor, indices: np.ndarray) -> None:
    """Freeze the fastest batch after timing all three sizes at maximum width."""
    measurements = []
    for batch_size in (256, 512, 1024):
        torch.manual_seed(0)
        model = make_model(counts.shape[1], configurations()[-1], "cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        values = counts[torch.as_tensor(indices[:batch_size], device="cuda")]
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for step in range(8):
            torch.cuda.synchronize()
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective(model, values, .1)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= 2:
                timings.append(time.monotonic() - start)
        measurements.append(dict(batch_size=batch_size, seconds=float(np.median(timings)),
                                 windows_per_second=batch_size / float(np.median(timings)),
                                 **memory_check()))
        del optimizer, model, values, loss
        torch.cuda.empty_cache()
    write_json(sweep / "langevinflow/benchmark.json", dict(
        measurements=measurements,
        batch_size=max(measurements, key=lambda x: x["windows_per_second"])["batch_size"]))


@beartype
def run(sweep: Path, counts: torch.Tensor, anchors: np.ndarray,
        indices: np.ndarray, config: dict, batch_size: int) -> None:
    """Train ten complete shuffled epochs and save only the final model."""
    import wandb
    if not json.loads((sweep / "manifest.json").read_text()).get("wandb_privacy_verified"):
        raise RuntimeError("W&B privacy gate is not verified")
    output = sweep / "langevinflow" / config["config_id"]
    output.mkdir(exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    global PROCESS_PEAK_BYTES
    PROCESS_PEAK_BYTES = 0
    torch.manual_seed(0)
    np.random.seed(0)
    config = dict(config, seed=0, epochs=10, batch_size=batch_size, source_revision=REVISION,
                  source_sha256=hashlib.sha256((SOURCE / "nlb_lightning/models.py").read_bytes()).hexdigest(),
                  counts_sha256=hashlib.sha256((ROOT / "data/nitschke_20090812/counts.npy").read_bytes()).hexdigest(),
                  index_sha256=hashlib.sha256((sweep / "shared/index.npz").read_bytes()).hexdigest(),
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  preprocessing="Raw counts; independent within-trial [t-9,...,t] windows",
                  extraction="Endpoint predecoder q,p,h with zero Gaussian innovations",
                  adaptation="GRU consumes observation j at state j, including endpoint9",
                  window_count=len(indices), gamma=.55, step=.01, dropout=.05, cd_rate=.5,
                  target_kl=.1, target_weight_decay=2e-5,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(),
                  gpu=torch.cuda.get_device_name(0))
    write_json(output / "config.json", config)
    tracking = wandb.init(entity="jkbhagatio", project="NLDisco-paper", group="20260914_s10_endpoint_v1",
                          name="langevinflow_" + config["config_id"], config=config,
                          dir=str(output), job_type="langevinflow-sweep")
    write_json(output / "wandb.json", dict(id=tracking.id, url=tracking.url))
    config["wandb_url"] = tracking.url
    write_json(output / "config.json", config)
    (output / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "upstream_models.py").write_bytes((SOURCE / "nlb_lightning/models.py").read_bytes())
    model = make_model(counts.shape[1], config, "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.95, patience=10, threshold=0., min_lr=1e-5)
    history = []
    torch.cuda.synchronize()
    start = time.monotonic()
    for epoch in range(10):
        model.train()
        ramp = min(epoch / config["ramp"], 1.)
        optimizer.param_groups[0]["weight_decay"] = 2e-5 * ramp
        order = np.random.permutation(len(indices))
        sums = np.zeros(3)
        epoch_start = time.monotonic()
        for offset in range(0, len(order), batch_size):
            selected = indices[order[offset:offset+batch_size]]
            values = counts[torch.as_tensor(selected, device="cuda")]
            optimizer.zero_grad(set_to_none=True)
            loss, nll, kl = objective(model, values, .1 * ramp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite objective at epoch {epoch}, batch {offset}")
            loss.backward()
            optimizer.step()
            sums += np.array([loss.item(), nll.item(), kl.item()]) * len(values)
            if offset // batch_size % 20 == 0:
                memory_check()
        means = sums / len(indices)
        scheduler.step(means[1])
        torch.cuda.synchronize()
        record = dict(epoch=epoch, objective=means[0], poisson_nll=means[1], kl=means[2],
                      kl_weight=.1*ramp, weight_decay=2e-5*ramp,
                      learning_rate=optimizer.param_groups[0]["lr"],
                      epoch_seconds=time.monotonic()-epoch_start, **memory_check())
        history.append(record)
        write_json(output / "progress.json", dict(epochs_completed=epoch+1, history=history))
        tracking.log(record, step=epoch)
        print(json.dumps(dict(config_id=config["config_id"], **record)), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.monotonic() - start
    torch.save(dict(state_dict=model.state_dict(), config=config), output / "model_final.pt")
    extraction_start = time.monotonic()
    activation = extract(model, counts, indices, batch_size)
    np.save(output / "activations.npy", activation)
    np.savez(output / "index.npz", anchor_index=anchors, source_indices=indices)
    restored = make_model(counts.shape[1], config, "cuda")
    restored.load_state_dict(torch.load(output / "model_final.pt", weights_only=False)["state_dict"])
    actual = extract(restored, counts, indices[:batch_size], batch_size)
    np.testing.assert_allclose(actual, activation[:batch_size], rtol=1e-6, atol=1e-6)
    summary = dict(complete=True, epochs_completed=10, windows=len(indices),
                   dimensions=activation.shape[1], seconds=time.monotonic()-start,
                   train_seconds=train_seconds, extraction_and_reload_seconds=time.monotonic()-extraction_start,
                   wandb_url=tracking.url, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                   reload_max_abs=float(np.max(np.abs(actual-activation[:batch_size]))),
                   **memory_check())
    artifact = wandb.Artifact("langevinflow_"+config["config_id"], type="model")
    artifact.add_file(str(output / "model_final.pt"))
    artifact.add_file(str(output / "config.json"))
    tracking.log_artifact(artifact).wait()
    tracking.summary.update(summary)
    tracking.finish()
    write_json(output / "summary.json", summary)


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--worker", type=int, choices=(0, 1))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"Unexpected source revision {revision}")
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(34_000_000_000 / total, .85))
    counts, anchors, indices = load_inputs(args.sweep)
    values = torch.as_tensor(counts, device="cuda")
    if args.benchmark:
        benchmark(args.sweep, values, indices)
        return
    batch_size = json.loads((args.sweep / "langevinflow/benchmark.json").read_text())["batch_size"]
    for config in configurations()[args.worker::2]:
        run(args.sweep, values, anchors, indices, config, batch_size)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

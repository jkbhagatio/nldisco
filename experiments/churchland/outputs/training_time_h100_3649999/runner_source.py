"""Sequential, fresh-process H100 timing of the three published 192-D models."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from beartype import beartype

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/churchland/data/nitschke_20090812"
SWEEP = Path("/ceph/aeon/aeon/nldisco/churchland/sweeps/20260914_s10_endpoint_v1")
GPU = os.environ.get("NLDISCO_TIMING_GPU_UUID", "GPU-5a924816-0a8e-d1b5-086e-e98ffd4382bb")
METHODS = ("nldisco", "cebra", "langevinflow")
NAMES = ("NLDisco", "CEBRA-Time", "LangevinFlow")
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
EPOCHS, BATCH_SIZE, WINDOWS = 10, 1024, 130507


@beartype
def dump(path: Path, value: dict) -> None:
    """Write finite, readable experiment metadata."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@beartype
def digest(path: Path) -> str:
    """Hash an input or implementation for reproducibility."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@beartype
def schedule() -> list:
    """Rotate method order between five seeds to spread temporal drift."""
    return [(METHODS[(seed + offset) % 3], seed) for seed in range(5) for offset in range(3)]


@beartype
def device_pids() -> list:
    """Read compute-process occupancy of the allocated H100 only."""
    output = subprocess.check_output([
        "nvidia-smi", "-i", GPU, "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


@beartype
def fit_nldisco(seed: int, folder: Path) -> tuple:
    """Reuse the production trainer, measuring calibration at its real boundary."""
    import torch
    from torch.utils.data import DataLoader

    from .sweeps.s10_endpoint_v1 import nldisco as implementation

    dataset, anchors, scale, mean = implementation.load_dataset()
    with np.load(SWEEP / "shared/index.npz") as saved:
        np.testing.assert_array_equal(anchors, saved["anchor_index"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    cfg = implementation.configuration(191, 192, 16, 2)
    loss = implementation.LossConfig(timebin_weights=[1.] * 10, type="mse")
    train = implementation.TrainConfig(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=.005,
        log_frequency=len(loader) // 4, dead_feature_window=len(loader) // 3)
    model = implementation.build_sed(cfg).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=train.learning_rate)
    config = dict(sed=asdict(cfg), loss=asdict(loss), training=asdict(train))
    config["sed"]["dtype"] = str(cfg.dtype)
    np.savez(folder / "preprocessing.npz", scale=scale, mean=mean)
    boundary = []
    original = model.sparsifier.begin_threshold_calibration

    def begin_calibration():
        torch.cuda.synchronize()
        boundary.append(time.perf_counter())
        return original()

    torch.cuda.synchronize()
    start = time.perf_counter()
    with patch.object(model.sparsifier, "begin_threshold_calibration", begin_calibration):
        history = implementation.train_model(model, loader, loss, train, optimizer=optimizer)
    torch.cuda.synchronize()
    end = time.perf_counter()
    if len(boundary) != 1 or max(history.loss) != 1279:
        raise AssertionError("Expected one calibration pass after 1,280 training updates.")
    dump(folder / "history.json", asdict(history))
    return model, config, dict(train_seconds=boundary[0] - start,
                              calibration_seconds=end - boundary[0], updates=1280,
                              training_examples=EPOCHS * WINDOWS,
                              epoch_definition="exhaustive shuffled passes")


@beartype
def fit_cebra(seed: int, folder: Path) -> tuple:
    """Reuse the selected official offset10 encoder and InfoNCE objective."""
    import torch
    from cebra.models import FixedCosineInfoNCE
    from sklearn.preprocessing import StandardScaler

    from .sweeps.s10_endpoint_v1 import cebra as implementation

    counts = np.load(DATA / "counts.npy").astype(np.float32)
    with np.load(DATA / "metadata.npz") as saved:
        trials = saved["trial_id"]
    with np.load(SWEEP / "shared/index.npz") as saved:
        anchors = saved["anchor_index"]
    scaler = StandardScaler()
    data = torch.from_numpy(scaler.fit_transform(np.sqrt(counts)).astype(np.float32)).cuda()
    refs = torch.as_tensor(implementation.pair_endpoints(anchors, trials, 1), device="cuda")
    endpoints = torch.as_tensor(anchors, device="cuda")
    model = implementation.encoder(191, 192, "cuda")
    objective = FixedCosineInfoNCE(temperature=1.).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    steps = math.ceil(EPOCHS * WINDOWS / BATCH_SIZE)
    config = dict(architecture="offset10-model", num_units=128, dimension=192,
                  temperature=1., lag=1, learning_rate=3e-4,
                  preprocessing="sqrt counts, unit z-score", batch_size=BATCH_SIZE)
    np.savez(folder / "preprocessing.npz", mean=scaler.mean_, scale=scaler.scale_)
    history = []
    model.train()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        reference = refs[torch.randint(len(refs), (BATCH_SIZE,), device="cuda")]
        negative = endpoints[torch.randint(len(endpoints), (BATCH_SIZE,), device="cuda")]
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(implementation.windows(
            data, torch.cat((reference, reference + 1, negative))))
        result = objective(*embeddings.chunk(3))
        if not all(torch.isfinite(value).all() for value in result):
            raise FloatingPointError("Nonfinite CEBRA objective")
        result[0].backward()
        optimizer.step()
        history.append([float(value.detach()) for value in result])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    np.save(folder / "history.npy", np.asarray(history))
    return model, config, dict(train_seconds=elapsed, calibration_seconds=0., updates=steps,
                              training_examples=steps * BATCH_SIZE,
                              epoch_definition="reference-draw equivalents; replacement sampling")


@beartype
def fit_langevinflow(seed: int, folder: Path) -> tuple:
    """Reuse the pinned observation-aligned adapter and selected 64-state fit."""
    import torch

    from .sweeps.s10_endpoint_v1 import langevinflow as implementation

    revision = subprocess.check_output(
        ["git", "-C", str(implementation.SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != implementation.REVISION:
        raise RuntimeError("LangevinFlow source revision changed")
    counts, anchors, indices = implementation.load_inputs(SWEEP)
    if len(anchors) != WINDOWS:
        raise AssertionError("Unexpected training windows")
    data = torch.as_tensor(counts, device="cuda")
    config = dict(hidden_size=64, dimension=192, learning_rate=.003, ramp=500,
                  gamma=.55, step=.01, dropout=.05, cd_rate=.5,
                  target_kl=.1, target_weight_decay=2e-5,
                  source_revision=revision, batch_size=BATCH_SIZE,
                  preprocessing="raw counts", adapter="observation-aligned independent windows")
    model = implementation.make_model(191, config, "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.95, patience=10, threshold=0., min_lr=1e-5)
    history = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        ramp = min(epoch / config["ramp"], 1.)
        optimizer.param_groups[0]["weight_decay"] = 2e-5 * ramp
        order = np.random.permutation(WINDOWS)
        sums = np.zeros(3)
        for offset in range(0, WINDOWS, BATCH_SIZE):
            selected = indices[order[offset:offset + BATCH_SIZE]]
            values = data[torch.as_tensor(selected, device="cuda")]
            optimizer.zero_grad(set_to_none=True)
            loss, nll, kl = implementation.objective(model, values, .1 * ramp)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite LangevinFlow objective")
            loss.backward()
            optimizer.step()
            sums += np.array([loss.item(), nll.item(), kl.item()]) * len(values)
        means = sums / WINDOWS
        scheduler.step(means[1])
        history.append(dict(epoch=epoch, loss=means[0], nll=means[1], kl=means[2]))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    dump(folder / "history.json", dict(epochs=history))
    return model, config, dict(train_seconds=elapsed, calibration_seconds=0., updates=1280,
                              training_examples=EPOCHS * WINDOWS,
                              epoch_definition="exhaustive shuffled passes")


@beartype
def worker(method: str, seed: int, folder: Path) -> None:
    """Execute one complete fit; initialization, saving and checks are untimed."""
    import torch

    from nldisco.util import set_seed

    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU or torch.cuda.device_count() != 1:
        raise RuntimeError("Pin the benchmark H100 by UUID")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This benchmark requires an H100")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(34e9 / torch.cuda.get_device_properties(0).total_memory)
    set_seed(seed)
    folder.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    model, config, result = globals()[f"fit_{method}"](seed, folder)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("Nonfinite model state")
    torch.save(model.state_dict(), folder / "model.pt")
    restored = torch.load(folder / "model.pt", weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored[key], rtol=0, atol=0)
    dump(folder / "config.json", config)
    result.update(status="complete", method=method, seed=seed, pid=os.getpid(),
                  started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  dimension=192, epochs=EPOCHS, windows=WINDOWS, batch_size=BATCH_SIZE,
                  gpu_uuid=GPU, gpu_name=torch.cuda.get_device_name(0), host=socket.gethostname(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  finite_parameters=True, checkpoint_roundtrip_exact=True,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  python_environment=str(Path(os.sys.executable)), cpu_threads=4,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  package_versions={p: importlib.metadata.version(p) for p in
                                    ("numpy", "torch", "cebra", "beartype")})
    dump(folder / "summary.json", result)
    print(json.dumps(result), flush=True)


@beartype
def validate_results(rows: list) -> None:
    """Reject missing, duplicated, contaminated, or incompatible timed fits."""
    if len(rows) != 15 or {(r["method"], r["seed"]) for r in rows} != set(schedule()):
        raise ValueError("Require exactly five distinct seeds per method (15 completed fits)")
    for row in rows:
        expected = 1275 if row["method"] == "cebra" else 1280
        if (row["status"] != "complete" or row["gpu_uuid"] != GPU
                or row["dimension"] != 192 or row["epochs"] != 10
                or row["updates"] != expected or row["foreign_gpu_pids"]
                or not math.isfinite(row["train_seconds"]) or row["train_seconds"] <= 0):
            raise ValueError(f"Invalid timing record: {row}")


@beartype
def report(output: Path) -> None:
    """Plot every measured fit and save machine-readable quartiles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((output / "results.json").read_text())["runs"]
    validate_results(rows)
    with (output / "results.csv").open("w") as stream:
        fields = ["method", "seed", "train_seconds", "calibration_seconds", "updates",
                  "training_examples", "gpu_uuid", "pid"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.2, 3.5), layout="constrained")
    summaries = {}
    for index, (method, name, color) in enumerate(zip(METHODS, NAMES, COLORS), 1):
        group = sorted((r for r in rows if r["method"] == method), key=lambda r: r["seed"])
        values = np.array([r["train_seconds"] for r in group])
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summaries[method] = dict(median=float(median), q1=float(q1), q3=float(q3),
                                 min=float(values.min()), max=float(values.max()),
                                 calibration_median=float(np.median(
                                     [r["calibration_seconds"] for r in group])))
        ax.boxplot([values], positions=[index], widths=.46, patch_artist=True,
                   showfliers=False, boxprops=dict(facecolor=color, alpha=.2, edgecolor=color),
                   medianprops=dict(color=color, linewidth=2),
                   whiskerprops=dict(color=color), capprops=dict(color=color))
        ax.scatter(index + np.linspace(-.12, .12, 5), values, c=color, s=34,
                   edgecolors="white", linewidths=.6, zorder=3)
        ax.text(index, values.max() + max(r["train_seconds"] for r in rows) * .055,
                f"{median:.1f} s", ha="center", fontweight="bold", color=color)
    ax.set(xticks=[1, 2, 3], xticklabels=NAMES, ylabel="Training time (s)",
           title="192 dimensions · 10 epochs · H100", ylim=(0, None))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    ax.margins(y=.18)
    fig.savefig(output / "training_times.pdf")
    fig.savefig(output / "training_times.png", dpi=220)
    plt.close(fig)
    dump(output / "statistics.json", summaries)
    table_rows = "\n".join(
        f"{name} & {summaries[method]['median']:.2f} & "
        f"[{summaries[method]['q1']:.2f}, {summaries[method]['q3']:.2f}] \\\\"
        for method, name in zip(METHODS, NAMES))
    calibration = summaries["nldisco"]["calibration_median"]
    appendix = r"""\subsubsection{Training-time comparison}
\label{subsubsection:training_time_comparison}

We compare the fixed 192-dimensional NLDisco, CEBRA-Time, and LangevinFlow
configurations used in the Churchland reaching analysis on the same NVIDIA
H100 80-GB GPU. Each method is trained from scratch five times (seeds 0--4),
for 15 sequential fits in separate processes. Method order rotates between
seeds. All fits use the same 130,507 complete within-trial windows, each
containing ten 50-ms bins from 191 channels, with batch size 1,024.
NLDisco and LangevinFlow make ten exhaustive shuffled passes (1,280 updates).
CEBRA-Time samples temporal-positive reference pairs and negatives with
replacement: its ten epoch-equivalents comprise 1,275 updates and 1,305,600
reference draws, rather than exhaustive passes through the windows.
The fixed configurations are NLDisco with nested widths 96/192 and sparsity
budgets 8/16; CEBRA-Time with a 192-dimensional offset10 encoder, temperature
1 and lag 1; and LangevinFlow with state width 64, exposing 192 concatenated
position, velocity and recurrent-state coordinates, learning rate 0.003 and
a 500-epoch KL/weight-decay ramp. Preprocessing and objectives remain
method-specific, as detailed in the Churchland experiment.

Wall-clock timing is synchronized with CUDA immediately before and after
optimization. It includes minibatch preparation, device transfers during
training, loss checks and the trainer's local progress bookkeeping; it
excludes process startup, initial data loading/preprocessing, model and
optimizer construction, checkpoint writing, and latent export. Each fit
starts in a fresh process, without discarded training warm-up updates.
NLDisco's post-training threshold calibration is timed separately and
excluded from the plotted optimization time. Its median calibration time
is CALIBRATION_SECONDS~s. External experiment tracking is disabled, and all
fits use four CPU threads and PyTorch 2.5.1 with CUDA 12.1. GPU process
occupancy is sampled every two seconds; no competing compute process is
observed on the benchmark GPU. Other GPUs on the host are not reserved.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/training_time_h100.pdf}
    \caption{\textbf{Training time for the three 192-dimensional models on
    one H100.} Each dot is one fresh ten-epoch fit (ten reference-draw
    epoch-equivalents for CEBRA-Time), with five fits per method. Boxes span
    the 25th--75th percentiles and mark the median; whiskers extend to the
    most extreme observations within 1.5 interquartile ranges. All five
    observations are overlaid, including any beyond the whiskers.
    Threshold calibration is excluded.}
    \label{figure:training_time_h100}
\end{figure}

\begin{center}
\begin{tabular}{lrr}
\toprule
Method & Median (s) & Interquartile interval (s) \\
\midrule
TABLE_ROWS
\bottomrule
\end{tabular}
\end{center}

These repeated timings compare the specified implementations at a fixed
data-exposure budget on common hardware. Equal exposed dimension does not
equalize parameter counts, training objectives, sampled observations, or
convergence; the measurements do not establish time to matched predictive
performance or discovery quality. The five fits characterize runtime and
initialization variability on one recording, not between-dataset variation.
"""
    appendix = appendix.replace("CALIBRATION_SECONDS", f"{calibration:.2f}")
    appendix = appendix.replace("TABLE_ROWS", table_rows)
    (output / "appendix_training_time.tex").write_text(appendix)


@beartype
def update_paper(output: Path) -> None:
    """Install only a complete, validated measured report into Section 6.3."""
    validate_results(json.loads((output / "results.json").read_text())["runs"])
    paper = ROOT / "paper/iclr_paper"
    for suffix in ("pdf", "png"):
        (paper / f"figures/training_time_h100.{suffix}").write_bytes(
            (output / f"training_times.{suffix}").read_bytes())
    (paper / "appendix/training_time_comparison.tex").write_bytes(
        (output / "appendix_training_time.tex").read_bytes())
    details = paper / "appendix/additional_pipeline_details.tex"
    marker = "\\input{appendix/training_time_comparison}"
    content = details.read_text()
    if marker not in content:
        details.write_text(content.rstrip() + "\n\n" + marker + "\n")


@beartype
def run(output: Path) -> None:
    """Launch exactly 15 nonoverlapping child processes on the requested H100."""
    if device_pids():
        raise RuntimeError("Benchmark GPU is occupied; no process is stopped automatically")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *sorted((ROOT / "src/nldisco").rglob("*.py")),
               *(ROOT / f"experiments/churchland/sweeps/s10_endpoint_v1/{m}.py" for m in METHODS),
               ROOT / "experiments/churchland/scripts/train_nldisco.py",
               ROOT / "experiments/churchland/environments/LangevinFlow_CCN/nlb_lightning/models.py"]
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), schedule=schedule(),
                      sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                      data={str(p): digest(p) for p in
                            (DATA / "counts.npy", DATA / "metadata.npz", SWEEP / "shared/index.npz")},
                      timing="CUDA-synchronized wall time, training loop only; calibration separate",
                      setup="fresh process per fit; no discarded training warm-up",
                      logging="local only; no W&B, checkpointing or export in timed interval",
                      gpu_uuid=GPU, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      other_gpus="not reserved; other host jobs may continue")
    dump(output / "manifest.json", provenance)
    (output / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, WANDB_MODE="disabled",
               PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT),
                                          os.environ.get("PYTHONPATH", ""))),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    rows = []
    for order, (method, seed) in enumerate(schedule()):
        if device_pids():
            raise RuntimeError("Benchmark GPU acquired another job; stop before the next fit")
        folder = output / f"{order + 1:02d}_{method}_seed{seed}"
        python = (ROOT / "experiments/churchland/environments/langevinflow-venv/bin/python"
                  if method == "langevinflow" else ROOT / ".venv/bin/python")
        command = ["uv", "run", "--no-project", "--python", str(python), "python", "-u", "-m",
                   "experiments.churchland.training_time_benchmark", "worker", "--output",
                   str(folder), "--method", method, "--seed", str(seed)]
        print(f"Starting {order + 1}/15: {method}, seed {seed}", flush=True)
        samples = []
        with (output / f"{order + 1:02d}_{method}_seed{seed}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                samples.append(dict(time=time.time(), pids=device_pids()))
                time.sleep(2)
        dump(output / f"{order + 1:02d}_occupancy.json", dict(samples=samples))
        if child.returncode:
            raise RuntimeError(f"Fit failed ({child.returncode}); see {folder.name}.log")
        row = json.loads((folder / "summary.json").read_text())
        row["foreign_gpu_pids"] = sorted({pid for s in samples for pid in s["pids"]} - {row["pid"]})
        row["order"] = order + 1
        rows.append(row)
        dump(output / "results.json", dict(runs=rows))
        if row["foreign_gpu_pids"]:
            raise RuntimeError("GPU contention detected; timings must not be reported")
        print(f"Completed {order + 1}/15: {row['train_seconds']:.3f} s", flush=True)
    validate_results(rows)
    for source, expected in provenance["sources"].items():
        if digest(ROOT / source) != expected:
            raise RuntimeError(f"Source changed during the benchmark: {source}")
    report(output)
    print(f"Complete: {output}", flush=True)


@beartype
def main() -> None:
    """Run the benchmark, one subprocess, or regenerate the final figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "worker", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--update-paper", action="store_true")
    args = parser.parse_args()
    if args.stage == "worker":
        worker(args.method, args.seed, args.output)
    else:
        (run if args.stage == "run" else report)(args.output)
        if args.update_paper:
            update_paper(args.output)


if __name__ == "__main__":
    main()

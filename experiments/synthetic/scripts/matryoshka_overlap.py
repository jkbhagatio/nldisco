"""Separate, fixed-design feasibility test for overlapping broad/narrow place cells.

No canonical paper data or trained models are modified. Run as a repository module.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from jaxtyping import Float
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

from experiments.synthetic.scripts.feature_selection import (
    SelectionConfig,
    best_features,
    biological_traceback,
    compute_tuning_curves,
    ground_truth_curves,
    normalize_tuning_curves,
    score_tuning_curves,
)
from nldisco.config import LossConfig, SedConfig, TrainConfig
from nldisco.data import SpikeWindowDataset
from nldisco.model import build_sed
from nldisco.train import train_model, weighted_reconstruction_loss
from nldisco.util import set_seed

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/paper_simulation"
DATA = ROOT / "data/matryoshka_overlap"
RUNS = ROOT / "outputs/runs/matryoshka_overlap"
LEVELS = (16, 32, 64, 128)
SELECTION = replace(
    SelectionConfig(), cell_centers=(0.2, 0.4, 0.7, 0.7),
    left_widths=(0.05, 0.10, 0.14, 0.06), right_widths=(0.10, 0.05, 0.14, 0.06),
)


def sha256(path: Path) -> str:
    """Return a file fingerprint for source/result validation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rates(positions: Float[np.ndarray, "position"]) -> Float[np.ndarray, "cell position"]:  # noqa: F821
    """Evaluate unchanged asymmetric first fields and concentric final fields."""
    displacement = positions[None] - np.asarray(SELECTION.cell_centers)[:, None]
    widths = np.where(displacement < 0, np.asarray(SELECTION.left_widths)[:, None],
                      np.asarray(SELECTION.right_widths)[:, None])
    return SELECTION.baseline_rate_hz + np.asarray(SELECTION.peak_amplitudes_hz)[:, None] * np.exp(
        -0.5 * (displacement / widths) ** 2
    )


def generate_variant(source: Path = SOURCE, destination: Path = DATA) -> dict:
    """Create a new variant, changing only place-cell spike rows 2 and 3."""
    if destination.exists():
        raise FileExistsError(f"Refusing to replace dataset: {destination}")
    original = json.loads((source / "parameters.json").read_text())
    positions = np.load(source / "positions.npy")
    spikes = np.load(source / "spike_matrix.npy")
    if spikes.shape != (100, 60000):
        raise ValueError("Expected the complete four-place-cell/96-noise paper recording")
    values = rates(positions)
    uniforms = np.random.default_rng(original["place_spike_seed"]).random((4, len(positions)))
    replacement = uniforms < values * original["dt_s"]
    np.testing.assert_array_equal(replacement[:2], spikes[:2])
    spikes[2:4] = replacement[2:4]
    grid = np.linspace(0, 1, 501)
    edges = np.load(source / "empirical_rate_bin_edges.npy")
    occupancy = np.histogram(positions, edges)[0] * original["dt_s"]
    unsmoothed = np.stack([np.histogram(positions[row], edges)[0] for row in spikes[:4]]) / occupancy
    arrays = {
        "positions": positions, "spike_matrix": spikes,
        "timestamps": np.load(source / "timestamps.npy"),
        "velocities": np.load(source / "velocities.npy"),
        "generative_place_rates": values,
        "theoretical_rate_positions": grid, "theoretical_rate_maps": rates(grid),
        "empirical_rate_bin_edges": edges, "empirical_rate_occupancy_s": occupancy,
        "empirical_rate_maps_unsmoothed": unsmoothed,
        "empirical_rate_maps": gaussian_filter1d(unsmoothed, 0.5, axis=1, mode="nearest"),
    }
    destination.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(destination / f"{name}.npy", array)
    metadata = {
        "design": "exploratory_matryoshka_concentric_broad_narrow_fields",
        "source_dataset_sha256": sha256(source / "spike_matrix.npy"),
        "source_parameters": original,
        "centers_m": list(SELECTION.cell_centers),
        "left_widths_m": list(SELECTION.left_widths),
        "right_widths_m": list(SELECTION.right_widths),
        "peak_amplitudes_hz": list(SELECTION.peak_amplitudes_hz),
        "baseline_hz": SELECTION.baseline_rate_hz,
        "changed_units_zero_based": [2, 3],
        "unchanged": "positions, timestamps, velocities, units0/1 and all96 noise spike rows",
        "generation": "Reuse original seed123 4x60000 uniform draws; replace only units2/3 with new Bernoulli rates",
        "scope": "Full-recording descriptive feasibility; no vanilla comparison; no strict support nesting",
        "selection_config": SELECTION.to_dict(),
        "files": {f"{name}.npy": sha256(destination / f"{name}.npy") for name in arrays},
        "generator_sha256": sha256(Path(__file__)),
    }
    (destination / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def joint_recovery(selected: dict) -> bool:
    """Require both cells to pass as distinct latents at the same dictionary level."""
    broad, narrow = selected["cell_2"], selected["cell_3"]
    return bool(broad["hit"] and narrow["hit"] and broad["latent_idx"] != narrow["latent_idx"])


def run_seed(seed: int, epochs: int, device: str) -> dict:
    """Train one fixed Matryoshka configuration and evaluate every nested level."""
    output = RUNS / f"seed_{seed:02d}"
    if output.exists():
        raise FileExistsError(f"Refusing to replace run: {output}")
    metadata = json.loads((DATA / "parameters.json").read_text())
    if metadata["selection_config"] != json.loads(json.dumps(SELECTION.to_dict())):
        raise ValueError("Variant selection configuration mismatch")
    if sha256(DATA / "spike_matrix.npy") != metadata["files"]["spike_matrix.npy"]:
        raise ValueError("Variant spike data fingerprint mismatch")
    set_seed(seed)
    values = np.load(DATA / "spike_matrix.npy").T.astype(np.float32)
    mean, std = values.mean(axis=0), values.std(axis=0)
    if (std <= 0).any():
        raise ValueError("All units must have nonzero variance")
    values = (values - mean) / std + 1e-8
    positions = np.load(DATA / "positions.npy")
    dataset = SpikeWindowDataset(t.from_numpy(values), 1)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True,
                        generator=t.Generator().manual_seed(seed))
    evaluation = DataLoader(dataset, batch_size=1024, shuffle=False)
    cfg = SedConfig(n_neurons=100, seq_len=1, dsed_topk_map={level: 4 for level in LEVELS},
                    dtype=t.bfloat16 if device.startswith("cuda") else t.float32)
    loss = LossConfig(timebin_weights=[1.0], type="msle", tau=1.0)
    training_cfg = TrainConfig(epochs=epochs, batch_size=1024, learning_rate=0.005,
                              dead_feature_window=max(1, len(loader) // 3))
    config = {"seed": seed, "sed_config": {**asdict(cfg), "dtype": str(cfg.dtype)},
              "loss_config": asdict(loss), "train_config": asdict(training_cfg),
              "selection_config": SELECTION.to_dict(), "dataset_sha256": sha256(DATA / "spike_matrix.npy"),
              "scope": "all60000samples train/select/evaluate; exploratory, not paper-data replacement",
              "level_loss_weighting": "equal reconstruction loss weights across nested prefixes",
              "script_sha256": sha256(Path(__file__))}
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = build_sed(cfg).to(device)
    history = train_model(model, loader, loss, training_cfg)
    (output / "history.json").write_text(json.dumps(asdict(history), indent=2) + "\n")
    t.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": config, "mean": t.from_numpy(mean), "std": t.from_numpy(std)}, output / "checkpoint.pt")
    activations = {level: [] for level in LEVELS}
    losses = {level: [] for level in LEVELS}
    sizes = []
    model.eval()
    with t.inference_mode():
        for batch in evaluation:
            target = batch.values.to(device=device, dtype=cfg.dtype)
            result = model(target)
            sizes.append(len(target))
            for level in LEVELS:
                activations[level].append(result.sparse_acts[level].float().cpu().numpy())
                value, _ = weighted_reconstruction_loss(target.float(), result.reconstructions[level].float(), loss)
                losses[level].append(float(value))
    decoder = model.decoder.weight.detach().float().cpu().numpy()[0].T
    index = pd.DataFrame({"source_time_idx": np.arange(len(values))})
    summary = {"seed": seed, "config": config, "levels": {}}
    for level in LEVELS:
        dense = np.concatenate(activations[level])
        if not np.isfinite(dense).all():
            raise ValueError("Nonfinite feature activations")
        source, latent = np.nonzero(dense > 0)
        table = pd.DataFrame({"source_time_idx": source, "latent_idx": latent,
                              "activation_value": dense[source, latent]})
        curves, centers = compute_tuning_curves(table, index, positions, level, SELECTION)
        scores = score_tuning_curves(curves, centers, SELECTION)
        scores, traces = biological_traceback(table, values, positions, decoder[:level], scores, SELECTION)
        selected = best_features(scores)
        scores.to_csv(output / f"level_{level}_scores.csv", index=False)
        np.savez_compressed(output / f"level_{level}_artifacts.npz", activations=dense,
                            position_centers=centers, curves=normalize_tuning_curves(curves),
                            ground_truth_curves=ground_truth_curves(centers, SELECTION), **traces)
        summary["levels"][str(level)] = {
            "selected": selected, "both_cells_recovered": joint_recovery(selected),
            "full_recording_msle": float(np.average(losses[level], weights=sizes)),
            "mean_l0": float((dense > 0).sum(axis=1).mean()),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"seed": seed, "joint_levels": [level for level in LEVELS if summary["levels"][str(level)]["both_cells_recovered"]]}), flush=True)
    return summary


def aggregate() -> dict:
    """Summarize every saved replica, without substituting failed features."""
    summaries = [json.loads(path.read_text()) for path in sorted(RUNS.glob("seed_*/summary.json"))]
    if not summaries:
        raise ValueError("No completed Matryoshka runs")
    if len({s["config"]["dataset_sha256"] for s in summaries}) != 1:
        raise ValueError("Dataset mismatch across runs")
    rows = []
    for summary in summaries:
        for level in LEVELS:
            item = summary["levels"][str(level)]
            rows.append({"seed": summary["seed"], "level": level,
                         "both": item["both_cells_recovered"], "msle": item["full_recording_msle"],
                         **{f"cell_{unit}_{key}": item["selected"][f"cell_{unit}"][key]
                            for unit in (2, 3) for key in ("hit", "latent_idx", "score")}})
    frame = pd.DataFrame(rows)
    frame.to_csv(RUNS / "recovery.csv", index=False)
    counts = frame.groupby("level")[["cell_2_hit", "cell_3_hit", "both"]].sum().astype(int)
    result = {"n_runs": len(summaries), "seeds": [s["seed"] for s in summaries],
              "recovery_by_level": counts.to_dict(orient="index"),
              "same_run_any_level_joint_count": int(frame.groupby("seed").both.any().sum()),
              "scope": "Feasibility only, no vanilla baseline or unique Matryoshka advantage claim"}
    (RUNS / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def figure(retained_seed=None, save: bool = True) -> plt.Figure:
    """Plot a same-run broad/narrow pair across all nested levels if recovered."""
    frame = pd.read_csv(RUNS / "recovery.csv")
    pairs = frame[frame.both].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), layout="constrained")
    grid = np.load(DATA / "theoretical_rate_positions.npy")
    known = np.load(DATA / "theoretical_rate_maps.npy")
    for unit, label, color in ((2, "Cell 3: broad", "#33658A"), (3, "Cell 4: narrow", "#d97706")):
        axes[0, 0].plot(grid, known[unit], color=color, label=label)
    axes[0, 0].set(title="Overlapping input fields", xlabel="Position (m)", ylabel="Expected rate (Hz)")
    axes[0, 0].legend(frameon=False)
    counts = frame.groupby("level")[["cell_2_hit", "cell_3_hit", "both"]].sum()
    for column, label in (("cell_2_hit", "Broad"), ("cell_3_hit", "Narrow"), ("both", "Both in same run")):
        axes[1, 0].plot(np.arange(len(LEVELS)), counts[column], "o-", label=label)
    axes[1, 0].set(xticks=np.arange(len(LEVELS)), xticklabels=LEVELS, ylim=(-0.2, frame.seed.nunique() + 0.2),
                   xlabel="Dictionary size", ylabel="Qualifying runs", title="Recovery across all seeds")
    axes[1, 0].legend(frameon=False, fontsize=8)
    if pairs.empty:
        for ax in axes[:, 1:].flat:
            ax.text(0.5, 0.5, "No same-level broad/narrow pair recovered", ha="center", transform=ax.transAxes)
        chosen_seed = None
    else:
        pairs["quality"] = pairs[["cell_2_score", "cell_3_score"]].min(axis=1)
        chosen_seed = (int(pairs.sort_values(["quality", "level", "seed"], ascending=[False, False, True]).iloc[0].seed)
                       if retained_seed is None else int(retained_seed))
        if chosen_seed not in pairs.seed.values:
            raise ValueError("Retained example must have a qualifying same-level pair")
        for level, ax in zip(LEVELS, axes[:, 1:].flat):
            artifact = np.load(RUNS / f"seed_{chosen_seed:02d}/level_{level}_artifacts.npz")
            row = frame[(frame.seed == chosen_seed) & (frame.level == level)].iloc[0]
            for unit, color in ((2, "#33658A"), (3, "#d97706")):
                ax.plot(artifact["position_centers"], artifact["ground_truth_curves"][unit], "--", color=color, alpha=0.6)
                if row[f"cell_{unit}_hit"]:
                    latent = int(row[f"cell_{unit}_latent_idx"])
                    ax.plot(artifact["position_centers"], artifact["curves"][latent], color=color,
                            label=f"Cell {unit + 1}: latent {latent}")
                else:
                    ax.plot([], [], color=color, label=f"Cell {unit + 1}: no qualifying feature")
            ax.set(title=f"Seed {chosen_seed}, prefix {level}", xlabel="Position (m)", ylabel="Normalized activation", ylim=(0, 1.05))
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Matryoshka overlap recovery: fixed criteria, {frame.seed.nunique()} initialization seeds")
    if not save:
        return fig
    destination = RUNS / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"overlap_recovery.{suffix}", dpi=250, bbox_inches="tight")
    (destination / "selection.json").write_text(json.dumps({"seed": chosen_seed, "policy": "best minimum pair score among joint-passing run/level combinations; same seed shown at all levels"}, indent=2) + "\n")
    return fig


def main() -> None:
    """Run a small fixed experiment or regenerate its saved-artifact report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    t.set_num_threads(2)
    if not args.report_only:
        for seed in args.seeds:
            if (RUNS / f"seed_{seed:02d}").exists():
                raise FileExistsError(f"Requested seed {seed} already exists; use --report-only")
        if not DATA.exists():
            generate_variant()
        RUNS.mkdir(parents=True, exist_ok=True)
        plan = {"seeds": args.seeds, "levels": LEVELS, "topk_per_level": 4,
                "epochs": args.epochs, "selection": SELECTION.to_dict(),
                "primary_success": "distinct broad/narrow latents passing in same run and same level"}
        (RUNS / "fixed_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        for seed in args.seeds:
            run_seed(seed, args.epochs, args.device)
    print(json.dumps(aggregate(), indent=2))
    figure()


if __name__ == "__main__":
    main()

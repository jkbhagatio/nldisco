"""Train matched FlatWindow/TransformerWindow SED paper replicas.

Example: uv run python experiments/synthetic/scripts/train_window.py --architecture FlatWindow
All data are used for training, selection, and descriptive discovery analysis.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from dataset_metadata import load_parameters
from einops import rearrange, reduce
from model_names import ARCHITECTURES, RUN_DIRECTORIES, canonical_run_config
from temporal_analysis import (
    decoder_metrics,
    matched_direction_metrics,
    nonoverlapping_top_indices,
    select_features,
    window_motion,
)
from torch.utils.data import DataLoader

from nldisco.config import (
    EncoderConfig,
    LossConfig,
    SedConfig,
    TrainConfig,
    canonical_encoder_type,
)
from nldisco.data import SpikeWindowDataset
from nldisco.model import build_sed
from nldisco.train import train_model, weighted_reconstruction_loss
from nldisco.util import set_seed


def evaluate(model, dataset, loss_config, batch_size):
    """Stream evaluation without retaining redundant overlapping reconstructions."""
    model.eval()
    device = next(model.parameters()).device
    scores, losses, sizes = [], [], []
    per_unit_error = np.zeros(model.cfg.n_neurons)
    with t.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            values = batch.values.to(device=device, dtype=model.cfg.dtype)
            output = model(values)
            reconstruction = output.reconstructions[model.cfg.n_features]
            loss, _ = weighted_reconstruction_loss(
                values.float(), reconstruction.float(), loss_config
            )
            scores.append(output.sparse_acts[model.cfg.n_features].float().cpu().numpy())
            losses.append(float(loss))
            sizes.append(len(values))
            per_unit_error += (values.float() - reconstruction.float()).square().mean(
                (0, 1)
            ).cpu().numpy() * len(values)
    return (
        np.concatenate(scores),
        float(np.average(losses, weights=sizes)),
        per_unit_error / sum(sizes),
    )


def main():
    """Run a bounded, reproducible matched architecture comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", type=canonical_encoder_type,
                        choices=ARCHITECTURES, default="FlatWindow")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bin-factor", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    t.set_num_threads(2)
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "experiments/synthetic/data/matryoshka_overlap"
    output_dir = args.output_dir or root / "experiments/synthetic/outputs/runs" / RUN_DIRECTORIES[args.architecture]
    output_dir.mkdir(parents=True, exist_ok=True)
    spikes_path = data_dir / "spike_matrix.npy"
    spikes = np.load(spikes_path).T.astype(np.float32)
    positions = np.load(data_dir / "positions.npy")
    timestamps = np.load(data_dir / "timestamps.npy")
    assert spikes.shape == (60000, 100), "Refuse to train on an unexpected paper dataset"
    parameters = load_parameters(data_dir)
    centers = np.asarray(parameters["centers_m"])
    dt = float(np.median(np.diff(timestamps)))
    raw_train_stop = len(spikes)
    statistics = []
    for length in (50, 75, 100, 125, 150, 175, 200):
        motion = window_motion(
            positions, np.arange(0, raw_train_stop - length + 1, 5), length,
            pair_centers=centers[:2],
        )
        strict_counts = [int((motion.strict_first_pair_transition & (motion.direction == sign)).sum()) for sign in (1, -1)]
        purposeful_counts = [int((motion.purposeful_first_pair_transition & (motion.direction == sign)).sum()) for sign in (1, -1)]
        statistics.append(
            {
                "raw_bins": length,
                "duration_s": (length - 1) * dt,
                "median_abs_displacement_m": float(motion.displacement.abs().median()),
                "reversal_fraction": float(motion.reversal.mean()),
                "clean_meaningful_fraction": float(motion.clean.mean()),
                "strict_first_pair_right_windows": strict_counts[0],
                "strict_first_pair_left_windows": strict_counts[1],
                "strict_first_pair_balanced_windows": min(strict_counts),
                "purposeful_first_pair_right_windows": purposeful_counts[0],
                "purposeful_first_pair_left_windows": purposeful_counts[1],
            }
        )
    window_stats = pd.DataFrame(statistics)
    # Freeze from movement alone: actual first-pair coverage in both directions.
    chosen = window_stats.sort_values(
        ["strict_first_pair_balanced_windows", "raw_bins"], ascending=[False, True]
    ).iloc[0]
    raw_length = int(chosen.raw_bins)
    seq_len = raw_length // args.bin_factor
    raw_length = seq_len * args.bin_factor
    window_stats.to_csv(output_dir / "window_length_statistics.csv", index=False)
    counts = reduce(spikes, "(time group) neuron -> time neuron", "sum", group=args.bin_factor)
    tensor = t.from_numpy(counts)
    split_bounds = {"train": (0, len(counts)), "analysis": (0, len(counts))}
    datasets = {}
    metadata = {}
    for label, (start, stop) in split_bounds.items():
        allowed = (np.arange(len(counts)) >= start) & (np.arange(len(counts)) < stop)
        datasets[label] = SpikeWindowDataset(tensor, seq_len, stride=1, allowed_rows=allowed)
        starts = np.asarray(datasets[label].valid_starts) * args.bin_factor
        metadata[label] = window_motion(positions, starts, raw_length, pair_centers=centers[:2])
        metadata[label].to_csv(output_dir / f"{label}_windows.csv", index=False)
    occupancy = matched_direction_metrics(
        np.ones((len(metadata["analysis"]), 1)), metadata["analysis"],
        bins=40, eligible_column="ramp_clean",
    ).iloc[0]
    assert occupancy.matched_position_bins >= 2, "Insufficient position-matched ramp coverage"
    shared_config = {
        "architecture": args.architecture,
        "data_dir": str(data_dir),
        "spikes_sha256": hashlib.sha256(spikes_path.read_bytes()).hexdigest(),
        "raw_dt": dt,
        "bin_factor": args.bin_factor,
        "seq_len": seq_len,
        "raw_length": raw_length,
        "duration_s": (raw_length - 1) * dt,
        "window_rule": "motion-only fixed50,75,100,125,150,175,200 raw bins; maximize minimum direction count of strict first-pair crossings; shorter tie-break",
        "first_pair_centers_m": centers[:2].tolist(),
        "first_pair_crossing_rule": "trajectory visits <=first center+.02m and >=second center-.02m; strict=no raw10ms reversal; ramp selection also mean position between centers",
        "purposeful_motion_diagnostic": "same crossing with net>=pair gap-.04m, maximum backtracking<=.02m, net/pathlength>=.8; does not determine selection or pass gates",
        "matched_ramp_occupancy": occupancy.to_dict(),
        "direction_position_bin_width_m": 0.025,
        "minimum_direction_count_per_bin": 10,
        "split_raw_bounds": [0, len(spikes)],
        "analysis_scope": "in-sample descriptive discovery; full recording used for training and selection",
        "preprocessing": "sum nonoverlapping count bins; no standardization or smoothing",
        "minimum_displacement_m": 0.1,
        "selection": "full recording strict first-pair crossings; position-matched DI times first-pair decoder motion times log1p first-pair enrichment; passing candidates precede failures",
        "primary_metric_definitions": "right_selectivity uses ramp_clean windows in2.5cm occupancy-matched bins; decoder_center_displacement uses units0,1 positive squared weights early/late thirds; place_energy_enrichment uses units0,1 versus96noise; global_ prefixes retain fulltrack/all4 equivalents",
        "trajectory_selection": "top activation over ALL analysis windows, greedy disjoint, no direction or motion filtering",
        "hit_rule": "signed ramp DI>=.25, signed first-pair decoder displacement>=.01m, pair energy enrichment>=2, ramp active windows>=20, matched position bins>=2, each pair unit>=10%pair energy, pair>=50%allknown place energy",
    }
    print(json.dumps(shared_config), flush=True)
    all_summaries = []
    for seed in args.seeds:
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(exist_ok=True)
        set_seed(seed)
        sed_config = SedConfig(
            n_neurons=100,
            seq_len=seq_len,
            dsed_topk_map={128: 4},
            encoder=EncoderConfig(
                type=args.architecture,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_feedforward=256,
                causal=True,
            ),
            dtype=t.float32,
        )
        loss_config = LossConfig(timebin_weights=[1.0] * seq_len, type="mse")
        train_config = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.005, log_frequency=100
        )
        model = build_sed(sed_config).to(args.device)
        config = dict(
            shared_config,
            seed=seed,
            sed_config=asdict(sed_config),
            loss_config=asdict(loss_config),
            train_config=asdict(train_config),
            parameter_count=sum(p.numel() for p in model.parameters()),
        )
        config["sed_config"]["dtype"] = str(sed_config.dtype)
        config_text = json.dumps(config, indent=2)
        checkpoint_path = run_dir / "checkpoint.pt"
        if checkpoint_path.exists():
            if canonical_run_config(json.loads((run_dir / "config.json").read_text())) != canonical_run_config(config):
                raise ValueError(
                    "Existing run configuration differs; choose a fresh output directory"
                )
            model.load_state_dict(
                t.load(checkpoint_path, map_location=args.device, weights_only=True)[
                    "model_state_dict"
                ]
            )
        else:
            (run_dir / "config.json").write_text(config_text)
            loader = DataLoader(
                datasets["train"],
                batch_size=args.batch_size,
                shuffle=True,
                generator=t.Generator().manual_seed(seed),
            )
            history = train_model(model, loader, loss_config, train_config)
            pd.DataFrame(asdict(history)).rename_axis("step").to_csv(run_dir / "history.csv")
            t.save(
                {
                    "model_state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "seed": seed,
                    "config_json": config_text,
                },
                checkpoint_path,
            )
        decoder = model.decoder.weight.detach().cpu().numpy()
        structure = decoder_metrics(decoder, centers)
        saved_arrays = {
            "decoder": decoder,
            "known_centers": centers,
            "known_place_unit_ids": np.asarray(parameters["place_cell_ids"]),
            "seq_len": seq_len,
            "dt": dt * args.bin_factor,
            "raw_dt": dt,
            "raw_length": raw_length,
            "decoder_relative_time_s": (
                np.arange(seq_len) * args.bin_factor + (args.bin_factor - 1) / 2
            )
            * dt,
            "trajectory_relative_time_s": np.arange(raw_length) * dt,
        }
        selected = None
        summary = {"seed": seed, "architecture": args.architecture, "config": config}
        score_arrays = {}
        for split in ("analysis",):
            acts, mse, per_unit_error = evaluate(
                model, datasets[split], loss_config, args.batch_size
            )
            starts = metadata[split].start.to_numpy()
            metrics = matched_direction_metrics(
                acts, metadata[split], bins=40, eligible_column="ramp_clean"
            ).merge(
                structure, on="latent_idx"
            )
            global_metrics = matched_direction_metrics(acts, metadata[split]).rename(
                columns=lambda column: column if column == "latent_idx" else "global_" + column
            )
            metrics = metrics.merge(global_metrics, on="latent_idx")
            selected = select_features(metrics)
            metrics["is_directional_hit"] = metrics.right_passes | metrics.left_passes
            metrics.to_csv(run_dir / f"{split}_latent_metrics.csv", index=False)
            summary["selected_directions"] = {
                label: {
                    "latent_idx": feature,
                    "passes": bool(metrics.loc[feature, f"{label}_passes"]),
                    "selection_score": float(metrics.loc[feature, f"{label}_selection_score"]),
                }
                for label, feature in selected.items()
            }
            summary[f"{split}_mse"] = mse
            summary[f"{split}_mean_l0"] = float((acts > 0).sum(axis=1).mean())
            summary[f"{split}_directional_hit_count"] = int(metrics.is_directional_hit.sum())
            summary[f"{split}_selected_metrics"] = metrics[
                metrics.latent_idx.isin(selected.values())
            ].to_dict("records")
            score_arrays[f"{split}_activations"] = acts
            score_arrays[f"{split}_starts"] = starts
            score_arrays[f"{split}_per_unit_mse"] = per_unit_error
            for label, feature in selected.items():
                top = nonoverlapping_top_indices(acts[:, feature], starts, raw_length)
                saved_arrays[f"{label}_feature"] = feature
                saved_arrays[f"{label}_decoder"] = decoder[:, :, feature]
                saved_arrays[f"{label}_trajectories"] = positions[
                    starts[top, None] + np.arange(raw_length)
                ]
                saved_arrays[f"{label}_trajectory_starts"] = starts[top]
                saved_arrays[f"{label}_trajectory_clean"] = metadata[split].clean.to_numpy()[top]
                saved_arrays[f"{label}_trajectory_ramp_clean"] = metadata[split].ramp_clean.to_numpy()[top]
                saved_arrays[f"{label}_trajectory_activations"] = acts[top, feature]
                saved_arrays[f"{label}_trajectory_directions"] = metadata[
                    split
                ].direction.to_numpy()[top]
                # Activation weighted raw count coactivity and unconditional baseline.
                windows = np.lib.stride_tricks.sliding_window_view(counts, seq_len, axis=0)
                windows = rearrange(
                    windows[starts // args.bin_factor], "window neuron time -> window time neuron"
                )
                weights = acts[:, feature]
                coactivity = np.einsum("w,wtn->tn", weights, windows) / max(
                    float(weights.sum()), 1e-12
                )
                saved_arrays[f"{label}_raw_coactivity"] = coactivity
                saved_arrays[f"{label}_raw_coactivity_excess"] = coactivity - windows.mean(axis=0)
        summary["selected_features"] = selected
        np.savez_compressed(run_dir / "scores.npz", **score_arrays)
        np.savez_compressed(run_dir / "artifacts.npz", **saved_arrays)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries.append(summary)
        print(
            f"Completed {args.architecture} seed {seed}: {summary['analysis_mse']:.5f}, "
            f"hits={summary['analysis_directional_hit_count']}, selected={selected}",
            flush=True,
        )
    (output_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()

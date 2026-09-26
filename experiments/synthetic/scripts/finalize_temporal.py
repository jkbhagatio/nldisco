"""Add transparent example and coactivity diagnostics to finished temporal runs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from model_names import ARCHITECTURES, RUN_DIRECTORIES
from temporal_analysis import (
    decoder_metrics,
    matched_direction_metrics,
    nonoverlapping_top_indices,
    select_features,
)

from nldisco.config import canonical_encoder_type


def safe_fraction(values):
    """Return an explicit undefined value when a candidate has no active examples."""
    return float(np.mean(values)) if len(values) else None


def safe_correlation(first, second):
    """Return undefined for a constant decoder or coactivity signal."""
    first, second = first.ravel(), second.ravel()
    if np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def main():
    """Preserve all-window examples; supplement with direction-unfiltered ramp examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", type=canonical_encoder_type, choices=ARCHITECTURES, required=True
    )
    parser.add_argument("--null-shifts", type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = root / "experiments/synthetic/outputs/runs" / RUN_DIRECTORIES[args.architecture]
    positions = np.load(root / "experiments/synthetic/data/matryoshka_overlap/positions.npy")
    metadata = pd.read_csv(output / "analysis_windows.csv")
    all_summaries = []
    for summary_path in sorted(output.glob("seed_*/summary.json")):
        run = summary_path.parent
        summary = json.loads(summary_path.read_text())
        artifacts = dict(np.load(run / "artifacts.npz"))
        scores = np.load(run / "scores.npz")
        acts = scores["analysis_activations"]
        starts = scores["analysis_starts"]
        length = int(artifacts["raw_length"])
        observed_metrics = pd.read_csv(run / "analysis_latent_metrics.csv").set_index("latent_idx")
        structure = decoder_metrics(artifacts["decoder"], artifacts["known_centers"])
        rng = np.random.default_rng(10000 + summary["seed"])
        null_scores = {label: [] for label in ("right", "left")}
        offsets = rng.integers(length, len(acts) - length, size=args.null_shifts)
        for offset in offsets:
            shifted = np.roll(acts, int(offset), axis=0)
            metrics = matched_direction_metrics(
                shifted, metadata, bins=40, eligible_column="ramp_clean"
            ).merge(
                structure, on="latent_idx"
            )
            select_features(metrics)
            for label in null_scores:
                null_scores[label].append(float(metrics[f"{label}_selection_score"].max()))
        artifacts["null_shift_offsets_windows"] = offsets
        for label, feature in summary["selected_features"].items():
            sign = 1 if label == "right" else -1
            observed = observed_metrics.loc[feature]
            gates = {
                "signed_ramp_DI_at_least_0.25": sign * observed.right_selectivity >= .25,
                "signed_pair_decoder_shift_at_least_0.01m": sign * observed.decoder_center_displacement >= .01,
                "pair_noise_energy_enrichment_at_least_2": observed.place_energy_enrichment >= 2,
                "at_least_20_active_ramp_windows": observed.left_active_windows + observed.right_active_windows >= 20,
                "at_least_2_matched_position_bins": observed.matched_position_bins >= 2,
                "each_pair_unit_at_least_10pct_pair_energy": observed.pair_min_unit_energy_fraction >= .1,
                "pair_at_least_50pct_known_place_energy": observed.pair_known_place_energy_fraction >= .5,
            }
            clean_scores = np.where(metadata.ramp_clean.to_numpy(), acts[:, feature], 0)
            top = nonoverlapping_top_indices(clean_scores, starts, length)
            artifacts[f"{label}_clean_trajectories"] = positions[
                starts[top, None] + np.arange(length)
            ]
            artifacts[f"{label}_clean_trajectory_starts"] = starts[top]
            artifacts[f"{label}_clean_trajectory_activations"] = acts[top, feature]
            artifacts[f"{label}_clean_trajectory_directions"] = metadata.direction.to_numpy()[top]
            artifacts[f"{label}_clean_trajectory_scope"] = np.asarray("strict first-pair crossings with mean position between centers; no direction filter")
            decoder = artifacts[f"{label}_decoder"][:, :4]
            excess = artifacts[f"{label}_raw_coactivity_excess"][:, :4]
            directions = artifacts[f"{label}_trajectory_directions"]
            null = np.asarray(null_scores[label])
            artifacts[f"{label}_null_max_selection_scores"] = null
            selection_score = summary["selected_directions"][label]["selection_score"]
            summary["selected_directions"][label].update(
                {
                    "gate_diagnostics": {key: bool(value) for key, value in gates.items()},
                    "failed_gates": [key for key, value in gates.items() if not value],
                    "candidate_available": selection_score >= 0,
                    "all_top_expected_direction_fraction": safe_fraction(directions == sign),
                    "all_top_clean_fraction": safe_fraction(artifacts[f"{label}_trajectory_clean"]),
                    "all_top_ramp_clean_fraction": safe_fraction(artifacts[f"{label}_trajectory_ramp_clean"]),
                    "all_top_n": len(directions),
                    "all_top_position_min_m": float(artifacts[f"{label}_trajectories"].min()) if len(directions) else None,
                    "all_top_position_max_m": float(artifacts[f"{label}_trajectories"].max()) if len(directions) else None,
                    "clean_top_expected_direction_fraction": safe_fraction(metadata.direction.to_numpy()[top] == sign),
                    "clean_top_n": len(top),
                    "decoder_raw_place_coactivity_excess_correlation": safe_correlation(decoder, excess),
                    "decoder_raw_first_pair_coactivity_excess_correlation": safe_correlation(decoder[:, :2], excess[:, :2]),
                    "first_pair_decoder_peak_lag_s": float(
                        (np.argmax(decoder[:, 1]) - np.argmax(decoder[:, 0])) * artifacts["dt"]
                    ),
                    "first_pair_raw_excess_peak_lag_s": float(
                        (np.argmax(excess[:, 1]) - np.argmax(excess[:, 0])) * artifacts["dt"]
                    ),
                    "circular_shift_max_score_95pct": float(np.quantile(null, 0.95)),
                    "circular_shift_max_score_exceedance": float(
                        (1 + np.sum(null >= selection_score)) / (1 + len(null))
                    ),
                }
            )
        summary["supplementary_diagnostics"] = {
            "clean_examples": "greedy disjoint top16 ramp_clean windows: strict first-pair crossings, mean position between first two centers; both directions compete; primary examples still use ALLwindows",
            "null": f"{args.null_shifts} fixed-seed circular shifts of all dense activations relative to behavior; same ramp eligibility and maximum score across all128 latents per direction; descriptive reference only, not across-seed corrected",
            "decoder_motion_caution": "primary center displacement summarizes positive squared dictionary weights over first two known place cells; not a calibrated position or speed estimate",
            "position_confound_caution": "matching window mean position does not match the full trajectory or endpoint-specific firing; this position-only simulator can produce apparent directional sequences from temporal traversal of static fields",
        }
        np.savez_compressed(run / "artifacts.npz", **artifacts)
        summary_path.write_text(json.dumps(summary, indent=2))
        all_summaries.append(summary)
        print(summary["seed"], summary["selected_directions"], flush=True)
    (output / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    pd.DataFrame(
        [
            {
                "seed": summary["seed"],
                "analysis_mse": summary["analysis_mse"],
                "analysis_mean_l0": summary["analysis_mean_l0"],
                "analysis_directional_hit_count": summary["analysis_directional_hit_count"],
                "forward_passes": summary["selected_directions"]["right"]["passes"],
                "reverse_passes": summary["selected_directions"]["left"]["passes"],
            }
            for summary in all_summaries
        ]
    ).to_csv(output / "replica_summary.csv", index=False)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("train_window.py"),
        Path(__file__).with_name("temporal_analysis.py"),
        *sorted((root / "src/nldisco/model").glob("*.py")),
        root / "src/nldisco/train.py",
        root / "src/nldisco/config.py",
    ]
    provenance = {
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seeding": "Python, NumPy, torch CPU/CUDA seeded; DataLoader generator seeded; deterministic CUDA algorithms not forced",
        "encoder_positions": "flat linear projection"
        if args.architecture == "FlatWindow"
        else "learned absolute positions; causal attention; last-token readout; global sparse vector",
        "hardware": "NVIDIA H10080GB; physical GPU1; two CPU threads per worker",
        "reproduce": [
            "CUDA_VISIBLE_DEVICES=1 uv run python experiments/synthetic/scripts/train_window.py --architecture "
            + args.architecture
            + " --seeds 0 1 2 3 4",
            "uv run python experiments/synthetic/scripts/finalize_temporal.py --architecture "
            + args.architecture,
        ],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

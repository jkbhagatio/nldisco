"""Aggregate ten run artifacts and retain the five best runs per feature class."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "experiments/synthetic/outputs/runs/vanilla"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "summary"
TARGETS = tuple(f"cell_{unit}" for unit in range(4))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-selected", type=int, default=5)
    return parser.parse_args()


def _best_row(scores: pd.DataFrame, target: str) -> pd.Series:
    return scores.sort_values(
        [f"{target}_hit", f"{target}_score", "latent_idx"], ascending=[False, False, True]
    ).iloc[0]


def aggregate(runs_dir: Path, output_dir: Path, n_runs: int, n_selected: int) -> None:
    """Validate run artifacts and save compact, notebook-only plotting artifacts."""

    run_dirs = sorted(path for path in runs_dir.glob("seed_*") if path.is_dir())
    if len(run_dirs) != n_runs:
        raise ValueError(f"Expected exactly {n_runs} run directories, found {len(run_dirs)}")
    if not 1 <= n_selected <= n_runs:
        raise ValueError("n_selected must be between one and n_runs")

    summaries: List[Dict[str, object]] = []
    best_rows = []
    diagnostics = []
    curves_by_seed: Dict[int, np.ndarray] = {}
    traces_by_seed = {}
    position_centers = None
    selection_config = None
    true_curves = None
    any_hit_counts = dict.fromkeys(TARGETS, 0)
    joint_hit_seeds = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        seed = int(summary["seed"])
        if seed in curves_by_seed:
            raise ValueError(f"Duplicate replica seed: {seed}")
        if summaries:
            for key in ("dataset_sha256", "model_config", "data_usage"):
                if summary.get(key) != summaries[0].get(key):
                    raise ValueError(f"All runs must use identical {key}")
        scores = pd.read_csv(run_dir / "feature_scores.csv")
        artifacts = np.load(run_dir / "tuning_curves.npz")
        run_centers = artifacts["position_centers"]
        if position_centers is None:
            position_centers = run_centers
            selection_config = summary["selection_config"]
            true_curves = artifacts["ground_truth_curves"]
        elif not np.array_equal(position_centers, run_centers):
            raise ValueError("All runs must use identical position bins")
        if summary["selection_config"] != selection_config:
            raise ValueError("All runs must use an identical selection configuration")
        if not np.array_equal(true_curves, artifacts["ground_truth_curves"]):
            raise ValueError("All runs must use identical ground-truth shapes")
        curves_by_seed[seed] = artifacts["normalized_tuning_curves"]
        with np.load(run_dir / "traceback.npz") as traces:
            traces_by_seed[seed] = {
                key: traces[key]
                for key in ("conditional_zscore", "activation_weighted_zscore", "decoder_weights")
            }
        if all(scores[f"{target}_hit"].any() for target in TARGETS):
            joint_hit_seeds.append(seed)
        if (scores[[f"{target}_hit" for target in TARGETS]].sum(axis=1) > 1).any():
            raise ValueError("A latent cannot identify more than one biological cell")
        for unit, target in enumerate(TARGETS):
            expected_units = [unit]
            diagnostics.append(
                {
                    "seed": seed,
                    "target": target,
                    "n_shape_passing": int(
                        scores.get(f"{target}_shape_hit", scores[f"{target}_hit"]).sum()
                    ),
                    "n_biology_passing": int(
                        scores.get(f"{target}_biology_hit", scores[f"{target}_hit"]).sum()
                    ),
                    "n_all_criteria_passing": int(scores[f"{target}_hit"].sum()),
                    "largest_min_expected_unit_decoder_loading": float(
                        traces_by_seed[seed]["decoder_weights"][:, expected_units]
                        .min(axis=1)
                        .max()
                    ),
                }
            )
        summaries.append(summary)
        for target in TARGETS:
            any_hit_counts[target] += int(scores[f"{target}_hit"].any())
            row = _best_row(scores, target)
            best_rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "latent_idx": int(row["latent_idx"]),
                    "score": float(row[f"{target}_score"]),
                    "hit": bool(row[f"{target}_hit"]),
                    "peak_position": float(row["peak_position"]),
                    "outside_mean": float(row[f"{target}_outside_mean"]),
                    "peak_distance": float(row[f"{target}_peak_distance"]),
                    "weighted_discovery_reconstruction": float(
                        summary["weighted_discovery_reconstruction"]
                    ),
                }
            )

    all_best = pd.DataFrame(best_rows).sort_values(["target", "score"], ascending=[True, False])
    selected_rows = []
    payload: Dict[str, np.ndarray] = {
        "position_centers": position_centers,
        "ground_truth_curves": true_curves,
    }
    hit_rates = {}
    for target in TARGETS:
        target_rows = all_best.query("target == @target").sort_values(
            ["hit", "score", "seed"], ascending=[False, False, True]
        )
        selected = target_rows.loc[target_rows["hit"]].head(n_selected).copy()
        selected_rows.append(selected)
        hit_rates[target] = int(target_rows["hit"].sum())
        payload[f"{target}_curves"] = np.asarray(
            [curves_by_seed[int(row.seed)][int(row.latent_idx)] for row in selected.itertuples()]
        ).reshape(-1, len(position_centers))
        payload[f"{target}_seeds"] = selected["seed"].to_numpy(dtype=np.int64)
        payload[f"{target}_latent_indices"] = selected["latent_idx"].to_numpy(dtype=np.int64)
        payload[f"{target}_scores"] = selected["score"].to_numpy(dtype=np.float64)
        for key in ("conditional_zscore", "activation_weighted_zscore", "decoder_weights"):
            payload[f"{target}_{key}"] = np.asarray(
                [
                    traces_by_seed[int(row.seed)][key][int(row.latent_idx)]
                    for row in selected.itertuples()
                ]
            ).reshape(-1, 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_best.to_csv(output_dir / "best_features_all_runs.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "selection_diagnostics.csv", index=False)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected.to_csv(output_dir / "selected_features.csv", index=False)
    np.savez_compressed(output_dir / "final_features.npz", **payload)
    final_summary = {
        "architecture": summaries[0]["architecture"],
        "model_config": summaries[0]["model_config"],
        "dataset_sha256": summaries[0].get("dataset_sha256"),
        "data_usage": summaries[0]["data_usage"],
        "decoder_loading_definition": summaries[0]["decoder_loading_definition"],
        "traceback_definition": summaries[0]["traceback_definition"],
        "n_trained_runs": n_runs,
        "n_selected_per_target": {
            target: int((selected["target"] == target).sum()) for target in TARGETS
        },
        "max_selected_per_target": n_selected,
        "selection_policy": (
            "Choose each run's best passing latent; rank passing runs by shape score and "
            "retain up to the requested maximum. No nonpassing fallback. Training, "
            "selection and description use the same full simulation."
        ),
        "curve_normalization": "Each selected curve is independently divided by its peak.",
        "shading": "Pointwise mean +/- 1 SD across the selected independent runs.",
        "hit_counts": hit_rates,
        "hit_counts_definition": "Runs with any passing latent on the full discovery dataset.",
        "all_four_joint_hit_count": len(joint_hit_seeds),
        "all_four_joint_hit_seeds": sorted(joint_hit_seeds),
        "any_hit_counts": any_hit_counts,
        "selection_config": selection_config,
        "seeds": sorted(curves_by_seed),
    }
    (output_dir / "summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(selected.to_string(index=False))
    print(json.dumps(final_summary, indent=2))


def main() -> None:
    args = parse_args()
    aggregate(args.runs_dir, args.output_dir, args.n_runs, args.n_selected)


if __name__ == "__main__":
    main()

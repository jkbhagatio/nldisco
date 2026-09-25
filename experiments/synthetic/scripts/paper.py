"""Reproduce synthetic main/appendix figures from the overlapping-field saved runs.

Only outputs/runs holds model artifacts; outputs/paper holds generated reports.
All analyses describe the same full recording used for training.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxtyping import Bool, Float
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments.synthetic.scripts.dataset_metadata import load_parameters
from experiments.synthetic.scripts.model_names import ARCHITECTURES, PAPER_LABELS, RUN_DIRECTORIES
from nldisco.config import canonical_encoder_type

SYNTHETIC = Path(__file__).resolve().parents[1]
DATA = SYNTHETIC / "data/matryoshka_overlap"
RUNS = SYNTHETIC / "outputs/runs"
PAPER = SYNTHETIC / "outputs/paper"
COLOR = "#33658A"




def select_matryoshka_runs(summaries: list) -> list:
    """Keep the original four joint hits and add the strongest new joint hit."""
    by_seed = {item["seed"]: item for item in summaries}
    if len(by_seed) != len(summaries) or set(by_seed) != set(range(7)):
        raise ValueError("Expected seven unique Matryoshka seeds, 0 through 6")

    def passes(seed):
        pair = by_seed[seed]["levels"]["128"]["selected"]
        broad, narrow = pair["cell_2"], pair["cell_3"]
        return (broad["hit"] and narrow["hit"]
                and broad["latent_idx"] != narrow["latent_idx"])

    retained = [0, 1, 2, 3]
    if not all(passes(seed) for seed in retained):
        raise ValueError("An original qualifying Matryoshka pair no longer passes")
    candidates = [seed for seed in (5, 6) if passes(seed)]
    if not candidates:
        raise ValueError("Neither new Matryoshka run passes; cannot show five qualifying runs")
    added = min(candidates, key=lambda seed: (
        -min(by_seed[seed]["levels"]["128"]["selected"][f"cell_{cell}"]["score"]
             for cell in (2, 3)), seed))
    return [by_seed[seed] for seed in retained + [added]]


def load_results() -> Dict[str, object]:
    """Load completed runs and validate that every model used the same dataset."""
    params = load_parameters(DATA)
    if (params["n_units"], params["n_samples"], params["sampling_rate_hz"]) != (100, 60000, 100):
        raise ValueError("The paper report requires the four-place-cell/96-noise design")
    fingerprint = params["files"]["spike_matrix.npy"]["sha256"]
    matryoshka_paths = sorted((RUNS / "matryoshka_overlap").glob("seed_*/summary.json"))
    matryoshka_summaries = []
    for path in matryoshka_paths:
        summary = json.loads(path.read_text())
        if summary["config"]["dataset_sha256"] != fingerprint:
            raise ValueError(f"Dataset mismatch: {path}")
        matryoshka_summaries.append(summary)
    selected_matryoshka = select_matryoshka_runs(matryoshka_summaries)
    pair_curves, identities = [], []
    reference_centers = reference_truth = None
    for summary in selected_matryoshka:
        seed = summary["seed"]
        pair = summary["levels"]["128"]["selected"]
        features = [int(pair[f"cell_{cell}"]["latent_idx"]) for cell in (2, 3)]
        with np.load(RUNS / f"matryoshka_overlap/seed_{seed:02d}/level_128_artifacts.npz") as artifact:
            centers, truth = artifact["position_centers"], artifact["ground_truth_curves"]
            if reference_centers is not None:
                np.testing.assert_array_equal(centers, reference_centers)
                np.testing.assert_array_equal(truth, reference_truth)
            reference_centers, reference_truth = centers, truth
            curves = artifact["curves"][features]
            if not np.isfinite(curves).all() or np.any(curves.max(axis=1) <= 0):
                raise ValueError("Selected Matryoshka curves must be finite and active")
            pair_curves.append(curves / curves.max(axis=1, keepdims=True))
        identities.append({"seed": seed, "broad_latent": features[0], "narrow_latent": features[1],
                           "minimum_pair_score": min(pair[f"cell_{cell}"]["score"] for cell in (2, 3))})
    matryoshka = {"position_centers": reference_centers, "ground_truth_curves": reference_truth,
                 "selected_curves": np.stack(pair_curves)}
    matryoshka_selection = {
        "n_trained_runs": len(matryoshka_summaries), "n_selected_runs": 5, "prefix": 128,
        "levels": [16, 32, 64, 128], "selected": identities,
        "joint_passing_seeds": [s["seed"] for s in matryoshka_summaries
                                if s["levels"]["128"]["both_cells_recovered"]],
        "policy": "Retain seeds 0/1/2/3; add highest minimum-pair-score joint hit among seeds 5/6, ties by lower seed",
        "band": "Pointwise mean ±1 sample SD (ddof=1), independently peak-normalized curves; shading clipped to [0,1]",
        "scope": "Selected successful runs on the full training recording, not uncertainty over all seven runs",
    }
    vanilla_dir = RUNS / "vanilla/summary"
    vanilla_summary = json.loads((vanilla_dir / "summary.json").read_text())
    if vanilla_summary["n_trained_runs"] != 10:
        raise ValueError("Expected the full ten-seed vanilla experiment")
    vanilla_paths = sorted((RUNS / "vanilla").glob("seed_*/summary.json"))
    if len(vanilla_paths) != 10:
        raise ValueError("Expected ten completed vanilla run folders")
    for path in vanilla_paths:
        if json.loads(path.read_text())["dataset_sha256"] != fingerprint:
            raise ValueError(f"Dataset mismatch: {path}")
    temporal_runs = {}
    for architecture in ARCHITECTURES:
        records = []
        for path in sorted((RUNS / RUN_DIRECTORIES[architecture]).glob("seed_*/summary.json")):
            summary = json.loads(path.read_text())
            summary["architecture"] = canonical_encoder_type(summary["architecture"])
            if summary["config"]["spikes_sha256"] != fingerprint:
                raise ValueError(f"Dataset mismatch: {path}")
            records.append({
                "summary": summary,
                "artifacts": dict(np.load(path.parent / "artifacts.npz")),
                "path": path.parent,
            })
        if len(records) != 5:
            raise ValueError(f"Expected five completed {architecture} runs, found {len(records)}")
        temporal_runs[architecture] = records
    matryoshka_path = RUNS / "matryoshka_overlap/seed_02"
    matryoshka_summary = json.loads((matryoshka_path / "summary.json").read_text())
    if matryoshka_summary["config"]["dataset_sha256"] != fingerprint:
        raise ValueError("Retained Matryoshka model used a different dataset")
    pair = matryoshka_summary["levels"]["128"]["selected"]
    for cell, feature in ((2, 3), (3, 73)):
        if not pair[f"cell_{cell}"]["hit"] or pair[f"cell_{cell}"]["latent_idx"] != feature:
            raise ValueError("Retained Matryoshka example has changed")
    return {
        "matryoshka": matryoshka,
        "matryoshka_selection": matryoshka_selection,
        "matryoshka_example": {"seed": 2, "prefix": 128, "cell_2_latent": 3,
                               "cell_3_latent": 73, "dataset_sha256": fingerprint,
                               "status": "Retained existing trained model; not reselected"},
        "parameters": params,
        "positions": np.load(DATA / "positions.npy"),
        "timestamps": np.load(DATA / "timestamps.npy"),
        "empirical_rates": np.load(DATA / "empirical_rate_maps.npy"),
        "empirical_edges": np.load(DATA / "empirical_rate_bin_edges.npy"),
        "theoretical_rates": np.load(DATA / "theoretical_rate_maps.npy"),
        "theoretical_positions": np.load(DATA / "theoretical_rate_positions.npy"),
        "vanilla": dict(np.load(vanilla_dir / "final_features.npz")),
        "vanilla_summary": vanilla_summary,
        "selected": pd.read_csv(vanilla_dir / "selected_features.csv"),
        "temporal": temporal_runs,
    }


def temporal_records(results: Dict[str, object]) -> pd.DataFrame:
    """Return each seed's selected directional metrics, including failed candidates."""
    rows = []
    for architecture, records in results["temporal"].items():
        for record in records:
            summary = record["summary"]
            metric_map = {
                int(row["latent_idx"]): row for row in summary["analysis_selected_metrics"]
            }
            for direction in ("right", "left"):
                selection = summary["selected_directions"][direction]
                feature = int(selection["latent_idx"])
                metric = metric_map[feature]
                rows.append({
                    "architecture": architecture,
                    "seed": int(summary["seed"]),
                    "direction": direction,
                    "latent_idx": feature,
                    "passes": bool(selection["passes"]),
                    "candidate_available": bool(selection.get("candidate_available", True)),
                    "selection_score": float(selection["selection_score"]),
                    "direction_selectivity": float(metric["right_selectivity"]),
                    "global_direction_selectivity": float(metric["global_right_selectivity"]),
                    "decoder_displacement_m": float(metric["decoder_center_displacement"]),
                    "place_energy_enrichment": float(metric["place_energy_enrichment"]),
                    "active_clean_windows": int(metric["left_active_windows"] + metric["right_active_windows"]),
                    "mse": float(summary["analysis_mse"]),
                    "mean_l0": float(summary["analysis_mean_l0"]),
                    "parameter_count": int(summary["config"]["parameter_count"]),
                    **{key: selection[key] for key in (
                        "all_top_expected_direction_fraction",
                        "all_top_clean_fraction",
                        "clean_top_expected_direction_fraction",
                        "decoder_raw_place_coactivity_excess_correlation",
                        "circular_shift_max_score_95pct",
                        "circular_shift_max_score_exceedance",
                    )},
                    **{key: selection.get(key) for key in (
                        "all_top_ramp_clean_fraction",
                        "decoder_raw_first_pair_coactivity_excess_correlation",
                        "first_pair_decoder_peak_lag_s",
                        "first_pair_raw_excess_peak_lag_s",
                    )},
                })
    return pd.DataFrame(rows)


def choose_temporal(results: Dict[str, object], architecture: str, direction: str,
                    allow_diagnostic: bool = False):
    """Select passing runs first; permit an explicitly requested failed diagnostic."""
    candidates = []
    for record in results["temporal"][architecture]:
        selection = record["summary"]["selected_directions"][direction]
        if selection["passes"] and selection.get("candidate_available", True):
            candidates.append(record)
    if not candidates:
        if not allow_diagnostic:
            return None
        candidates = [record for record in results["temporal"][architecture]
                      if record["summary"]["selected_directions"][direction].get(
                          "candidate_available", True)]
        if not candidates:
            return None
    return max(candidates, key=lambda record: (
        record["summary"]["selected_directions"][direction]["selection_score"],
        -int(record["summary"]["seed"]),
    ))


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def choose_overall_temporal(results: Dict[str, object], direction: str):
    """Choose the strongest passing fixed score across both temporal architectures."""
    candidates = [choose_temporal(results, architecture, direction)
                  for architecture in ARCHITECTURES]
    candidates = [record for record in candidates if record is not None]
    return max(candidates, key=lambda record: (
        record["summary"]["selected_directions"][direction]["selection_score"],
        -int(record["summary"]["seed"]),
    )) if candidates else None


def _model_name(record) -> str:
    return PAPER_LABELS[canonical_encoder_type(record["summary"]["architecture"])]


def _save(fig: plt.Figure, name: str) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(PAPER / f"{name}.{suffix}", dpi=300, bbox_inches="tight")


def _missing(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _spatial(ax, results, cell: int, show_ground_truth: bool = False,
             mean_linewidth: float = 2.2) -> None:
    data = results["vanilla"]
    target = f"cell_{cell}"
    curves = data[f"{target}_curves"]
    center = results["parameters"]["centers_m"][cell]
    title = f"Place-field recovery: {center:.1f} m"
    ax.set(xlabel="Position along track (m)", ylabel="Normalized mean activation",
           title=title, xlim=(0, 1), ylim=(0, 1.04))
    if not len(curves):
        _missing(ax, "No feature passed the\nspatial and biological criteria")
        return
    positions = data["position_centers"]
    if show_ground_truth:
        ax.plot(positions, data["ground_truth_curves"][cell], color="#222222",
                linestyle="--", linewidth=1.1, label="Generative field")
    for curve in curves:
        ax.plot(positions, curve, color=COLOR, alpha=0.24, linewidth=0.9)
    mean = curves.mean(axis=0)
    if len(curves) > 1:
        sd = curves.std(axis=0, ddof=1)
        ax.fill_between(positions, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                        color=COLOR, alpha=0.2, linewidth=0)
    ax.plot(positions, mean, color=COLOR, linewidth=mean_linewidth)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", alpha=0.18)


def _decoder(fig, ax, record, direction: str, title_prefix: str = "") -> None:
    if record is None:
        _missing(ax, "No qualifying directional feature")
        return
    data = record["artifacts"]
    matrix = data[f"{direction}_decoder"][:, data["known_place_unit_ids"]].T
    matrix = matrix / max(float(np.abs(matrix).max()), 1e-12)
    raw_time = data["trajectory_relative_time_s"]
    decoder_time = data["decoder_relative_time_s"] - raw_time[-1]
    dt = float(data["dt"])
    edges = np.r_[decoder_time - dt / 2, decoder_time[-1] + dt / 2]
    image = ax.pcolormesh(edges, np.arange(matrix.shape[0] + 1), matrix,
                          shading="flat", cmap="viridis", vmin=-1, vmax=1, rasterized=True)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_yticklabels([f"{center:.1f}" for center in data["known_centers"]])
    feature = int(data[f"{direction}_feature"])
    label = "Rightward" if direction == "right" else "Leftward"
    ax.set(xlabel="Time from window end (s)", ylabel="Place-field center (m)",
           title=f"{title_prefix}{label}: latent {feature}, seed {record['summary']['seed']}")
    fig.colorbar(image, ax=ax, label="Normalized decoder weight", fraction=0.035, pad=0.02)


def _trajectories(ax, record, direction: str, ramp_conditioned: bool = False) -> None:
    if record is None:
        _missing(ax, "No qualifying directional feature")
        return
    data = record["artifacts"]
    scope = "clean_" if ramp_conditioned else ""
    trajectories = data[f"{direction}_{scope}trajectories"]
    times = data["trajectory_relative_time_s"]
    times = times - times[-1]
    if not len(trajectories):
        _missing(ax, "No active example windows")
        return
    for trajectory in trajectories:
        ax.plot(times, trajectory, color="#6b7280", alpha=0.25, linewidth=0.8)
    mean = trajectories.mean(axis=0)
    if len(trajectories) > 1:
        sd = trajectories.std(axis=0, ddof=1)
        ax.fill_between(times, mean - sd, mean + sd, color=COLOR, alpha=0.18, linewidth=0)
    ax.plot(times, mean, color=COLOR, linewidth=2.2)
    # Fit all selected examples; no clipping to a desired position/direction.
    lower = max(0, float(trajectories.min()) - 0.04)
    upper = min(1, float(trajectories.max()) + 0.04)
    title = "Top clean ramp crossings" if ramp_conditioned else "Highest-activating windows"
    ax.set(xlabel="Time from window end (s)", ylabel="Position (m)",
           ylim=(lower, upper), title=f"{title} (n={len(trajectories)})")


def _paired_spatial(ax, results) -> None:
    """Overlay independently selected five-run recoveries and exact target fields."""
    data = results["vanilla"]
    positions = data["position_centers"]
    for cell, color in enumerate((COLOR, "#D97706")):
        curves = data[f"cell_{cell}_curves"]
        if len(curves) != 5:
            raise ValueError(f"Main panel c requires five qualifying runs for cell {cell + 1}")
        mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        ax.fill_between(positions, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                        color=color, alpha=0.20, linewidth=0)
        ax.plot(positions, mean, color=color, linewidth=0.8, label=f"Cell {cell + 1}")
        ax.plot(positions, data["ground_truth_curves"][cell], color=color,
                linestyle="--", linewidth=1.0)
    ax.set(xlabel="Position along track (m)", ylabel="Normalized activation / rate",
           title="Vanilla SED: cells 1 and 2", xlim=(0, 1), ylim=(0, 1.04))
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.18)


def _overlapping_spatial(ax, results) -> None:
    """Plot five jointly qualifying Matryoshka pairs with sample SD bands."""
    data = results["matryoshka"]
    curves = data["selected_curves"]
    if curves.shape != (5, 2, len(data["position_centers"])) or not np.isfinite(curves).all():
        raise ValueError("Matryoshka panel requires five qualifying paired curves")
    for pair_index, cell, color, label in ((0, 2, COLOR, "Broad cell 3"),
                                         (1, 3, "#D97706", "Narrow cell 4")):
        mean = curves[:, pair_index].mean(axis=0)
        sd = curves[:, pair_index].std(axis=0, ddof=1)
        ax.fill_between(data["position_centers"], np.clip(mean - sd, 0, 1),
                        np.clip(mean + sd, 0, 1), color=color, alpha=0.20, linewidth=0)
        ax.plot(data["position_centers"], mean, color=color, linewidth=0.8, label=label)
        ax.plot(data["position_centers"], data["ground_truth_curves"][cell],
                color=color, linestyle="--", linewidth=1.0)
    ax.set(xlabel="Position along track (m)", ylabel="Normalized activation / rate",
           title="Matryoshka SED: cells 3 and 4", xlim=(0, 1), ylim=(0, 1.04))
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.18)


def matryoshka_figure(results: Dict[str, object]) -> plt.Figure:
    """Save a standalone version of the five-run Figure 3d panel."""
    _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
    _overlapping_spatial(ax, results)
    _save(fig, "figure3d_matryoshka")
    return fig


def main_figure(results: Dict[str, object]) -> plt.Figure:
    """Render two input panels above three complementary feature examples."""
    _style()
    fig = plt.figure(figsize=(10, 6.3), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:]),
            fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[1, 4:])]
    axes[0].plot(results["timestamps"], results["positions"], color=COLOR, lw=0.5)
    axes[0].set(xlabel="Time (s)", ylabel="Position (m)", ylim=(0, 1),
                   xlim=(0, 600), title="Simulated position on the 1-m track")
    rates = results["empirical_rates"]
    image = axes[1].pcolormesh(results["empirical_edges"], np.arange(5), rates,
                                  cmap="viridis", vmin=0, shading="flat", rasterized=True)
    axes[1].set_yticks(np.arange(4) + 0.5)
    axes[1].set_yticklabels(np.arange(1, 5))
    axes[1].set(xlabel="Position along track (m)", ylabel="Place-cell index",
                   title="Place-cell firing-rate heatmap")
    fig.colorbar(image, ax=axes[1], label="Firing rate (Hz)", fraction=0.035, pad=0.02)
    _paired_spatial(axes[2], results)
    _overlapping_spatial(axes[3], results)
    record = choose_overall_temporal(results, "right")
    _trajectories(axes[4], record, "right", ramp_conditioned=True)
    if record is not None:
        count = len(record["artifacts"]["right_clean_trajectories"])
        axes[4].set_title("Position over time for activating\n"
                          f"examples (n={count}) of a {_model_name(record)} latent")
    for letter, ax in zip("abcde", axes):
        ax.text(-0.14, 1.05, letter, transform=ax.transAxes, weight="bold", fontsize=13)
    _save(fig, "figure3")
    return fig


def spatial_figure(results: Dict[str, object]) -> plt.Figure:
    """Show recovery of all four known fields without failed substitutes."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    for cell, ax in enumerate(axes.flat):
        _spatial(ax, results, cell, show_ground_truth=True)
    axes[0, 0].legend(handles=[
        Line2D([], [], color="#222222", linestyle="--", label="Generative field"),
        Line2D([], [], color=COLOR, alpha=0.3, label="Individual run"),
        Line2D([], [], color=COLOR, linewidth=2, label="Mean"),
        Patch(color=COLOR, alpha=0.2, label="±1 SD"),
    ], frameon=False, fontsize=7, loc="upper right")
    _save(fig, "supplement_all_place_fields")
    return fig












def transformer_figure(results: Dict[str, object]) -> plt.Figure:
    """Compare the forward temporal examples under matched training settings."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    for row, architecture in enumerate(ARCHITECTURES):
        record = choose_temporal(results, architecture, "right", allow_diagnostic=True)
        name = PAPER_LABELS[architecture]
        if record is not None and not record["summary"]["selected_directions"]["right"]["passes"]:
            name += " (diagnostic: fails fixed criteria)"
        _decoder(fig, axes[row, 0], record, "right", title_prefix=name + "\n")
        _trajectories(axes[row, 1], record, "right")
        if record is None:
            axes[row, 0].set_title(name)
    _save(fig, "supplement_temporal_comparison")
    return fig


def write_report(results: Dict[str, object]) -> Dict[str, object]:
    """Save manuscript-facing statistics, selected example identities, and tables."""
    PAPER.mkdir(parents=True, exist_ok=True)
    records = temporal_records(results)
    records.to_csv(PAPER / "temporal_results.csv", index=False)
    results["selected"].to_csv(PAPER / "spatial_selected_features.csv", index=False)
    temporal_summary = {}
    for architecture in ARCHITECTURES:
        frame = records[records.architecture == architecture]
        seeds = frame.drop_duplicates("seed")
        selected_examples = {}
        for direction in ("right", "left"):
            record = choose_temporal(results, architecture, direction)
            selected_examples[direction] = (
                None if record is None else {
                    "seed": int(record["summary"]["seed"]),
                    **record["summary"]["selected_directions"][direction],
                }
            )
        diagnostic = choose_temporal(results, architecture, "right", allow_diagnostic=True)
        diagnostic_selection = (
            None if diagnostic is None else diagnostic["summary"]["selected_directions"]["right"]
        )
        temporal_summary[architecture] = {
            "n_runs": len(seeds),
            "right_hit_runs": int(frame[frame.direction == "right"].passes.sum()),
            "left_hit_runs": int(frame[frame.direction == "left"].passes.sum()),
            "mean_mse": float(seeds.mse.mean()),
            "sd_mse": float(seeds.mse.std(ddof=1)),
            "mean_l0": float(seeds.mean_l0.mean()),
            "parameter_count": int(seeds.parameter_count.iloc[0]),
            "selected_examples": selected_examples,
            "displayed_diagnostic_examples": (
                {} if diagnostic_selection is None or diagnostic_selection["passes"] else {"right": {
                    "seed": int(diagnostic["summary"]["seed"]),
                    **diagnostic_selection,
                }}
            ),
            "config": results["temporal"][architecture][0]["summary"]["config"],
        }
    report = {
        "analysis_scope": "Full-recording descriptive feature discovery; no held-out split",
        "earlier_dataset_fallback_used": False,
        "dataset": results["parameters"],
        "vanilla": results["vanilla_summary"],
        "matryoshka_example": results["matryoshka_example"],
        "matryoshka_selection": results["matryoshka_selection"],
        "temporal": temporal_summary,
        "overall_temporal_examples": {
            direction: (None if choose_overall_temporal(results, direction) is None else {
                "architecture": choose_overall_temporal(results, direction)["summary"]["architecture"],
                "seed": choose_overall_temporal(results, direction)["summary"]["seed"],
                **choose_overall_temporal(results, direction)["summary"]["selected_directions"][direction],
            }) for direction in ("right", "left")
        },
        "figure_definitions": {
            "main_layout": "Two input panels above five-run vanilla cells 1/2, five-run Matryoshka cells 3/4, and rightward temporal examples",
            "main_spatial_selection": "Five qualifying runs independently selected per first-pair cell; matching-color dashed exact generative fields",
            "spatial_band": "Mean ±1 sample SD across up to five qualifying independent runs; clip [0,1]",
            "trajectory_band": "Mean ±1 sample SD across highest-activating disjoint windows",
            "main_trajectory_selection": "Top disjoint strict first-pair crossings; both directions compete; no requested-direction filter",
            "unrestricted_trajectory_selection": "Supplementary top disjoint windows with no position, direction, or motion filtering",
            "direction_metric_scope": "Position-matched strict first-pair traversals; global DI is reported separately",
            "main_temporal_selection": "Highest passing frozen score across both architectures and all ten temporal runs",
            "conditioned_trajectory_selection": "Main rightward and supplementary bidirectional top disjoint strict first-pair crossings, both directions competing",
            "traceback": "Unweighted mean biological-unit z-score conditional on latent activation",
            "top_heatmap": results["parameters"]["empirical_rate_maps"],
            "decoder": "Signed weights normalized by maximum absolute weight among four place units",
        },
    }
    (PAPER / "results_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    selection = results["matryoshka_selection"]
    pd.DataFrame(selection["selected"]).to_csv(PAPER / "matryoshka_selected_features.csv", index=False)
    np.savez_compressed(PAPER / "matryoshka_selected_features.npz",
                        **results["matryoshka"],
                        seeds=np.array([item["seed"] for item in selection["selected"]]),
                        latent_ids=np.array([[item["broad_latent"], item["narrow_latent"]]
                                             for item in selection["selected"]]),
                        mean=results["matryoshka"]["selected_curves"].mean(axis=0),
                        sample_sd=results["matryoshka"]["selected_curves"].std(axis=0, ddof=1))
    inputs = [DATA / "parameters.json", DATA / "spike_matrix.npy",
              RUNS / "vanilla/summary/final_features.npz",
              RUNS / "vanilla/summary/selected_features.csv",
              RUNS / "matryoshka_overlap/extension_plan.json"]
    inputs += [RUNS / f"matryoshka_overlap/seed_{item['seed']:02d}/level_128_artifacts.npz"
               for item in results["matryoshka_selection"]["selected"]]
    inputs += sorted(RUNS.glob("*/seed_*/checkpoint.pt"))
    inputs += sorted(RUNS.glob("*/seed_*/summary.json"))
    inputs += sorted(RUNS.glob("*/seed_*/artifacts.npz"))
    inputs += sorted(DATA.glob("*.npy"))
    inputs += [RUNS / "vanilla/summary/summary.json",
               RUNS / "matryoshka_overlap/recovery.csv",
               RUNS / "matryoshka_overlap/summary.json"]
    inputs += sorted((RUNS / "matryoshka_overlap/seed_02").glob("level_*_artifacts.npz"))
    inputs += sorted((SYNTHETIC / "scripts").glob("*.py"))
    manifest = {str(path.relative_to(SYNTHETIC)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in inputs}
    (PAPER / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report

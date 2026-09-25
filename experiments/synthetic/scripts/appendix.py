"""Regenerate only the figures and diagnostics used in synthetic appendix 6.4.1."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype

from experiments.synthetic.scripts import paper
from experiments.synthetic.scripts.temporal_analysis import matched_direction_metrics




@beartype
def selected_features(results: dict) -> dict:
    """Keep the same TW-SED examples throughout S6, S7, and S8."""
    records = {}
    for direction, seed, feature in (("right", 0, 20), ("left", 2, 66)):
        record = paper.choose_temporal(results, "TransformerWindow", direction)
        if (record is None or record["summary"]["seed"] != seed
                or int(record["artifacts"][f"{direction}_feature"]) != feature):
            raise ValueError("The prespecified TW-SED appendix examples have changed")
        records[direction] = record
    return records


@beartype
def place_field_figure(results: dict) -> plt.Figure:
    """Show the four generative fields, without the former population-sum panel."""
    paper._style()
    fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
    for cell, center in enumerate(results["parameters"]["centers_m"]):
        ax.plot(results["theoretical_positions"], results["theoretical_rates"][cell],
                color=plt.colormaps["viridis"](center), linestyle=":" if cell == 3 else "-",
                label=f"Cell {cell + 1}: {center:.1f} m")
    ax.set(xlabel="Position (m)", ylabel="Expected firing rate (Hz)",
           title="Designed position-only place fields", xlim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    paper._save(fig, "supplement_rate_design")
    return fig


@beartype
def temporal_features_figure(results: dict) -> plt.Figure:
    """Show both TW-SED decoder patterns and unrestricted top-activation windows."""
    paper._style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), layout="constrained")
    for row, (direction, record) in enumerate(selected_features(results).items()):
        paper._decoder(fig, axes[row, 0], record, direction, "TW-SED\n")
        paper._trajectories(axes[row, 1], record, direction)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, weight="bold", fontsize=12)
    paper._save(fig, "supplement_temporal_features")
    return fig


@beartype
def clean_crossings_figure(results: dict) -> plt.Figure:
    """Show the same two latents' clean-crossing trajectories without repeated decoders."""
    paper._style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True, sharey=True,
                             layout="constrained")
    records = selected_features(results)
    for ax, (direction, record) in zip(axes, records.items()):
        paper._trajectories(ax, record, direction, ramp_conditioned=True)
        feature = record["summary"]["selected_directions"][direction]["latent_idx"]
        ax.set_title(f"{'Forward' if direction == 'right' else 'Backward'}: "
                     f"latent {feature}, seed {record['summary']['seed']}")
    trajectories = [record["artifacts"][f"{direction}_clean_trajectories"]
                    for direction, record in records.items()]
    axes[0].set_ylim(max(0, min(float(a.min()) for a in trajectories) - 0.04),
                     min(1, max(float(a.max()) for a in trajectories) + 0.04))
    fig.suptitle("Top 16 clean crossings per latent; both directions eligible", fontsize=10)
    paper._save(fig, "supplement_ramp_conditioned")
    return fig


@beartype
def direction_table(results: dict) -> pd.DataFrame:
    """Recompute and export both DI definitions for every feature shown in S6."""
    records = selected_features(results)
    metadata = pd.read_csv(paper.RUNS / "temporal_transformer/analysis_windows.csv")
    rows = []
    for direction, record in records.items():
        selection = record["summary"]["selected_directions"][direction]
        feature = int(selection["latent_idx"])
        with np.load(record["path"] / "scores.npz") as scores:
            np.testing.assert_array_equal(scores["analysis_starts"], metadata.start.to_numpy())
            acts = scores["analysis_activations"]
            clean = matched_direction_metrics(acts, metadata, bins=40,
                                               eligible_column="ramp_clean").iloc[feature]
            full = matched_direction_metrics(acts, metadata).iloc[feature]
        stored = next(item for item in record["summary"]["analysis_selected_metrics"]
                      if int(item["latent_idx"]) == feature)
        np.testing.assert_allclose([clean.right_selectivity, full.right_selectivity],
                                   [stored["right_selectivity"], stored["global_right_selectivity"]],
                                   rtol=1e-6, atol=1e-8)
        rows.append({
            "direction": direction, "seed": record["summary"]["seed"], "latent": feature,
            "clean_crossing_DI": float(clean.right_selectivity),
            "full_track_DI": float(full.right_selectivity),
            "unrestricted_preferred": int(round(selection["all_top_expected_direction_fraction"] * 16)),
            "unrestricted_n": int(selection["all_top_n"]),
            "clean_preferred": int(round(selection["clean_top_expected_direction_fraction"] * 16)),
            "clean_n": int(selection["clean_top_n"]),
            "active_clean_right": int(clean.right_active_windows),
            "active_clean_left": int(clean.left_active_windows),
            "clean_position_bins": int(clean.matched_position_bins),
            "clean_matching_weight": int(clean.matched_windows_per_direction),
            "full_track_position_bins": int(full.matched_position_bins),
            "full_track_matching_weight": int(full.matched_windows_per_direction),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(paper.PAPER / "appendix_temporal_direction_scores.csv", index=False)
    return frame




@beartype
def write_provenance(results: dict) -> dict:
    """Record appendix-only inputs without modifying main-figure artifacts."""
    records = [record for runs in results["temporal"].values() for record in runs]
    inputs = [Path(__file__), Path(paper.__file__),
              Path(__file__).with_name("temporal_analysis.py"),
              paper.DATA / "parameters.json",
              paper.RUNS / "vanilla/summary/final_features.npz",
              paper.RUNS / "vanilla/summary/selected_features.csv",
              paper.RUNS / "vanilla/summary/summary.json",
              paper.RUNS / "temporal_transformer/analysis_windows.csv"]
    inputs.extend(sorted(paper.DATA.glob("*.npy")))
    inputs.extend(sorted((paper.RUNS / "vanilla").glob("seed_*/summary.json")))
    inputs.extend(sorted((paper.RUNS / "matryoshka_overlap").glob("seed_*/summary.json")))
    for record in records:
        inputs.extend(record["path"] / name for name in ("summary.json", "scores.npz", "artifacts.npz"))
    manifest = {str(p.relative_to(paper.SYNTHETIC)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in inputs}
    (paper.PAPER / "appendix_artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Render the current synthetic figures through the shared notebook workflow."""
    from experiments.synthetic.reproduce import main as reproduce
    reproduce()


if __name__ == "__main__":
    main()

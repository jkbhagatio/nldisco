"""Render the multi-session appendix figures and table from saved runs.

No model is fitted here. Session A values come from the paper's saved analysis
inputs; session B values come from the adaptation and replication runs on Ceph.
"""

# ruff: noqa: F821 -- jaxtyping dimension strings are not Python names.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.metrics import roc_curve

from experiments.aeon.sweep_20260921 import paper_results as source
from experiments.aeon.sweep_20260921.shared_analysis import complete_support
from experiments.rebuttal.change02_multisession.adapt import MIN_CONDITION_WINDOWS, PAPER_WHEEL_THRESHOLD
from experiments.rebuttal.change02_multisession.common import (
    DATA, SEQ_LEN, SESSION_A, behavior_conditions, score_latents, score_one,
)
from experiments.rebuttal.change02_multisession.profiles import (
    condition_panel, condition_transfer, session_a_profile,
)

REPO = Path(__file__).resolve().parents[3]
FIGURES = REPO / "paper/iclr_paper/figures/rebuttal/change02_multisession"
A_COLOR, B_COLOR, CONTROL, REFERENCE = "#1f77b4", "#ff7f0e", "#7f7f7f", "#2ca02c"
TITLES = {"wheel": "Wheel activity", "area": "Smaller area near the patch",
          "speed": "Highest centroid-speed quintile", "direction": "Radially outward movement"}
PAPER_LATENTS = {"wheel": (192, 52), "area": (256, 107), "speed": (256, 109), "direction": (256, 10)}


@beartype
def style() -> None:
    """Match the paper's Aeon figure typography."""
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "legend.frameon": False, "legend.fontsize": 7})


@beartype
def panel_letter(ax: plt.Axes, letter: str) -> None:
    """Bold lowercase panel label, as in the Aeon appendix figures."""
    ax.text(-.14, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")


@beartype
def save(fig: plt.Figure, name: str) -> None:
    """Save vector and raster versions."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=220)
    plt.close(fig)


@beartype
def load_run(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return saved native activations and endpoint bins of a run."""
    return np.load(run / "activations.npy", mmap_mode="r"), np.load(run / "endpoint_bins.npy")


@beartype
def at_fpr(values: np.ndarray, condition: np.ndarray, valid: np.ndarray, target: float) -> dict:
    """Scores at the activation threshold whose false-positive rate is nearest ``target``."""
    marker = source.roc_operating_point(values[valid], condition[valid], target)
    return dict(coverage=marker["tpr"], specificity=1 - marker["fpr"],
                selectivity=marker["tpr"] / (marker["tpr"] + marker["fpr"]), threshold=marker["threshold"])


@beartype
def wheel_rows(runs: dict, conditions_b: dict) -> tuple[list[dict], dict]:
    """Collect every wheel-activity row of the transfer table, plus curves for plotting."""
    condition, valid = conditions_b["wheel"]
    paper = pd.read_csv(SESSION_A / "paper_revision_wheelcms_20260921/feature_scores.csv").set_index("feature")
    marker = json.loads((SESSION_A / "paper_revision_wheelcms_20260921/wheel_roc_marker.json").read_text())
    paper_fpr = marker["fpr"]
    rows = [dict(row="Session A, paper model", latent="52", coverage=marker["tpr"],
                 specificity=1 - paper_fpr, selectivity=marker["tpr"] / (marker["tpr"] + paper_fpr),
                 auroc=float(paper.loc["wheel", "auroc"]))]
    curves = {}
    for key, label in [("position", "Session B, transferred"),
                       ("zero", "Session B, position matching only"),
                       ("random", None)]:
        if key not in runs:
            continue
        codes, _ = load_run(runs[key])
        values = np.asarray(codes[:, 52], dtype=np.float64)
        thresholded = np.where(values >= PAPER_WHEEL_THRESHOLD, values, 0.)
        scores = score_one(thresholded, condition, valid)
        scores["auroc"] = score_one(values, condition, valid)["auroc"]
        fpr, tpr, _ = roc_curve(condition[valid], values[valid], drop_intermediate=True)
        curves[key] = dict(values=values, fpr=fpr, tpr=tpr, marker=(1 - scores["specificity"], scores["coverage"]))
        if label is None:
            continue
        rows.append(dict(row=label, latent="52", **scores))
        if key == "position":
            table = score_latents(np.ascontiguousarray(codes[:, :192]), condition, valid)
            others = table.drop(index=52)
            curves["all_auroc"] = table.auroc.to_numpy()
            chance = dict(row="Session B, other 191 shared latents", latent="--",
                          coverage=np.nan, specificity=np.nan, selectivity=np.nan,
                          auroc=float(others.auroc.median()), auroc_p95=float(others.auroc.quantile(.95)),
                          auroc_max=float(others.auroc.max()))
    rows.append(chance)
    codes, _ = load_run(runs["b_only_192"])
    table = score_latents(np.ascontiguousarray(codes), condition, valid)
    best = int(table.auroc.idxmax())
    values = np.asarray(codes[:, best], dtype=np.float64)
    matched = at_fpr(values, condition, valid, paper_fpr)
    rows.append(dict(row="Session B only, best-matching latent", latent=str(best),
                     coverage=matched["coverage"], specificity=matched["specificity"],
                     selectivity=matched["selectivity"], auroc=float(table.auroc[best])))
    fpr, tpr, _ = roc_curve(condition[valid], values[valid], drop_intermediate=True)
    curves["b_only"] = dict(fpr=fpr, tpr=tpr, latent=best)
    return rows, curves


@beartype
def appendix_rows(runs: dict, conditions_b: dict) -> list[dict]:
    """Rows for the three D=256 features: session A, transferred, control, session-B-only."""
    paper = pd.read_csv(SESSION_A / "paper_revision_wheelcms_20260921/feature_scores.csv").set_index("feature")
    rows = []
    loaded = {key: load_run(runs[key])[0] for key in ("position_256", "zero_256", "b_only_256")
              if key in runs}
    for name in ("area", "speed", "direction"):
        latent = PAPER_LATENTS[name][1]
        condition, valid = conditions_b[name]
        reference = paper.loc[name]
        rows.append(dict(feature=name, row="Session A, paper model", latent=str(latent),
                         coverage=float(reference.tpr), specificity=float(1 - reference.fpr),
                         selectivity=float(reference.selectivity), auroc=float(reference.auroc)))
        for key, label in [("position_256", "Session B, transferred"),
                           ("zero_256", "Session B, position matching only")]:
            if key in loaded:
                rows.append(dict(feature=name, row=label, latent=str(latent),
                                 **score_one(np.asarray(loaded[key][:, latent], dtype=np.float64), condition, valid)))
        if "position_256" in loaded:
            table = score_latents(np.ascontiguousarray(loaded["position_256"][:, :256]), condition, valid)
            others = table.drop(index=latent)
            rows.append(dict(feature=name, row="Session B, other 255 shared latents",
                             latent="--", coverage=np.nan, specificity=np.nan, selectivity=np.nan,
                             auroc=float(others.auroc.median()), auroc_p95=float(others.auroc.quantile(.95)),
                             auroc_max=float(others.auroc.max()),
                             rank=int((table.auroc > table.auroc[latent]).sum()) + 1))
        if "b_only_256" in loaded:
            table = score_latents(np.ascontiguousarray(loaded["b_only_256"]), condition, valid)
            best = int(table.auroc.idxmax())
            rows.append(dict(feature=name, row="Session B only, best-matching latent", latent=str(best),
                             **table.loc[best, ["coverage", "specificity", "selectivity", "auroc"]].to_dict()))
    return rows


@beartype
def render_table(wheel: list[dict], others: list[dict]) -> None:
    """Write the transfer-metrics table in the style of the Aeon feature table."""
    def cells(row: dict) -> str:
        def number(value, digits):
            return "--" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"
        auroc = number(row["auroc"], 3)
        if "auroc_p95" in row:
            auroc = f"{row['auroc']:.3f} ({row['auroc_p95']:.3f}, {row['auroc_max']:.3f})"
        return " & ".join([row["row"], row["latent"], number(row["coverage"], 3), number(row["specificity"], 3),
                           number(row["selectivity"], 3), auroc]) + r" \\"
    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
             r"Model and session & Latent & \metric{Cov} & \metric{Spec} & \metric{Sel} & \metric{AUROC} \\",
             r"\midrule", r"\multicolumn{6}{l}{\textit{Wheel activity} ($D=192$, $k=12$)} \\"]
    lines += [cells(row) for row in wheel]
    for name in ("area", "speed", "direction"):
        lines += [r"\midrule", r"\multicolumn{6}{l}{\textit{" + TITLES[name] + r"} ($D=256$, $k=12$)} \\"]
        lines += [cells(row) for row in others if row["feature"] == name]
    lines += [r"\bottomrule", r"\end{tabular*}", ""]
    (FIGURES / "multisession_transfer_metrics.tex").write_text("\n".join(lines))
    (FIGURES / "multisession_transfer_metrics.json").write_text(
        json.dumps(dict(wheel=wheel, appendix_features=others), indent=2, default=float) + "\n")


@beartype
def excerpt_start(wheel_active: np.ndarray, near: np.ndarray, valid: np.ndarray, latent: np.ndarray,
                  width: int = 18000) -> int:
    """Start bin of a six-minute excerpt with several wheel bouts and patch visits without wheel use.

    Among excerpts with at least four bouts and both states well represented, the
    one where latent and wheel state agree best is shown.
    """
    best, start = -1., 0
    for candidate in range(0, len(wheel_active) - width, 1500):
        chunk = slice(candidate, candidate + width)
        if valid[chunk].mean() < .999:
            continue
        on = wheel_active[chunk]
        off = near[chunk] & ~on
        bouts = int(np.diff(on.astype(np.int8)).clip(min=0).sum())
        if bouts < 4 or on.mean() < .15 or off.mean() < .15 or (~near[chunk]).mean() < .1:
            continue
        value = (latent[chunk][on].mean() + (~latent[chunk][~on]).mean()) / 2
        if value > best:
            best, start = value, candidate
    return start


@beartype
def render_wheel_figure(curves: dict, runs: dict, conditions_b: dict) -> dict:
    """Tuning, ROC, all-latent AUROCs and a session B excerpt for latent 52."""
    paper_dir = SESSION_A / "paper_revision_wheelcms_20260921"
    behavior = pd.read_parquet(DATA["B"] / "behavior.parquet",
                               columns=["wheel_speed", "wheel_valid", "patch_distance", "camera_analysis_valid"])
    _, endpoints = load_run(runs["position"])
    condition, valid = conditions_b["wheel"]
    speed = behavior.wheel_speed.to_numpy(dtype=float)[endpoints]
    fig = plt.figure(figsize=(7.2, 4.9), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1., .62])
    ax = fig.add_subplot(grid[0, 0])
    reference = np.load(paper_dir / "tuning_wheel.npz")
    centers = reference["centers_cm_s"]
    ax.plot(centers, reference["probability"], "o-", ms=3, lw=1.2, color=A_COLOR, label="Session A")
    tuning = {}
    for key, color, label, ls in [("position", B_COLOR, "Session B, transferred", "-")]:
        tuning[key] = source.wheel_tuning(speed, curves[key]["values"] > 0, valid)
        ax.plot(centers, tuning[key]["probability"], "o", ms=3, lw=1.2, ls=ls, color=color, label=label)
    ax.axvline(source.WHEEL_CUTOFF_CM_S, color=".5", ls="--", lw=.8)
    ax.set_xscale("symlog", linthresh=.01 * source.WHEEL_CM_PER_TURN)
    ax.set_xlim(-.0005 * source.WHEEL_CM_PER_TURN, float(centers[-1]) * 1.12)
    ax.set_xticks([0, .25, 1, 5, 25], ["0", "0.25", "1", "5", "25"])
    ax.set(ylim=(0, 1.04), xlabel="Wheel speed (cm/s)", ylabel="P(latent 52 active)")
    ax.legend(loc="center right", handlelength=1.4, bbox_to_anchor=(1.02, .62))
    panel_letter(ax, "a")
    ax = fig.add_subplot(grid[0, 1])
    roc = np.load(paper_dir / "roc_wheel.npz")
    marker = json.loads((paper_dir / "wheel_roc_marker.json").read_text())
    ax.plot(roc["fpr"], roc["tpr"], color=A_COLOR, lw=1.25, label="Session A")
    ax.scatter(marker["fpr"], marker["tpr"], color=A_COLOR, s=18, zorder=3)
    ax.plot(curves["position"]["fpr"], curves["position"]["tpr"], color=B_COLOR, lw=1.25,
            label="Session B, transferred")
    ax.scatter(*curves["position"]["marker"], color=B_COLOR, s=18, zorder=3)
    ax.plot(curves["b_only"]["fpr"], curves["b_only"]["tpr"], color=REFERENCE, lw=1., ls=":",
            label="Session B only")
    ax.plot([0, 1], [0, 1], ":", color=".75", lw=.7)
    ax.set(xlim=(0, 1), ylim=(0, 1.04), xlabel="False-positive rate", ylabel="True-positive rate")
    ax.legend(loc="lower right", handlelength=1.2, fontsize=6.5, bbox_to_anchor=(1.03, .02))
    panel_letter(ax, "b")
    ax = fig.add_subplot(grid[0, 2])
    auroc = curves["all_auroc"]
    order = np.argsort(-auroc)
    ax.plot(np.arange(1, len(auroc) + 1), auroc[order], ".", ms=2.5, color=".55")
    position = int(np.flatnonzero(order == 52)[0])
    ax.plot(position + 1, auroc[52], "o", ms=5, color=B_COLOR)
    ax.annotate("latent 52", (position + 1, auroc[52]), xytext=(12, -3), textcoords="offset points", fontsize=7,
                color=B_COLOR)
    ax.axhline(.5, color=".75", ls=":", lw=.7)
    ax.set(xlabel="Shared latent (ranked)", ylabel="Wheel-activity AUROC in session B", ylim=(0, 1.0),
           xlim=(-4, len(auroc) + 4))
    panel_letter(ax, "c")
    ax = fig.add_subplot(grid[1, :])
    n_bins = len(behavior)
    row_of_bin = np.full(n_bins, -1, dtype=np.int64)
    row_of_bin[endpoints] = np.arange(len(endpoints))
    wheel_full = behavior.wheel_speed.to_numpy(dtype=float) > source.WHEEL_CUTOFF
    wheel_ok = complete_support(behavior.wheel_valid.to_numpy() & np.isfinite(behavior.wheel_speed.to_numpy()))
    near_full = (behavior.patch_distance.to_numpy(dtype=float) <= 100) & behavior.camera_analysis_valid.to_numpy()
    latent_full = np.zeros(n_bins, dtype=bool)
    latent_full[endpoints] = curves["position"]["values"] >= PAPER_WHEEL_THRESHOLD
    start = excerpt_start(wheel_full, near_full, wheel_ok & (row_of_bin >= 0), latent_full)
    bins = np.arange(start, start + 18000)
    latent = latent_full[bins]
    minutes = (bins - start) * .02 / 60
    for y, mask, color, label in [(2, near_full[bins], ".35", "Within 100 px of patch"),
                                  (1, wheel_full[bins], ".35", "Wheel speed > 0.75 cm/s"),
                                  (0, latent, B_COLOR, "Latent 52 active")]:
        ax.plot(minutes[mask], np.full(int(mask.sum()), y), ls="", marker="|", markersize=9,
                markeredgewidth=.3, color=color, rasterized=True)
    ax.set(yticks=[2, 1, 0], yticklabels=["Within 100 px of patch", "Wheel speed > 0.75 cm/s", "Latent 52 active"],
           ylim=(-.6, 2.6), xlim=(0, 6), xlabel=f"Time (min) from hour {start * .02 / 3600:.1f} of session B")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(-.2, 1.08, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")
    save(fig, "multisession_wheel_transfer")
    return dict(excerpt_start_bin=int(start), excerpt_start_hour=float(start * .02 / 3600),
                tuning_b={key: dict(probability=value["probability"].tolist(), total=value["total"].tolist())
                          for key, value in tuning.items()})


@beartype
def render_appendix_latents(runs: dict, conditions_b: dict) -> None:
    """Session A and transferred session B tuning for the three D=256 features."""
    paper_dir = SESSION_A / "paper_revision_wheelcms_20260921"
    codes, endpoints = load_run(runs["position_256"])
    behavior = pd.read_parquet(DATA["B"] / "behavior.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), layout="constrained")
    width = .38
    for ax, name, field, label, letter in zip(
            axes, ("area", "speed"), ("area", "speed"),
            ("Area quintile (within 100 px of patch)", "Centroid-speed quintile"), "ab"):
        latent = PAPER_LATENTS[name][1]
        condition, valid = conditions_b[name]
        values = behavior[field].to_numpy(dtype=float)[endpoints]
        edges = np.quantile(values[valid], np.linspace(0, 1, 6))
        tuning_b = source.binned_probability(values, np.asarray(codes[:, latent]) > 0, valid, edges)
        with np.load(paper_dir / f"tuning_{name}.npz") as tuning_a:
            ax.bar(np.arange(1, 6) - width / 2, tuning_a["probability"], width, color=A_COLOR, label="Session A")
        ax.bar(np.arange(1, 6) + width / 2, tuning_b["probability"], width, color=B_COLOR,
               label="Session B, transferred")
        ax.set(xticks=np.arange(1, 6), ylim=(0, 1.03), xlabel=label, ylabel=f"P(latent {latent} active)")
        panel_letter(ax, letter)
    axes[0].legend(loc="upper right", handlelength=1.2)
    ax = axes[2]
    x = behavior.x.to_numpy()[endpoints] - 731.
    y = behavior.y.to_numpy()[endpoints] - 561.
    heading = behavior.heading.to_numpy()[endpoints]
    error = np.rad2deg((heading - np.arctan2(y, x) + np.pi) % (2 * np.pi) - np.pi)
    condition, valid = conditions_b["direction"]
    tuning_b = source.binned_probability(error, np.asarray(codes[:, 10]) > 0, valid, np.linspace(-180., 180., 25))
    with np.load(paper_dir / "tuning_direction.npz") as tuning_a:
        centers = (tuning_a["edges"][:-1] + tuning_a["edges"][1:]) / 2
        ax.plot(centers, tuning_a["probability"], "o-", ms=2.5, lw=1.1, color=A_COLOR)
    ax.plot(centers, tuning_b["probability"], "o-", ms=2.5, lw=1.1, color=B_COLOR)
    ax.axvspan(-45, 45, color=".5", alpha=.10, lw=0)
    ax.set(xticks=[-180, -90, 0, 90, 180], ylim=(0, 1.03), xlabel="Heading relative to outward (deg)",
           ylabel="P(latent 10 active)")
    panel_letter(ax, "c")
    save(fig, "multisession_appendix_latents")


@beartype
def render_pellet_figure(pellet_dir: Path, latent: int, window: tuple[float, float]) -> None:
    """Peri-delivery raster and activation probability of the session-B-only food-delivery latent."""
    saved = np.load(pellet_dir / "pellet_latents.npz")
    lags = saved["lags_s"]
    keep = (lags >= -1.) & (lags <= 3.)
    raster = saved[f"raster_{latent}"][:, keep]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), layout="constrained")
    event, lag = np.nonzero(np.nan_to_num(raster) > 0)
    axes[0].plot(lags[keep][lag], event + 1, ls="", marker="|", markersize=1.6, markeredgewidth=.5,
                 color=B_COLOR, rasterized=True)
    axes[0].axvline(0, color=".4", lw=.8, ls="--")
    axes[0].set(xlim=(-1, 3), ylim=(0, raster.shape[0] + 1), xlabel="Time from pellet-delivery trigger (s)",
                ylabel="Delivery (chronological)")
    panel_letter(axes[0], "a")
    axes[1].axvspan(*window, color=B_COLOR, alpha=.12, lw=0)
    axes[1].plot(lags[keep], saved[f"probability_{latent}"][keep], color=B_COLOR, lw=1.25)
    axes[1].axvline(0, color=".4", lw=.8, ls="--")
    axes[1].set(xlim=(-1, 3), ylim=(0, 1.04), xlabel="Time from pellet-delivery trigger (s)",
                ylabel=f"P(latent {latent} active)")
    panel_letter(axes[1], "b")
    save(fig, "multisession_food_delivery")


@beartype
def panel_sizes(data: Path) -> np.ndarray:
    """Number of windows inside each condition of the fixed behavior panel for one session."""
    endpoints = np.load(data / "valid_starts.npy") + SEQ_LEN - 1
    return np.array([int((c.mask & c.valid).sum()) for c in condition_panel("B", endpoints, data)])


@beartype
def control_rows(groups: list[dict]) -> list[dict]:
    """One row per initialisation and seed: paper-latent AUROCs, ranks and condition transfer."""
    rows = []
    sizes = {}
    for group in groups:
        data = Path(group["data"])
        if data not in sizes:
            sizes[data] = panel_sizes(data)
        row = dict(session=group["session"], init=group["init"], seed=group["seed"])
        for d in (192, 256):
            run = group.get(f"d{d}")
            if run is None or not (Path(run) / "summary.json").exists():
                continue
            summary = json.loads((Path(run) / "summary.json").read_text())
            for feature in summary["features"]:
                if "latent" in feature:
                    row[f"{feature['feature']}_auroc"] = feature["auroc"]
                    row[f"{feature['feature']}_rank"] = feature["auroc_rank_among_shared"]
                    if "coverage_at_paper_threshold" in feature:
                        row.update({f"wheel_{key}": feature[f"{key}_at_paper_threshold"]
                                    for key in ("coverage", "specificity", "selectivity")})
                    else:
                        row.update({f"{feature['feature']}_{key}": feature[key]
                                    for key in ("coverage", "specificity", "selectivity")})
            reference, reference_sizes = session_a_profile(d, "cpu")
            profile = np.load(Path(run) / "tuning_profile.npy")[:d]
            keep = (reference_sizes >= MIN_CONDITION_WINDOWS) & (sizes[data] >= MIN_CONDITION_WINDOWS)
            transfer = condition_transfer(np.ascontiguousarray(reference[:, keep]),
                                          np.ascontiguousarray(profile[:, keep]))
            row.update({f"d{d}_conditions": transfer["n_conditions"], f"d{d}_top1": transfer["top1"],
                        f"d{d}_top3": transfer["top3"], f"d{d}_mse": summary["mse"][str(d)]})
        rows.append(row)
    return rows


@beartype
def render_controls_table(rows: list[dict]) -> None:
    """Write the initialisation-controls and further-session table."""
    def cell(row: dict, name: str) -> str:
        if f"{name}_auroc" not in row:
            return "--"
        return f"{row[f'{name}_auroc']:.3f} ({row[f'{name}_rank']})"

    def share(row: dict, d: int) -> str:
        return "--" if f"d{d}_top1" not in row else f"{100 * row[f'd{d}_top1']:.0f} / {100 * row[f'd{d}_top3']:.0f}"

    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{$D=192$} & \multicolumn{4}{c}{$D=256$} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-9}",
             r"Session & Initialisation & Seed & \shortstack{Wheel\\latent 52} & \shortstack{Top-1 / top-3\\(\%)}"
             r" & \shortstack{Area\\latent 107} & \shortstack{Speed\\latent 109} & \shortstack{Direction\\latent 10}"
             r" & \shortstack{Top-1 / top-3\\(\%)} \\", r"\midrule"]
    previous = None
    for row in rows:
        key = (row["session"], row["init"])
        if previous is not None and key[0] != previous[0]:
            lines.append(r"\midrule")
        first = previous is None or key != previous
        lines.append(" & ".join([
            row["session"] if previous is None or key[0] != previous[0] else "",
            row["init"] if first else "", "--" if row["seed"] is None else str(row["seed"]),
            cell(row, "wheel"), share(row, 192), cell(row, "area"), cell(row, "speed"), cell(row, "direction"),
            share(row, 256)]) + r" \\")
        previous = key
    lines += [r"\bottomrule", r"\end{tabular*}", ""]
    (FIGURES / "multisession_controls.tex").write_text("\n".join(lines))
    (FIGURES / "multisession_controls.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="JSON mapping of run keys to directories")
    parser.add_argument("--controls", type=Path, default=None, help="JSON list of control-table rows")
    parser.add_argument("--only-controls", action="store_true")
    parser.add_argument("--pellet", type=Path, required=True)
    parser.add_argument("--pellet-latent", type=int, required=True)
    parser.add_argument("--pellet-window", type=float, nargs=2, default=[.3, 1.])
    args = parser.parse_args()
    style()
    if args.controls is not None:
        rows = control_rows(json.loads(args.controls.read_text()))
        render_controls_table(rows)
        print(pd.DataFrame(rows).to_string())
        if args.only_controls:
            return
    runs = {key: Path(value) for key, value in json.loads(args.runs.read_text()).items()}
    _, endpoints = load_run(runs["position"])
    conditions_b = behavior_conditions("B", endpoints)
    wheel, curves = wheel_rows(runs, conditions_b)
    others = appendix_rows(runs, conditions_b)
    render_table(wheel, others)
    extra = render_wheel_figure(curves, runs, conditions_b)
    if "position_256" in runs:
        render_appendix_latents(runs, conditions_b)
    render_pellet_figure(args.pellet, args.pellet_latent, tuple(args.pellet_window))
    (FIGURES / "multisession_figure_inputs.json").write_text(json.dumps(dict(
        runs={key: str(value) for key, value in runs.items()}, **extra), indent=2) + "\n")
    print(pd.DataFrame(wheel).to_string())
    print(pd.DataFrame(others).to_string())


if __name__ == "__main__":
    main()

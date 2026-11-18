"""Aggregate seed results into the appendix table and a machine-readable summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beartype import beartype

from experiments.rebuttal.change06_heldout import metrics

METRICS = ("cov", "spec", "sel", "auroc")
ORDER = (
    ("macaque", "recent_braking", "Recent deceleration", "NLDisco 192/16"),
    ("macaque", "fast_target_specific", "Fast leftward target reach", "NLDisco 192/16"),
    ("macaque", "early_leftward", "Early leftward movement", "NLDisco 192/16"),
    ("macaque", "late_target_specific", "Late, target-specific reach", "NLDisco 192/16"),
    ("aeon", "wheel", "Wheel activity", "NLDisco 192/12"),
    ("aeon", "area", "Smaller segmented area near the patch", "NLDisco 256/12"),
    ("aeon", "speed", "Highest centroid-speed quintile", "NLDisco 256/12"),
    ("aeon", "direction", "Radially outward movement", "NLDisco 256/12"),
)
FLAG_DROP = 0.05  # held-out mean lower than training mean by more than this in Sel or AUROC


@beartype
def aggregate(runs: dict, feature: str, criterion: str, operating: bool = False) -> dict:
    """Mean/SD/range over seeds of training and held-out metrics, plus selected latents."""
    suffix = "_operating" if operating else ""
    out = {"latents": [], "train": {}, "heldout": {}}
    for run in runs.values():
        record = run[feature][criterion]
        out["latents"].append(record["latent"])
    for part in ("train", "heldout"):
        for metric in METRICS:
            values = [run[feature][criterion][part + suffix][metric] for run in runs.values()]
            out[part][metric] = metrics.summarize(values)
    out["delta"] = {m: out["heldout"][m]["mean"] - out["train"][m]["mean"] for m in METRICS}
    out["flag"] = bool(any(out["delta"][m] < -FLAG_DROP for m in ("sel", "auroc")))
    out["oracle_heldout_auroc"] = metrics.summarize([run[feature]["oracle_heldout_auroc"] for run in runs.values()])
    out["heldout_positives"] = [run[feature][criterion]["heldout"]["n_positive"] for run in runs.values()][0]
    out["train_positives"] = [run[feature][criterion]["train"]["n_positive"] for run in runs.values()][0]
    return out


@beartype
def cell(summary: dict) -> str:
    """Format ``mean ± sd`` for LaTeX."""
    return f"${summary['mean']:.3f}{{\\pm}}{summary['sd']:.3f}$"


@beartype
def build(results: dict, criterion: str = "auroc") -> tuple:
    """Return the LaTeX tabular body and the summary dictionary for one selection criterion."""
    lines, summary = [], {}
    for dataset, feature, title, model in ORDER:
        runs = results[dataset]
        operating = feature == "wheel"
        agg = aggregate(runs, feature, criterion, operating)
        summary[feature] = agg
        latents = ", ".join(str(i) for i in agg["latents"])
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}} ({model}; latents {latents})}} \\\\")
        for part, label in (("train", "Training"), ("heldout", "Held-out")):
            cells = " & ".join(cell(agg[part][m]) for m in METRICS)
            lines.append(f"\\quad {label} & {cells} \\\\" if False else f"{label} & {cells} \\\\")
        lines.append("\\midrule" if feature != ORDER[-1][1] else "")
    header = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrr@{}}", "\\toprule",
              "Part & \\metric{Cov} & \\metric{Spec} & \\metric{Sel} & \\metric{AUROC} \\\\", "\\midrule"]
    return "\n".join(header + [l for l in lines if l] + ["\\bottomrule", "\\end{tabular*}", ""]), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macaque", type=Path, required=True)
    parser.add_argument("--aeon", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = dict(macaque=json.loads(args.macaque.read_text()), aeon=json.loads(args.aeon.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    everything = {}
    for criterion, name in (("auroc", "heldout_metrics"), ("sel", "heldout_metrics_best_sel")):
        table, summary = build(results, criterion)
        (args.output_dir / f"{name}.tex").write_text(table)
        everything[criterion] = summary
    (args.output_dir / "heldout_summary.json").write_text(json.dumps(everything, indent=2))
    for criterion, summary in everything.items():
        for feature, agg in summary.items():
            print(criterion, feature, agg["latents"], {m: round(agg["delta"][m], 3) for m in METRICS},
                  "FLAG" if agg["flag"] else "")


if __name__ == "__main__":
    main()

"""Tables and the example figure for the appendix subsection of change 3.

    python -m experiments.rebuttal.change03_discovery.paper --data DATA --results RESULTS

Reads the Part A and Part B outputs and writes LaTeX tables and one figure to
`paper/iclr_paper/figures/rebuttal/change03_discovery/`, plus `headline.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from beartype import beartype  # noqa: E402

from experiments.rebuttal.change03_discovery import data, rules, scoring  # noqa: E402
from experiments.rebuttal.change03_discovery.part_a import LABELS, METHODS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FIGURES = REPO / "paper/iclr_paper/figures/rebuttal/change03_discovery"
OUTPUTS = Path(__file__).resolve().parent / "outputs"
BLUE, GREY = "#2166ac", "#9a9a9a"
PRIMARY = "primary (AUROC>=0.75, Sel>=0.75)"


def _tex(text: str) -> str:
    """Escape rule text for LaTeX."""
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                     ("^", r"\^{}"), ("<=", r"$\leq$"), (">=", r"$\geq$"), ("<", r"$<$"),
                     (">", r"$>$")):
        text = text.replace(old, new)
    return text


def _tabular(columns: str, header: str, body: list) -> str:
    return "\n".join([rf"\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{columns}@{{}}}}",
                      r"\toprule", header + r" \\", r"\midrule", *body, r"\bottomrule",
                      r"\end{tabular*}", ""])


@beartype
def definitions_table(definitions: list) -> str:
    """Part A feature definitions with held-out prevalence."""
    body = []
    for number, f in enumerate(definitions, 1):
        share = 100 * f["heldout_positive"] / (f["heldout_positive"] + f["heldout_negative"])
        body.append(f"{number} & {f['label']} & {_tex(f['definition'])} & {f['population']} & "
                    f"{share:.1f} \\\\")
    return _tabular(r"rp{0.19\linewidth}p{0.36\linewidth}p{0.24\linewidth}r",
                    r"\# & Feature & Condition & Compared within & \% held-out", body)


@beartype
def scores_table(table: pd.DataFrame, methods: list) -> str:
    """Held-out AUROC and Sel of each method's selected latent, per feature."""
    body = []
    features = table[table.method == methods[0]]
    for number, (name, label) in enumerate(zip(features.feature, features.label), 1):
        cells = []
        for method in methods:
            row = table[(table.method == method) & (table.feature == name)].iloc[0]
            if not row.scorable:
                cells += ["--", "--"]
                continue
            text = [f"{row.auroc:.2f}", f"{row.sel:.2f}"]
            cells += [rf"\textbf{{{t}}}" for t in text] if row.passed else text
        body.append(f"{number} & {label} & " + " & ".join(cells) + r" \\")
    counts = [int(table[(table.method == m)].passed.sum()) for m in methods]
    body.append(r"\midrule")
    body.append(r"\multicolumn{2}{l}{Features recovered} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{c}}}" for c in counts) + r" \\")
    header = (r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{LABELS[m]}}}" for m in methods)
              + r" \\" + "\n" + r"\# & Feature & "
              + " & ".join([r"\metric{AUROC} & \metric{Sel}"] * len(methods)))
    return _tabular("rl" + "rr" * len(methods), header, body)


@beartype
def part_a_summary_table(summary: pd.DataFrame, widths: dict) -> str:
    """Recovery counts under every fixed criterion and mean held-out scores."""
    body = []
    best = summary[PRIMARY].max()
    for row in summary.itertuples():
        primary = str(int(summary.loc[row.Index, PRIMARY]))
        if int(primary) == best:
            primary = rf"\textbf{{{primary}}}"
        body.append(" & ".join([
            LABELS[row.method], str(widths[row.method]), f"{primary}/{row.n_features}",
            str(int(summary.loc[row.Index, "both>=0.70"])),
            str(int(summary.loc[row.Index, "both>=0.80"])),
            str(int(summary.loc[row.Index, "AUROC>=0.75 only"])),
            f"{row.mean_cov:.2f}", f"{row.mean_spec:.2f}", f"{row.mean_sel:.2f}",
            f"{row.mean_auroc:.2f}"]) + r" \\")
    header = (r"Method & Latents & Recovered & \shortstack{Both\\$\geq0.70$} & "
              r"\shortstack{Both\\$\geq0.80$} & \shortstack{\metric{AUROC}\\$\geq0.75$} & "
              r"\metric{Cov} & \metric{Spec} & \metric{Sel} & \metric{AUROC}")
    return _tabular("lrrrrrrrrr", header, body)


@beartype
def part_b_summary_table(summary: dict) -> str:
    """Blind-discovery yield, agreement and effort per method."""
    body = []
    best = max(m["validated_proportion"] for m in summary["methods"])
    for m in summary["methods"]:
        share = f"{100 * m['validated_proportion']:.0f}\\%"
        if m["validated_proportion"] == best:
            share = rf"\textbf{{{share}}}"
        effort = (f"{m['latents_per_validated']:.1f}", f"{m['minutes_per_validated']:.1f}") \
            if np.isfinite(m["latents_per_validated"]) else ("--", "--")
        kappa = "--" if not np.isfinite(m["kappa_validated"]) else f"{m['kappa_validated']:.2f}"
        body.append(" & ".join([
            m["label"], str(m["inspected"]), f"{m['interpreted_mean']:.1f}",
            f"{m['validated_mean']:.1f} ({m['validated_min']}--{m['validated_max']})", share,
            str(m["validated_majority"]), kappa, str(m["distinct_features"]), *effort]) + r" \\")
    header = (r"Method & \shortstack{Latents\\inspected} & \shortstack{Given a\\rule} & "
              r"\shortstack{Validated\\(range)} & \shortstack{Validated\\share} & "
              r"\shortstack{Validated,\\$\geq2$ raters} & \shortstack{Fleiss'\\$\kappa$} & "
              r"\shortstack{Distinct\\features} & \shortstack{Latents per\\validated} & "
              r"\shortstack{Rater min per\\validated}")
    return _tabular("lrrrrrrrrr", header, body)


@beartype
def part_b_features_table(summary: dict, methods: list) -> str:
    """Distinct validated features and the methods that surfaced each."""
    body = []
    ordered = sorted(summary["features"], key=lambda f: (-len(f["methods"]), -f["best_auroc"]))
    for number, feature in enumerate(ordered, 1):
        marks = [r"\checkmark" if m in feature["methods"] else "" for m in methods]
        rule = rf"\texttt{{{_tex(feature['rule'])}}} ({feature['direction']})"
        body.append(f"{number} & {rule} & " + " & ".join(marks) + r" \\")
    header = r"\# & Representative rule (direction) & " + " & ".join(LABELS[m] for m in methods)
    return _tabular(r"rp{0.5\linewidth}" + "c" * len(methods), header, body)


@beartype
def examples_figure(session: data.Session, answers: pd.DataFrame, key: dict, methods: list,
                    path: Path) -> list:
    """Held-out value distributions inside and outside the rule for one validated
    description per method (the one with the highest held-out AUROC)."""
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    chosen = [answers[(answers.method == m) & answers.passed].sort_values(
        "auroc", ascending=False).head(1) for m in methods]
    chosen = [c.iloc[0] for c in chosen if len(c)]
    if not chosen:
        return []
    fig, axes = plt.subplots(1, len(chosen), figsize=(7.1, 1.9), constrained_layout=True,
                             squeeze=False)
    records = []
    for letter, ax, row in zip("abcdefgh", axes[0], chosen):
        values = np.load(key[row.card]["path"], mmap_mode="r")
        mask, valid = rules.evaluate(row.rule, session.variables, session.support)
        held = session.heldout & valid
        sign = 1 if row.direction == "high" else -1
        column = sign * np.asarray(values[np.flatnonzero(held), int(row.latent)], dtype=float)
        inside = mask[held]
        lo, hi = np.quantile(column, [0.001, 0.999])
        bins = np.linspace(lo, hi, 60) if hi > lo else 10
        for flag, color, label in ((~inside, GREY, "rule does not hold"), (inside, BLUE, "rule holds")):
            weights = np.full(int(flag.sum()), 1 / max(int(flag.sum()), 1))
            ax.hist(np.clip(column[flag], lo, hi), bins=bins, weights=weights, color=color,
                    alpha=.65, label=label)
        ax.axvline(np.clip(row.threshold, lo, hi), color="k", ls="--", lw=.8)
        ax.set_yscale("log")
        ax.set_xlabel("Latent value" + (" (sign-reversed)" if sign < 0 else ""))
        ax.set_title(f"{LABELS[row.method]}\n{row.rule}", fontsize=6.5)
        ax.set_title(letter, loc="left", fontsize=10, fontweight="bold")
        ax.text(.98, .97, f"Cov {row['cov']:.2f}\nSpec {row.spec:.2f}\nSel {row.sel:.2f}\n"
                f"AUROC {row.auroc:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
        records.append(dict(method=row.method, latent=int(row.latent), rule=row.rule,
                            direction=row.direction, description=row.description,
                            rater=int(row.rater), cov=float(row["cov"]), spec=float(row.spec),
                            sel=float(row.sel), auroc=float(row.auroc)))
    axes[0][0].set_ylabel("Fraction of held-out windows")
    axes[0][0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return records


@beartype
def run(data_root: Path, results: Path, figures: Path = FIGURES,
        outputs: Path = OUTPUTS) -> None:
    """Write every table, the figure and the headline numbers."""
    FIGURES, OUTPUTS = figures, outputs  # noqa: N806 (shadow the defaults for this call)
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(results / "part_a/part_a_scores.csv")
    summary_a = pd.read_csv(results / "part_a/part_a_summary.csv")
    definitions = json.loads((results / "part_a/feature_definitions.json").read_text())["features"]
    selected = json.loads((results / "part_a/selected_runs.json").read_text())
    summary_b = json.loads((results / "part_b/part_b_summary.json").read_text())
    answers = pd.read_csv(results / "part_b/part_b_answers.csv")
    key = json.loads((results / "part_b/key/part_b_key.json").read_text())
    methods = [m for m in METHODS if m in set(scores.method)]
    widths = {m: int(np.load(selected[m]["path"], mmap_mode="r").shape[1]) for m in methods}
    (FIGURES / "part_a_definitions.tex").write_text(definitions_table(definitions))
    (FIGURES / "part_a_scores.tex").write_text(scores_table(scores, methods))
    (FIGURES / "part_a_summary.tex").write_text(part_a_summary_table(summary_a, widths))
    (FIGURES / "part_b_summary.tex").write_text(part_b_summary_table(summary_b))
    (FIGURES / "part_b_features.tex").write_text(part_b_features_table(summary_b, methods))
    session = data.load_session(data_root)
    examples = examples_figure(session, answers, key, methods, FIGURES / "examples.pdf")
    headline = dict(
        part_a={row.method: dict(recovered=int(summary_a.loc[row.Index, PRIMARY]),
                                 n_features=int(row.n_features), mean_auroc=row.mean_auroc,
                                 mean_sel=row.mean_sel,
                                 sensitivity={k: int(summary_a.loc[row.Index, k])
                                              for k in scoring.SENSITIVITY})
                for row in summary_a.itertuples()},
        part_b={m["method"]: {k: m[k] for k in (
            "inspected", "interpreted_mean", "validated_mean", "validated_min", "validated_max",
            "validated_proportion", "validated_majority", "kappa_validated", "kappa_interpreted",
            "distinct_features", "latents_per_validated", "minutes_per_validated",
            "seconds_per_card")} for m in summary_b["methods"]},
        agreement=summary_b["overall"], widths=widths,
        selected={m: dict(config=selected[m]["config"],
                          candidates={c["config"]: c["validation_loss"]
                                      for c in selected[m]["candidates"]}) for m in methods},
        distinct_features=summary_b["features"], examples=examples)
    (OUTPUTS / "headline.json").write_text(json.dumps(headline, indent=2, default=float))
    for name in ("part_a_scores.csv", "part_a_summary.csv", "feature_definitions.json",
                 "selected_runs.json"):
        (OUTPUTS / name).write_text((results / "part_a" / name).read_text())
    for name in ("part_b_answers.csv", "part_b_summary.json"):
        (OUTPUTS / name).write_text((results / "part_b" / name).read_text())
    print(json.dumps({k: headline[k] for k in ("part_a", "part_b", "agreement")}, indent=1,
                     default=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    run(args.data, args.results, args.figures, args.outputs)


if __name__ == "__main__":
    main()

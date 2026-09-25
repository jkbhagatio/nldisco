"""Render balanced-accuracy tables while preserving other main-table metrics."""

import argparse
import json
from pathlib import Path

from beartype import beartype

from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision
from experiments.churchland.sweeps.s10_endpoint_v1.matched192_tables import method_label
from experiments.churchland.sweeps.s10_endpoint_v1.publication_style import TITLES
from experiments.evaluate_decoder_balanced_accuracy import DEFAULT_OUTPUT

FIGURES = Path(__file__).resolve().parents[1] / "paper/iclr_paper/figures"
MAIN_ORDER = (("nldisco", "illustrative"), ("cebra", "sel"), ("cebra", "auroc"),
              ("langevinflow", "sel"), ("langevinflow", "auroc"))


@beartype
def render(output: Path = DEFAULT_OUTPUT / "churchland") -> None:
    """Use saved scores only; bold exact maxima, including ties."""
    results = json.loads((output / "results.json").read_text())["results"]
    selections = json.loads((revision.OUTPUT / "selection.json").read_text())["selections"]

    def probe(method, feature, column):
        return next(r for r in results if (r["method"], r["feature"], r["latent_id"])
                    == (method, feature, column))

    def selected(method, feature, criterion):
        row = next(r for r in selections if (r["method"], r["feature"], r["criterion"])
                   == (method, feature, criterion))
        return probe(method, feature, row["latent_id"])

    def cell(row, maximum, digits=4, interval=False):
        score = row["test"]["balanced_accuracy"]
        text = f"{score:.{digits}f}"
        if abs(score - maximum) < 1e-12:
            text = r"\textbf{" + text + "}"
        if interval:
            low, high = row["interval"]["ci_lower"], row["interval"]["ci_upper"]
            text += f" [{low:.3f}, {high:.3f}]"
        return text

    path = FIGURES / "churchland_comparison_main.tex"
    original = output / "main_table_before.tex"
    if not original.exists():
        original.write_text(path.read_text())
    lines = original.read_text().splitlines()
    data_indices = [i for i, line in enumerate(lines) if line.startswith(
        ("NLDisco &", "CEBRA-Time (", "LangevinFlow ("))]
    assert len(data_indices) == len(MAIN_ORDER)
    for feature, offset in (("recent_braking", 0), ("fast_target_specific", 6)):
        singles = [selected(method, feature, criterion) for method, criterion in MAIN_ORDER]
        fulls = [probe(method, feature, None) for method, _ in MAIN_ORDER]
        for column, rows in ((5 + offset, singles), (6 + offset, fulls)):
            maximum = max(row["test"]["balanced_accuracy"] for row in rows)
            for index, row in zip(data_indices, rows):
                cells = lines[index].removesuffix(r" \\").split(" & ")
                assert len(cells) == 13
                cells[column] = cell(row, maximum)
                lines[index] = " & ".join(cells) + r" \\"
    new_main = "\n".join(lines) + "\n"
    # Every non-decoding value and its formatting must remain unchanged.
    for before, after in zip(original.read_text().splitlines(), lines):
        if before.startswith(("NLDisco &", "CEBRA-Time (", "LangevinFlow (")):
            a, b = before.split(" & "), after.split(" & ")
            for col in (0, 1, 2, 3, 4, 7, 8, 9, 10):
                assert a[col] == b[col]
    (output / path.name).write_text(new_main)
    path.write_text(new_main)

    path = FIGURES / "churchland_decoding_all.tex"
    old = output / "appendix_table_before.tex"
    if not old.exists():
        old.write_text(path.read_text())
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Feature & Method / selector & Single Decode [95\% CI] & All-latent Decode [95\% CI] \\",
             r"\midrule"]
    for feature in decoder.FEATURES:
        entries = [(method_label(r["method"], r["criterion"]),
                    probe(r["method"], feature, r["latent_id"]),
                    probe(r["method"], feature, None))
                   for r in selections if r["feature"] == feature]
        for method in ("pca", "sparsenmf"):
            single = next(r for r in results if r["method"] == method
                          and r["feature"] == feature and r["latent_id"] is not None)
            entries.append((decoder.NAMES[method], single, probe(method, feature, None)))
        maxima = [max(row[i]["test"]["balanced_accuracy"] for row in entries) for i in (1, 2)]
        for i, (label, single, full) in enumerate(entries):
            label = label.replace("best Sel", r"best \metric{Sel}").replace("best AUROC", r"best \metric{AUROC}")
            title = "Recent deceleration" if feature == "recent_braking" else TITLES[feature]
            cells = [r"\textit{" + title + "}" if i == 0 else "", label]
            cells += [cell(row, maximum, digits=3, interval=True)
                      for row, maximum in zip((single, full), maxima)]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule" if feature != decoder.FEATURES[-1] else r"\bottomrule")
    content = "\n".join(lines + [r"\end{tabular}", ""])
    (output / path.name).write_text(content)
    path.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "churchland")
    render(parser.parse_args().input)

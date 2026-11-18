"""Part A: recovery of the prespecified features by each method, scored held-out.

    python -m experiments.rebuttal.change03_discovery.part_a --data DATA --runs RUNS --output OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from beartype import beartype

from experiments.rebuttal.change03_discovery import common, data, features, scoring

METHODS = ("nldisco", "cebra", "langevinflow", "stndt", "pca")  # fixed order (seeds use it)
LABELS = {"nldisco": "NLDisco", "cebra": "CEBRA-Time", "langevinflow": "LangevinFlow",
          "stndt": "ST-NDT", "pca": "PCA"}


@beartype
def select_runs(runs: Path) -> dict:
    """Pick one completed run per method by the method's own validation loss.

    Methods with a single completed run (NLDisco, PCA) need no selection.
    """
    selected = {}
    for method in METHODS:
        candidates = []
        for directory in sorted((runs / method).glob("*")) if (runs / method).is_dir() else []:
            if not (directory / "COMPLETE.json").is_file():
                continue
            summary = json.loads((directory / "summary.json").read_text())
            candidates.append(dict(config=directory.name, path=str(directory / "representation.npy"),
                                   validation_loss=summary.get("validation_loss")))
        if not candidates:
            continue
        if len(candidates) > 1 and any(c["validation_loss"] is None for c in candidates):
            raise ValueError(f"{method}: several runs but no validation loss to choose by")
        best = min(candidates, key=lambda c: (c["validation_loss"] or 0., c["config"]))
        selected[method] = dict(best, candidates=candidates)
    return selected


@beartype
def recover(representation: np.ndarray, feature_list: list, discovery: np.ndarray,
            heldout: np.ndarray, device: str) -> list[dict]:
    """Best discovery-AUROC latent per feature, then the fixed scoring rule."""
    index = np.flatnonzero(discovery)
    values = torch.as_tensor(np.ascontiguousarray(representation[index]), device=device)
    positive = torch.as_tensor(np.stack([f.mask[index] for f in feature_list], 1), device=device)
    negative = torch.as_tensor(np.stack([(f.valid & ~f.mask)[index] for f in feature_list], 1),
                               device=device)
    auc = scoring.auroc_matrix(values, positive, negative).cpu().numpy()
    del values, positive, negative
    rows = []
    for column, feature in enumerate(feature_list):
        row = dict(feature=feature.name, family=feature.family, label=feature.label)
        if not np.isfinite(auc[:, column]).all():
            rows.append(dict(row, scorable=False, passed=False))
            continue
        oriented = np.maximum(auc[:, column], 1 - auc[:, column])
        latent = int(np.argmax(oriented))
        orientation = 1 if auc[latent, column] >= 0.5 else -1
        result = scoring.evaluate(np.asarray(representation[:, latent]), orientation,
                                  feature.mask, feature.valid, discovery, heldout)
        rows.append(dict(row, latent=latent, selection_auroc=float(oriented[latent]), **result))
    return rows


@beartype
def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Per-method recovery counts under every fixed criterion and mean scores."""
    rows = []
    for method, group in table.groupby("method", sort=False):
        scored = group[group.scorable]
        row = dict(method=method, n_features=len(scored))
        for name, (auroc_min, sel_min) in scoring.SENSITIVITY.items():
            row[name] = int(((scored.auroc >= auroc_min) & (scored.sel >= sel_min)).sum())
        for metric in ("cov", "spec", "sel", "auroc"):
            row[f"mean_{metric}"] = float(scored[metric].mean())
            row[f"median_{metric}"] = float(scored[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


@beartype
def run(data_root: Path, runs: Path, output: Path, device: str) -> None:
    """Score every selected method on every prespecified feature."""
    output.mkdir(parents=True, exist_ok=True)
    session = data.load_session(data_root)
    feature_list = features.build_features(session.variables, session.support, session.discovery)
    selected = select_runs(runs)
    common.write_json(output / "selected_runs.json", selected)
    definitions = [dict(name=f.name, family=f.family, label=f.label, definition=f.definition,
                        population=f.population,
                        discovery_positive=int((f.mask & session.discovery).sum()),
                        discovery_negative=int((f.valid & ~f.mask & session.discovery).sum()),
                        heldout_positive=int((f.mask & session.heldout).sum()),
                        heldout_negative=int((f.valid & ~f.mask & session.heldout).sum()))
                   for f in feature_list]
    common.write_json(output / "feature_definitions.json", {"features": definitions})
    tables = []
    for method, info in selected.items():
        representation = np.load(info["path"])
        rows = recover(representation, feature_list, session.discovery, session.heldout, device)
        tables.append(pd.DataFrame(rows).assign(method=method, config=info["config"]))
        print(method, info["config"], "recovered", sum(r["passed"] for r in rows), flush=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(output / "part_a_scores.csv", index=False)
    summary = summarize(table)
    summary.to_csv(output / "part_a_summary.csv", index=False)
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.data, args.runs, args.output, args.device)


if __name__ == "__main__":
    main()

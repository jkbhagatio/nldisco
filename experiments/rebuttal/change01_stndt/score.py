"""Select the ST-NDT model and score it with the unchanged Section 4.2 pipeline.

Stages mirror the existing baselines: (1) rank the twelve sweep runs by the mean
best-eligible-latent AUROC over the four behavioral descriptions, with the original
development labels, sign choice and rate-matched thresholds of `comparison_features`;
(2) keep the best-Sel and best-AUROC eligible coordinates of the selected run;
(3) rescore those fixed coordinates, signs and thresholds on the current labels
(`nice_cutoffs_revision`); (4) fit the shared logistic probes on training trials, choose
regularisation and the decision threshold on validation trials and score test trials.

    python -m experiments.rebuttal.change01_stndt.score
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from beartype import beartype
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from experiments.churchland.sweeps.s10_endpoint_v1 import comparison_features as shared
from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision
from experiments.churchland.sweeps.s10_endpoint_v1.langevinflow_feature_selection import (
    select_configuration,
)
from experiments.decoder_metrics import (
    binary_metrics,
    select_balanced_threshold,
    trial_balanced_interval,
)

HERE = Path(__file__).resolve().parent
CEPH = Path("/ceph/aeon/aeon/nldisco/rebuttal_20261118/change01_stndt")
OUTPUT = HERE / "outputs"
FEATURES = shared.EXPECTED_FEATURES
METHOD = "stndt"


@beartype
def load_run(folder: Path) -> np.ndarray:
    """Load one complete export after checking its row order against the shared index."""
    summary = json.loads((folder / "summary.json").read_text())
    if not summary["complete"] or summary["reload_max_abs"] > 1e-5:
        raise ValueError(f"Incomplete run: {folder.name}")
    with np.load(folder / "index.npz") as saved, np.load(
            shared.SWEEP / "shared/index.npz") as index:
        np.testing.assert_array_equal(saved["anchor_index"], index["anchor_index"])
        np.testing.assert_array_equal(saved["source_indices"], index["source_indices"])
        if not saved["train_mask"].all():
            raise ValueError("Scored representations must come from all-window fits")
    values = np.load(folder / "activations.npy", mmap_mode="r")
    if values.shape != (130507, 191) or values.dtype != np.float32:
        raise ValueError(f"Unexpected export: {folder.name} {values.shape}")
    return values


@beartype
def best_selective(rows: list):
    """Highest Sel among eligible coordinates; AUROC, then lowest index, break ties."""
    return max((r for r in rows if r["eligible"]),
               key=lambda r: (r["sel"], r["auroc"], -r["latent_id"]), default=None)


@beartype
def select(sweep: Path) -> dict:
    """Rank all sweep runs and freeze the selected run's coordinates per feature."""
    features = shared.load_development_features()
    folders = sorted(p for p in sweep.iterdir() if (p / "summary.json").exists())
    failed = sorted(p.name for p in sweep.iterdir() if (p / "failure.json").exists())
    if len(folders) + len(failed) != 12:
        raise ValueError(f"Expected twelve finished runs, found {len(folders)}+{len(failed)}")
    winners, candidates = {}, {}
    for folder in folders:
        values = load_run(folder)
        candidates[folder.name] = {name: shared.score_dense(values, features[name])
                                   for name in FEATURES}
        winners[folder.name] = {name: shared.best_eligible(rows)
                                for name, rows in candidates[folder.name].items()}
        print(folder.name, {name: round(row["auroc"], 4) if row else None
                            for name, row in winners[folder.name].items()}, flush=True)
    choice = select_configuration(winners)
    if choice is None:
        raise ValueError("No run has an eligible coordinate for all four descriptions")
    chosen = candidates[choice["config_id"]]
    ranking = sorted(
        (dict(config_id=config_id,
              mean_auroc=float(np.mean([w[name]["auroc"] for name in FEATURES]))
              if all(w.values()) else None,
              best_auroc={name: (w[name] or {}).get("auroc") for name in FEATURES})
         for config_id, w in winners.items()),
        key=lambda row: (-(row["mean_auroc"] or 0), row["config_id"]))
    return dict(
        method=METHOD, selected=choice["config_id"], mean_auroc=choice["mean_auroc"],
        ranking=ranking, failed_runs=failed,
        selection_rule="Run with the highest equal-weight mean of the four best eligible "
                       "oriented AUROCs (original development labels), as for the "
                       "existing baselines",
        reference_rates={name: features[name].reference_rate for name in FEATURES},
        choices={name: dict(sel=best_selective(chosen[name]),
                            auroc=shared.best_eligible(chosen[name])) for name in FEATURES},
        candidates=chosen)


@beartype
def rescore(values: np.ndarray, selection: dict) -> tuple[list, dict]:
    """Rescore the frozen coordinates on the current labels and trace their ROC curves."""
    features = revision.features()
    rows, curves = [], {name: [] for name in FEATURES}
    for name in FEATURES:
        feature = features[name]
        for criterion in ("sel", "auroc"):
            row = revision.rescore(values, feature, selection["choices"][name][criterion])
            rows.append(dict(method=METHOD, feature=name, criterion=criterion,
                             config_id=selection["selected"], **row))
            if any(c["latent_id"] == row["latent_id"] for c in curves[name]):
                continue
            z = np.asarray(values[feature.rows, row["latent_id"]]) * row["orientation"]
            fpr, tpr, _ = roc_curve(feature.labels, z)
            np.testing.assert_allclose(roc_auc_score(feature.labels, z), row["auroc"])
            curves[name].append(dict(method=METHOD, latent_id=row["latent_id"],
                                     fpr=fpr.tolist(), tpr=tpr.tolist(), auroc=row["auroc"]))
    return rows, curves


@beartype
def decode(values: np.ndarray, rows: list, output: Path) -> list:
    """Fit the shared probes on development trials, then score test trials once."""
    development = revision.labels(shared.SWEEP, decoder.DATA, (0, 1))
    test = revision.labels(shared.SWEEP, decoder.DATA, (2,))
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in FEATURES:
        columns = sorted({row["latent_id"] for row in rows if row["feature"] == name})
        for column in (*columns, None):
            key = f"{METHOD}_{name}_{'full' if column is None else column}"
            x = values if column is None else np.asarray(values[:, [column]])
            dev, held = development[name], test[name]
            assert not np.intersect1d(dev["trials"], held["trials"]).size
            (scaler, model), tuned = decoder.tune_probe(x, dev)
            validation = dev["split"] == 1
            probabilities = model.predict_proba(scaler.transform(
                np.asarray(x[dev["rows"][validation]], dtype=np.float64)))[:, 1]
            np.testing.assert_allclose(
                roc_auc_score(dev["labels"][validation], probabilities),
                tuned["validation_auroc"], atol=1e-12)
            threshold = select_balanced_threshold(dev["labels"][validation], probabilities)
            probabilities = model.predict_proba(scaler.transform(
                np.asarray(x[held["rows"]], dtype=np.float64)))[:, 1]
            predicted = probabilities >= threshold
            joblib.dump((scaler, model), output / f"{key}_probe.joblib")
            np.savez_compressed(output / f"{key}_predictions.npz", labels=held["labels"],
                                probabilities=probabilities, predicted=predicted,
                                endpoint_row=held["rows"], trial_id=held["trials"])
            results.append(dict(
                key=key, method=METHOD, feature=name, latent_id=column,
                space="full" if column is None else "single", threshold=threshold, **tuned,
                test_auroc=float(roc_auc_score(held["labels"], probabilities)),
                test=binary_metrics(held["labels"], predicted),
                interval=trial_balanced_interval(held["labels"], predicted, held["trials"])))
            print(key, f"BA={results[-1]['test']['balanced_accuracy']:.4f}", flush=True)
    return results


@beartype
def convergence(folder: Path) -> list:
    """Collect held-out masked prediction of the training-trial fits."""
    rows = []
    for run in sorted(folder.iterdir()):
        if not (run / "summary.json").exists():
            continue
        history = json.loads((run / "progress.json").read_text())["history"]
        rows.append(dict(config_id=run.name,
                         train_masked_nll=[e["masked_nll"] for e in history],
                         validation=[e["validation"] for e in history]))
    return rows


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceph", type=Path, default=CEPH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = select(args.ceph / "sweep")
    decoder.write_json(args.output / "selection.json", selection, exclusive=True)
    values = load_run(args.ceph / "sweep" / selection["selected"])
    rows, curves = rescore(values, selection)
    decoder.write_json(args.output / "roc_plot_data.json", curves, exclusive=True)
    # One BLAS thread reproduces the frozen baseline probes exactly (same regularisation
    # choice, threshold and test score); more threads can flip near-tied validation AUROCs.
    with threadpool_limits(limits=1):
        probes = decode(values, rows, args.ceph / "decoding")
    decoder.write_json(args.output / "results.json", dict(
        method=METHOD, config_id=selection["selected"], dimension=int(values.shape[1]),
        representation=str(args.ceph / "sweep" / selection["selected"] / "activations.npy"),
        selections=rows, decoding=probes,
        convergence=convergence(args.ceph / "convergence")), exclusive=True)


if __name__ == "__main__":
    main()

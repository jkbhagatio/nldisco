"""Score saved paper decoders at validation-selected probability thresholds.

Run with ``uv run python -m experiments.evaluate_decoder_balanced_accuracy``.
Original fits, AUROC-selected regularization, inputs, and splits are retained.
Output is versioned separately; all thresholds are saved before test scoring.
"""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from beartype import beartype
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.churchland.sweeps.s10_endpoint_v1 import feature_decoding as decoder
from experiments.churchland.sweeps.s10_endpoint_v1 import nice_cutoffs_revision as revision
from experiments.decoder_metrics import (
    binary_metrics,
    select_balanced_threshold,
    trial_balanced_interval,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/outputs/decoder_balanced_accuracy_20260923"
AEON_ROOT = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
AEON_FIT = ROOT / "experiments/aeon/outputs/wheel_latent52_logistic_20260923"


@beartype
def prepare_output(output: Path) -> None:
    """Keep prior fits and scoring runs intact."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("decoder_metrics.py"), output / "decoder_metrics.py")
    decoder.write_json(output / "protocol.json", dict(
        metric="Test balanced accuracy = (sensitivity + specificity) / 2",
        threshold="Maximize validation balanced accuracy; highest threshold breaks exact ties; p >= t",
        regularization="Reuse existing L2 logistic fits selected by validation AUROC; no refitting",
        test_policy="Freeze all thresholds before test scoring; no test-based threshold tuning",
        scope="Reanalysis after prior test exposure; representation and feature discovery are transductive",
        bootstrap="Churchland: 300 whole-test-trial resamples, seed 734, fixed threshold",
    ))


@beartype
def churchland(output: Path) -> None:
    """Rescore all main and supplemental Churchland logistic probes."""
    prepare_output(output)
    source = revision.OUTPUT
    spec = json.loads((source / "frozen_spec.json").read_text())
    records = sum([json.loads((source / f"{stem}.json").read_text())["results"]
                   for stem in ("results", "supplemental_results")], [])
    validation = revision.labels(decoder.SWEEP, decoder.DATA, (1,))
    arrays = {method: decoder.load_representation(record, decoder.SWEEP)
              for method, record in spec["representations"].items()}
    arrays.update({method: np.load(revision.LINEAR / method / "activations.npy", mmap_mode="r")
                   for method in ("pca", "sparsenmf")})
    tuned, fitted = [], {}
    for record in records:
        method, feature, space = (record[key] for key in ("method", "feature", "space"))
        column = record.get("latent_id", record.get("coordinate")) if space == "single" else None
        key = f"{method}_{feature}_{column if column is not None else 'full'}"
        changed = feature in revision.CHANGED
        if method in ("pca", "sparsenmf") and not changed:
            path = revision.LINEAR / "evaluation" / f"{method}_{feature}_{space}_probe.joblib"
            prediction_path = revision.LINEAR / "evaluation" / f"{method}_{feature}_{space}_predictions.npz"
        else:
            folder = source if changed else revision.PREVIOUS
            path = folder / f"{key}_probe.joblib"
            prediction_path = folder / f"{key}_predictions.npz"
        scaler, model = joblib.load(path)
        values = arrays[method]
        values = values if column is None else values[:, [column]]
        val = validation[feature]
        probabilities = model.predict_proba(
            scaler.transform(np.asarray(values[val["rows"]], dtype=np.float64)))[:, 1]
        np.testing.assert_allclose(roc_auc_score(val["labels"], probabilities),
                                   record["validation_auroc"], atol=1e-12, rtol=0)
        threshold = select_balanced_threshold(val["labels"], probabilities)
        row = dict(key=key, method=method, feature=feature, space=space, latent_id=column,
                   chosen_c=record["chosen_c"], threshold=threshold,
                   validation=binary_metrics(val["labels"], probabilities >= threshold),
                   validation_auroc=record["validation_auroc"], model_path=str(path),
                   model_sha256=decoder.sha256(path), source_test_auroc=record["test_auroc"])
        tuned.append(row)
        fitted[key] = (values, scaler, model, prediction_path)
    decoder.write_json(output / "validation_selection.json", dict(results=tuned), exclusive=True)
    test = revision.labels(decoder.SWEEP, decoder.DATA, (2,))
    results = []
    for row in tuned:
        values, scaler, model, prediction_path = fitted[row["key"]]
        f = test[row["feature"]]
        assert not np.intersect1d(f["trials"], validation[row["feature"]]["trials"]).size
        probabilities = model.predict_proba(scaler.transform(values[f["rows"]]))[:, 1]
        np.testing.assert_allclose(roc_auc_score(f["labels"], probabilities),
                                   row["source_test_auroc"], atol=1e-12, rtol=0)
        if prediction_path.exists():
            with np.load(prediction_path) as saved:
                np.testing.assert_allclose(probabilities, saved["predictions"], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(f["labels"], saved["labels"])
        predicted = probabilities >= row["threshold"]
        scores = binary_metrics(f["labels"], predicted)
        np.testing.assert_allclose(scores["balanced_accuracy"],
                                   balanced_accuracy_score(f["labels"], predicted), atol=1e-15)
        result = dict(row, test=scores,
                      interval=trial_balanced_interval(f["labels"], predicted, f["trials"]))
        results.append(result)
        np.savez_compressed(output / f"{row['key']}_predictions.npz", labels=f["labels"],
                            probabilities=probabilities, predicted=predicted,
                            endpoint_row=f["rows"], trial_id=f["trials"])
        print(row["key"], f"BA={scores['balanced_accuracy']:.6f}", flush=True)
    decoder.write_json(output / "results.json", dict(results=results), exclusive=True)


@beartype
def aeon(output: Path) -> None:
    """Rescore the existing Aeon wheel-activity single-latent decoder."""
    prepare_output(output)
    fit = joblib.load(AEON_FIT / "decoder.joblib")
    original = json.loads((AEON_FIT / "results.json").read_text())
    with np.load(AEON_ROOT / "paper_revision_wheelcms_20260921/feature_arrays.npz") as arrays:
        endpoints = arrays["endpoint_bins"]
        values = arrays["values_wheel"].astype(np.float64)
        labels = arrays["condition_wheel"]
        valid = arrays["valid_wheel"]
    protocol = json.loads((AEON_FIT / "protocol.json").read_text())
    _, left, right, end = protocol["boundary_bins"]
    validation = valid & (endpoints - 19 >= left) & (endpoints < right)
    test = valid & (endpoints - 19 >= right) & (endpoints < end)
    assert endpoints[validation].max() < (endpoints[test] - 19).min()
    probabilities = fit["model"].predict_proba(fit["scaler"].transform(values[validation, None]))[:, 1]
    np.testing.assert_allclose(roc_auc_score(labels[validation], probabilities),
                               original["validation_auroc"], atol=1e-12, rtol=0)
    threshold = select_balanced_threshold(labels[validation], probabilities)
    selection = dict(threshold=threshold, chosen_c=original["chosen_c"],
                     validation=binary_metrics(labels[validation], probabilities >= threshold),
                     model_path=str(AEON_FIT / "decoder.joblib"),
                     model_sha256=decoder.sha256(AEON_FIT / "decoder.joblib"))
    decoder.write_json(output / "validation_selection.json", selection, exclusive=True)
    probabilities = fit["model"].predict_proba(fit["scaler"].transform(values[test, None]))[:, 1]
    with np.load(AEON_FIT / "test_predictions.npz") as saved:
        np.testing.assert_array_equal(endpoints[test], saved["endpoint_bins"])
        np.testing.assert_array_equal(labels[test], saved["labels"])
        np.testing.assert_array_equal(probabilities, saved["probabilities"])
    scores = binary_metrics(labels[test], probabilities >= threshold)
    np.testing.assert_allclose(scores["balanced_accuracy"],
                               balanced_accuracy_score(labels[test], probabilities >= threshold))
    result = dict(selection, test=scores, test_balanced_accuracy=scores["balanced_accuracy"],
                  test_auroc=float(roc_auc_score(labels[test], probabilities)),
                  original_protocol=protocol)
    decoder.write_json(output / "results.json", result, exclusive=True)
    np.savez_compressed(output / "test_predictions.npz", labels=labels[test],
                        probabilities=probabilities, predicted=probabilities >= threshold,
                        endpoint_bins=endpoints[test])
    print("Aeon", json.dumps(scores), "threshold", threshold, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", choices=("aeon", "churchland", "both"), default="both")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for name in ("aeon", "churchland"):
            if args.dataset in (name, "both"):
                globals()[name](args.output / name)

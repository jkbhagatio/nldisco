"""Fit the missing Table S3 decoders using the existing wheel-decoder protocol.

Run ``uv run --no-sync python -m experiments.aeon.fit_appendix_decoders``.
Feature definitions and frozen neural representations are never refitted.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
from beartype import beartype
from jaxtyping import Bool, Integer, jaxtyped
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.decoder_metrics import binary_metrics, select_balanced_threshold

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/ceph/aeon/aeon/nldisco/aeon/sweeps/20260921_18h_s20")
SOURCE = ROOT / "paper_revision_wheelcms_20260921"
PREVIOUS = REPO / "experiments/outputs/decoder_balanced_accuracy_20260923/aeon/results.json"
OUTPUT = REPO / "experiments/aeon/outputs/appendix_decoders_20260925"
FEATURES = {"wheel": (192, 52), "area": (256, 107), "speed": (256, 109), "direction": (256, 10)}
GRID = (.001, .01, .1, 1., 10., 100., 1000.)
BOUNDS = (0, 2592000, 2916000, 3240000)


@beartype
def write_json(path: Path, value: dict) -> None:
    """Write finite, reviewable numerical results."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@beartype
def sha256(path: Path) -> str:
    """Fingerprint a saved input or fitted decoder."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@jaxtyped(typechecker=beartype)
def split_masks(
    endpoints: Integer[np.ndarray, "n"], valid: Bool[np.ndarray, "n"],
    bounds: tuple = BOUNDS, width: int = 20,
) -> dict:
    """Split physical time before validity filtering, excluding crossing windows."""
    if width < 1 or len(bounds) != 4 or not np.all(np.diff(bounds) > 0):
        raise ValueError("Three increasing intervals and a positive window width are required.")
    if not np.all(np.diff(endpoints) > 0):
        raise ValueError("Window endpoints must be strictly increasing.")
    return {name: valid & (endpoints - width + 1 >= lo) & (endpoints < hi)
            for name, lo, hi in zip(("train", "validation", "test"), bounds[:-1], bounds[1:])}


@beartype
def fit_feature(name: str, output: Path, threads: int) -> list:
    """Freeze regularization and thresholds without scoring the test partition."""
    d, latent = FEATURES[name]
    folder = output / name
    folder.mkdir()
    with np.load(SOURCE / "feature_arrays.npz") as data:
        endpoints = data["endpoint_bins"]
        labels, valid = data[f"condition_{name}"], data[f"valid_{name}"]
        scalar = data[f"values_{name}"]
    run = ROOT / f"d{d}/k12_seed0"
    np.testing.assert_array_equal(endpoints, np.load(run / "endpoint_bins.npy"))
    values = np.load(run / "activations.npy", mmap_mode="r")
    assert values.shape == (len(endpoints), d)
    np.testing.assert_array_equal(values[:, latent], scalar)
    masks = split_masks(endpoints, valid)
    support = {}
    for partition, mask in masks.items():
        assert mask.any() and np.unique(labels[mask]).size == 2, (name, partition)
        support[partition] = dict(windows=int(mask.sum()), positives=int(labels[mask].sum()),
                                  negatives=int((~labels[mask]).sum()))
    train_y, val_y = labels[masks["train"]], labels[masks["validation"]]
    rows = []
    with threadpool_limits(limits=threads):
        for space in ("single", "all"):
            model_path = folder / f"{space}_decoder.joblib"
            if name == "wheel" and space == "single":
                previous = json.loads(PREVIOUS.read_text())
                shutil.copy2(previous["model_path"], model_path)
                fitted = joblib.load(model_path)
                probabilities = fitted["model"].predict_proba(fitted["scaler"].transform(
                    scalar[masks["validation"], None].astype(np.float64)))[:, 1]
                threshold = select_balanced_threshold(val_y, probabilities)
                assert threshold == previous["threshold"]
                selection = dict(chosen_c=previous["chosen_c"], threshold=threshold,
                                 validation_auroc=float(roc_auc_score(val_y, probabilities)),
                                 validation=binary_metrics(val_y, probabilities >= threshold),
                                 reused_from=str(PREVIOUS), candidates=[])
            else:
                selected = scalar[:, None] if space == "single" else values
                train_x = np.array(selected[masks["train"]], dtype=np.float64, order="C")
                val_x = np.array(selected[masks["validation"]], dtype=np.float64, order="C")
                assert np.isfinite(train_x).all() and np.isfinite(val_x).all()
                scaler = StandardScaler(copy=False).fit(train_x)
                scaler.transform(train_x)
                scaler.transform(val_x)
                candidates, best = [], None
                for c in GRID:
                    started = time.monotonic()
                    model = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                               max_iter=3000, tol=1e-5, random_state=0)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(train_x, train_y)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError(f"{name}/{space}, C={c}: fit did not converge")
                    probabilities = model.predict_proba(val_x)[:, 1]
                    auc = float(roc_auc_score(val_y, probabilities))
                    candidate = dict(c=c, validation_auroc=auc, iterations=int(model.n_iter_[0]),
                                     seconds=time.monotonic() - started)
                    candidates.append(candidate)
                    print(name, space, json.dumps(candidate), flush=True)
                    if best is None or auc > best[0]:
                        best = (auc, c, model, probabilities)
                auc, c, model, probabilities = best
                threshold = select_balanced_threshold(val_y, probabilities)
                # Saved scalers should not mutate callers' arrays during inference.
                scaler.copy = True
                joblib.dump(dict(scaler=scaler, model=model), model_path)
                selection = dict(chosen_c=c, threshold=threshold, validation_auroc=auc,
                                 validation=binary_metrics(val_y, probabilities >= threshold),
                                 candidates=candidates)
                del train_x, val_x
            row = dict(feature=name, space=space, D=d, k=12, latent=latent,
                       support=support, model_path=str(model_path), model_sha256=sha256(model_path),
                       **selection)
            write_json(folder / f"{space}_validation_selection.json", row)
            rows.append(row)
    return rows


@beartype
def score_test(row: dict, output: Path) -> dict:
    """Evaluate once using previously saved fits and validation thresholds."""
    name, space = row["feature"], row["space"]
    with np.load(SOURCE / "feature_arrays.npz") as data:
        endpoints = data["endpoint_bins"]
        mask = split_masks(endpoints, data[f"valid_{name}"])["test"]
        labels = data[f"condition_{name}"][mask]
        if space == "single":
            values = data[f"values_{name}"][mask, None].astype(np.float64)
        else:
            values = np.asarray(np.load(ROOT / f"d{row['D']}/k12_seed0/activations.npy",
                                        mmap_mode="r")[mask], dtype=np.float64)
    fitted = joblib.load(row["model_path"])
    probabilities = fitted["model"].predict_proba(fitted["scaler"].transform(values))[:, 1]
    predicted = probabilities >= row["threshold"]
    metrics = binary_metrics(labels, predicted)
    np.testing.assert_allclose(metrics["balanced_accuracy"], balanced_accuracy_score(labels, predicted),
                               atol=1e-15, rtol=0)
    if name == "wheel" and space == "single":
        assert metrics == json.loads(PREVIOUS.read_text())["test"]
    np.savez_compressed(output / name / f"{space}_test_predictions.npz",
                        endpoint_bins=endpoints[mask], labels=labels,
                        probabilities=probabilities, predicted=predicted)
    result = dict(row, test=metrics, test_auroc=float(roc_auc_score(labels, probabilities)))
    print("RESULT", name, space, json.dumps(metrics), flush=True)
    return result


@beartype
def main(output: Path, jobs: int, threads: int) -> None:
    """Fit all candidates, freeze all validation choices, then evaluate on test."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(REPO / "experiments/decoder_metrics.py", output / "decoder_metrics.py")
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    boundary_bins=BOUNDS, window_bins=20, c_grid=GRID, solver="lbfgs",
                    max_iter=3000, tol=1e-5, class_weight=None,
                    selection="Highest validation AUROC; exact ties choose smallest C; no refit.",
                    threshold="Maximize validation balanced accuracy; highest threshold breaks ties; p >= t.",
                    scaling="Training mean and standard deviation only, separately for each feature population.",
                    test_policy="All fits and thresholds frozen before any new test scores are calculated.",
                    scope="Frozen representations and exploratory feature definitions use the full recording.",
                    features=json.loads((SOURCE / "feature_definitions.json").read_text()),
                    input_path=str(SOURCE / "feature_arrays.npz"),
                    input_sha256=sha256(SOURCE / "feature_arrays.npz"))
    write_json(output / "protocol.json", protocol)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fit_feature, name, output, threads) for name in FEATURES]
        selections = [row for future in futures for row in future.result()]
    write_json(output / "validation_selection.json", dict(results=selections))
    with threadpool_limits(limits=threads):
        results = [score_test(row, output) for row in selections]
    write_json(output / "results.json", dict(protocol=protocol, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    main(args.output, args.jobs, args.threads)

"""Train independent vanilla SEDs matching the Figure 3 model provenance."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch as t
from torch.utils.data import DataLoader
from typeguard import typechecked

from experiments.synthetic.scripts.dataset_metadata import load_parameters
from experiments.synthetic.scripts.feature_selection import (
    SelectionConfig,
    best_features,
    biological_traceback,
    compute_tuning_curves,
    ground_truth_curves,
    normalize_tuning_curves,
    score_tuning_curves,
)
from nldisco.config import EncoderConfig, LossConfig, SedConfig, TrainConfig
from nldisco.data import SpikeWindowDataset
from nldisco.model import build_sed
from nldisco.train import evaluate_model, train_model
from nldisco.util import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "experiments/synthetic/data/matryoshka_overlap"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments/synthetic/outputs/runs/vanilla"
VANILLA_SEQ_LEN = 1
VANILLA_N_FEATURES = 128
VANILLA_TOPK = 4
VANILLA_BATCH_SIZE = 1024


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def _jsonable_training(training: object) -> Dict[str, Dict[str, float]]:
    return {
        key: {str(step): value for step, value in history.items()}
        for key, history in asdict(training).items()
    }


@typechecked
def run_seed(seed: int, output_dir: Path, epochs: int, device: t.device) -> Dict[str, object]:
    """Train, evaluate, score, and save one independent vanilla SED."""

    started = time.time()
    run_dir = output_dir / f"seed_{seed:02d}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    set_seed(seed)
    raw_spike_counts = np.load(DATA_DIR / "spike_matrix.npy").T.astype(np.float32)
    positions = np.load(DATA_DIR / "positions.npy").astype(np.float32)
    if raw_spike_counts.shape != (60000, 100):
        raise ValueError("Paper simulation must contain 60000 bins and 100 units")
    parameters = load_parameters(DATA_DIR)
    selection_config = SelectionConfig(**parameters["selection_config"])
    for key, expected in (
        ("centers_m", selection_config.cell_centers),
        ("left_widths_m", selection_config.left_widths),
        ("right_widths_m", selection_config.right_widths),
        ("peak_rates_hz", selection_config.peak_amplitudes_hz),
    ):
        np.testing.assert_array_equal(parameters[key], expected)
    train_mean = raw_spike_counts.mean(axis=0, keepdims=True)
    train_std = raw_spike_counts.std(axis=0, keepdims=True)
    if np.any(train_std == 0):
        raise ValueError("Every neuron must have nonzero training-set variance")
    spike_counts = (raw_spike_counts - train_mean) / train_std + 1e-8
    spike_tensor = t.from_numpy(spike_counts)
    train_data = SpikeWindowDataset(spike_tensor, VANILLA_SEQ_LEN)
    loader_generator = t.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=VANILLA_BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )
    discovery_loader = DataLoader(
        train_data,
        batch_size=VANILLA_BATCH_SIZE,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    sed_config = SedConfig(
        n_neurons=spike_counts.shape[1],
        seq_len=VANILLA_SEQ_LEN,
        dsed_topk_map={VANILLA_N_FEATURES: VANILLA_TOPK},
        encoder=EncoderConfig(type="FlatWindow"),
        dtype=t.bfloat16 if device.type == "cuda" else t.float32,
    )
    loss_config = LossConfig(timebin_weights=[1.0], type="msle", tau=1.0)
    train_config = TrainConfig(
        epochs=epochs,
        batch_size=VANILLA_BATCH_SIZE,
        learning_rate=5e-3,
        dead_feature_window=max(1, len(train_loader) // 3),
    )
    sed = build_sed(sed_config).to(device)
    training = train_model(sed, train_loader, loss_config, train_config)
    evaluation = evaluate_model(sed, discovery_loader, loss_config, replica_id=seed)

    decoder = sed.decoder.weight.detach().float().cpu().numpy()[0].T
    tuning_curves, position_centers = compute_tuning_curves(
        evaluation.activation_table,
        evaluation.evaluation_index,
        positions,
        sed_config.n_features,
        selection_config,
    )
    scores = score_tuning_curves(tuning_curves, position_centers, selection_config)
    scores, traces = biological_traceback(
        evaluation.activation_table, spike_counts, positions, decoder, scores, selection_config
    )
    activation_payload = {
        column: evaluation.activation_table[column].to_numpy()
        for column in evaluation.activation_table
    }
    activation_payload["eligible_source_time_idx"] = (
        evaluation.evaluation_index.source_time_idx.to_numpy()
    )
    selected = best_features(scores)
    run_dir.mkdir(parents=True, exist_ok=True)
    scores.insert(0, "seed", seed)
    scores.to_csv(run_dir / "feature_scores.csv", index=False)
    np.savez_compressed(
        run_dir / "tuning_curves.npz",
        position_centers=position_centers,
        tuning_curves=tuning_curves,
        normalized_tuning_curves=normalize_tuning_curves(tuning_curves),
        ground_truth_curves=ground_truth_curves(position_centers, selection_config),
    )
    np.savez_compressed(run_dir / "traceback.npz", unit_ids=np.arange(100), **traces)
    np.savez_compressed(run_dir / "activations.npz", **activation_payload)
    t.save(
        {
            "seed": seed,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in sed.state_dict().items()
            },
            "sed_config": {
                "architecture": "vanilla_per_time_bin",
                "n_neurons": sed_config.n_neurons,
                "seq_len": sed_config.seq_len,
                "dsed_topk_map": sed_config.dsed_topk_map,
                "encoder_type": sed_config.encoder.type,
                "dtype": str(sed_config.dtype),
                "inference_sparsity": sed_config.inference_sparsity,
            },
            "preprocessing": {
                "type": "per_neuron_z_score",
                "fit_rows": [0, len(positions)],
                "mean": t.from_numpy(train_mean),
                "std": t.from_numpy(train_std),
            },
            "loss_config": asdict(loss_config),
            "train_config": asdict(train_config),
            "dataset_parameters": parameters,
            "data_usage": "All rows used for training and descriptive feature discovery",
        },
        run_dir / "checkpoint.pt",
    )
    summary: Dict[str, object] = {
        "seed": seed,
        "architecture": "vanilla_per_time_bin",
        "model_config": {
            "seq_len": 1,
            "n_features": 128,
            "topk": 4,
            "loss": asdict(loss_config),
            "training": asdict(train_config),
            "dtype": str(sed_config.dtype),
        },
        "dataset": str(DATA_DIR.relative_to(PROJECT_ROOT)),
        "dataset_sha256": hashlib.sha256((DATA_DIR / "spike_matrix.npy").read_bytes()).hexdigest(),
        "data_usage": "All rows used for training and descriptive feature discovery; no held-out generalization estimate",
        "discovery_rows": [0, len(positions)],
        "decoder_loading_definition": "Signed weights in each unit-L2-norm decoder column; threshold 0.1 is an absolute normalized loading",
        "traceback_definition": "Mean training-standardized biological unit activity at unique source bins where latent activation > 0; coactivity, not causal attribution.",
        "device": str(device),
        "epochs": epochs,
        "n_train_examples": len(train_data),
        "n_discovery_examples": len(train_data),
        "weighted_discovery_reconstruction": evaluation.weighted_reconstruction,
        "inference_sparsity": evaluation.inference_sparsity,
        "selection_config": selection_config.to_dict(),
        "selected_features": selected,
        "training_history": _jsonable_training(training),
        "metrics_by_lag": evaluation.metrics_by_lag.to_dict(orient="records"),
        "elapsed_seconds": time.time() - started,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({key: summary[key] for key in ("seed", "selected_features", "elapsed_seconds")})
    )
    return summary


def main() -> None:
    """Run requested seeds sequentially on one assigned device."""

    args = parse_args()
    t.set_num_threads(args.cpu_threads)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    device = t.device(args.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection_config.json"
    frozen_selection = SelectionConfig(**load_parameters(DATA_DIR)["selection_config"]).to_dict()
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != json.loads(json.dumps(frozen_selection)):
            raise ValueError("Selection differs from the frozen pre-training configuration")
    else:
        selection_path.write_text(json.dumps(frozen_selection, indent=2) + "\n")
    for seed in args.seeds:
        run_seed(seed, args.output_dir, args.epochs, device)


if __name__ == "__main__":
    main()

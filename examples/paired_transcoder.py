"""Train a synthetic 5-input/3-output SED on a shared time grid.

Run from the repository: uv run python examples/paired_transcoder.py --layout flat
Other layouts: single_bin, transformer, shift_equivariant. No data files are required.
"""

import argparse

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import DataLoader

from nldisco.config import DecoderConfig, EncoderConfig, LossConfig, SedConfig, TrainConfig
from nldisco.data.preprocessing import fit_normalizer
from nldisco.data.window import SpikeWindowDataset
from nldisco.evaluation import AblationConfig, EvaluationConfig, SpectralConfig
from nldisco.model import build_sed
from nldisco.sweep.training import spectral_validation_loader
from nldisco.train import evaluate_model, train_model


@beartype
def main() -> None:
    """Fit separate training-row normalizers, train, evaluate, and infer from inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        default="flat",
        choices=("single_bin", "flat", "transformer", "shift_equivariant"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)

    # Repeated synthetic cycles give both splits the same nonnegative target range.
    phase = (np.arange(320) % 64) * (2 * np.pi / 64)
    first, second = 1 + np.sin(phase), 1 + np.cos(phase)
    input_rows = np.column_stack((first, second, first + second, first**2, second**2))
    target_rows = np.column_stack((first, second, first * second))
    train_rows = np.arange(len(phase)) < 256
    input_stats = fit_normalizer(input_rows[train_rows], "zscore")
    target_stats = fit_normalizer(target_rows[train_rows], "minmax")
    inputs = torch.from_numpy(input_stats.transform(input_rows)).float()
    targets = torch.from_numpy(target_stats.transform(target_rows)).float()

    temporal = args.layout == "shift_equivariant"
    seq_len = 1 if args.layout == "single_bin" else 4
    config = SedConfig(
        n_neurons=inputs.shape[1],
        n_output_neurons=targets.shape[1],
        seq_len=seq_len,
        dsed_topk_map={8: 2, 16: 4},
        encoder=EncoderConfig(
            type="TransformerWindow"
            if args.layout in {"transformer", "shift_equivariant"}
            else "FlatWindow",
            shift_equivariant=temporal,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_feedforward=32,
            attention_radius=1,
        ),
        decoder=DecoderConfig(output_activation="softplus", temporal_kernel_len=2),
    )
    model = build_sed(config)
    training = DataLoader(
        SpikeWindowDataset(inputs, seq_len, targets=targets, allowed_rows=train_rows),
        batch_size=32,
        shuffle=True,
    )
    validation = DataLoader(
        SpikeWindowDataset(
            inputs,
            seq_len,
            targets=targets,
            allowed_rows=~train_rows,
            stride=config.usable_occurrence_positions if temporal else seq_len,
            occurrence_support=config.temporal_occurrence_support() if temporal else None,
        ),
        batch_size=32,
    )
    loss = LossConfig(type="mse", timebin_weights=[1.0] * seq_len)
    train_model(model, training, loss, TrainConfig(epochs=args.epochs, batch_size=32))
    result = evaluate_model(
        model,
        validation,
        loss,
        diagnostics=EvaluationConfig(
            ablation=AblationConfig(enabled=True, latents=[0, 1]),
            spectral=SpectralConfig(enabled=True, bin_size=0.02, segment_duration=0.32),
        ),
        spectral_loader=spectral_validation_loader(validation),
    )

    # Inputs alone suffice at inference; convert predictions back to target units.
    model.eval()
    with torch.no_grad():
        prediction = model(inputs[256 : 256 + seq_len][None]).reconstructions[16]
    prediction_original_units = prediction.numpy() * target_stats.scale + target_stats.offset
    print(f"Validation target MSE: {result.weighted_reconstruction:.6f}")
    print(f"Input-only inference output shape: {prediction_original_units.shape}")
    print(result.diagnostics.ablation.query("scope == 'window'").to_string(index=False))


if __name__ == "__main__":
    main()

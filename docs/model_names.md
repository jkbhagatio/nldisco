# Models and inference

Use these names consistently in library code, notebooks, experiments, and documentation:

| Code / documentation | Paper abbreviation | Meaning |
| --- | --- | --- |
| `Window` | W-SED | Family of SEDs that encode and reconstruct a multi-bin window |
| `FlatWindow` | FW-SED | Vectorize the window and apply a learned linear projection |
| `TransformerWindow` | TW-SED | Integrate population-state tokens using self-attention |

`Window` is the family name, not a third encoder option. `WindowConfig` describes
window construction. Select a variant through `EncoderConfig.type`:

```python
from nldisco import EncoderConfig, SedConfig, build_sed

config = SedConfig(
    n_neurons=100,
    seq_len=20,
    dsed_topk_map={128: 4},
    encoder=EncoderConfig(type="TransformerWindow"),
)
model = build_sed(config)
```

The encoder implementations are `FlatWindowEncoder` and `TransformerWindowEncoder`,
exported from `nldisco.model`. The FlatWindow encoder also handles a one-bin vanilla
SED; the name alone does not imply that the configured sequence length exceeds one.

For compatibility, `flat` and `flat_window` normalize to `FlatWindow`, and
`temporal_transformer` normalizes to `TransformerWindow`. New serialized configs
use canonical names. The old `TemporalTransformerEncoder` import remains an alias
for existing imports and full-module checkpoints. Tensor state-dictionary keys
and model computations are unchanged.

## Matryoshka levels and paired populations

`dsed_topk_map={128: 4, 512: 8}` specifies nested dictionaries: the first 128 latents
use training sparsity 4, and all 512 use sparsity 8. Each level reconstructs the full
target window. Independent replicas are separate training runs, not a tensor axis.

Set `n_output_neurons` to reconstruct a different population; it defaults to
`n_neurons`. Both populations have the same window length. See
[paired preprocessing](preprocessing.md#paired-populations).

## Temporal motifs

`EncoderConfig(type="TransformerWindow", shift_equivariant=True)` removes absolute
positions, uses local attention, sparsifies across time-feature occurrences, and
uses a shared temporal-convolution decoder. `attention_radius` and the decoder's
`temporal_kernel_len` determine occurrence support. This mode requires equal
`LossConfig.timebin_weights`; other modes allow nonuniform positive weights.

## Inference

Training applies BatchTopK across each minibatch and tracks an exponential moving
average of its activation cutoff. After optimization, `train_model` runs a
no-gradient calibration pass and stores the mean batch cutoff. By default,
`inference_sparsity="training_threshold"` applies that fixed threshold at evaluation,
so sparsity can vary by example without depending on evaluation batch size.
`inference_sparsity="sample_topk"` instead selects a fixed TopK per example (across
supported time-feature positions for temporal codes).

Shift-equivariant evaluation requires each physical occurrence exactly once. The
runner strides by the number of fully supported positions and, when needed, adds
an end-aligned tail window while masking repeated occurrences. Reconstruction still
uses the full code for each window. For a manual loader, pass
`stride=config.usable_occurrence_positions` and
`occurrence_support=config.temporal_occurrence_support()` to `SpikeWindowDataset`.
See the [runnable example](../examples/paired_transcoder.py).

# Installation

NLDisco requires **Python 3.9–3.13**.

## Work from source

Download the source ZIP from the
[anonymized repository](https://anonymous.4open.science/r/F3E2-ocsidln), extract it,
and open a terminal in the extracted repository root (the directory containing
`pyproject.toml`). Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then run:

```console
uv sync --locked
uv run python examples/paired_transcoder.py --layout flat
```

`uv sync --locked` creates a project environment in `.venv`, installs NLDisco in
editable mode, and includes the development tools. It uses the committed lockfile
and will provision a compatible Python interpreter when needed. Changes to
`src/nldisco/` are immediately visible in that environment.

The example generates its own data and runs training, evaluation, and inference.
Other layouts are `single_bin`, `transformer`, and `shift_equivariant`; add
`--epochs 1` for a shorter run.

To work from the downloaded source in an existing environment instead,
activate that environment and use either:

```console
uv pip install -e .
```

```console
python -m pip install -e .
```

These editable installs do not install the development dependency group or use
`uv.lock`. Run `python examples/paired_transcoder.py --layout flat` in that active
environment. The `uv sync --locked` route is recommended for reproducing the
repository environment and running its tests.

## Optional experiment dependencies

The base install includes the NLDisco library and training CLI. CEBRA comparisons
require the `analysis` extra; Churchland data preparation requires `churchland`.
From a source checkout:

```console
uv sync --locked --extra analysis
uv sync --locked --extra analysis --extra churchland
```

Extras install Python dependencies, not experiment data or saved checkpoints.
See the [experiment instructions](../experiments/README.md) for those inputs.

## Running the guides

Commands prefixed with `uv run` assume a source checkout prepared with `uv sync`.
When using an existing environment, activate it and omit `uv run`: for example,
`python -m nldisco.sweep --config-name train data.path=/path/to/counts.npy`.
Paths such as `examples/`, `experiments/`, and `tests/` require a source checkout.

Training defaults to CPU. GPU execution requires a PyTorch installation compatible
with your hardware; follow [PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
and the [training guide](training_and_sweeps.md) for device selection.

## Development

From a checkout prepared with `uv sync --locked`:

```console
uv run pytest
uv run pytest experiments/tests
uv run jupyter lab
```

Some experiment tests require the optional dependencies above. To register the
project environment as a notebook kernel:

```console
uv run python -m ipykernel install --user --name=nldisco
```

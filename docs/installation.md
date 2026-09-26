# Installation

NLDisco requires **Python 3.9–3.13**.

## Install from PyPI into an existing environment

Activate your Python virtual environment or Conda environment, then use either:

```console
uv pip install nldisco
```

```console
python -m pip install nldisco
```

No repository checkout is needed for the library or training CLI. To target a
specific interpreter with uv, use `uv pip install --python /path/to/env/bin/python nldisco`.
For a project managed by uv, use `uv add nldisco` to record the dependency.

Verify the installation:

```console
python -c "from nldisco import SedConfig, build_sed; print('NLDisco import OK')"
python -m nldisco.sweep --config-name train --cfg job
```

The second command displays the training configuration without starting a run.
The examples and experiment datasets/notebooks are in the source repository;
they are not installed with the wheel.

## Work from source

Clone the repository and install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```console
git clone https://github.com/jkbhagatio/nldisco.git nldisco
cd nldisco
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

To work from the cloned/extracted source in an existing environment instead,
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

For an existing environment installed from PyPI:

```console
uv pip install "nldisco[analysis,churchland]"
# Or:
python -m pip install "nldisco[analysis,churchland]"
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

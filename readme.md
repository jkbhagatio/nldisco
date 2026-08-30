# NLDisco

**Ne**ural **L**atent **Disco**very pipeline
<br>
<img width="755" height="153" alt="Screenshot 2025-09-28 at 16 35 19" src="https://github.com/user-attachments/assets/2414056b-2611-4fb9-a904-3abf7d221606" />

<br>

[NeurIPS 2025 Data on the Brain & Mind Workshop paper](https://openreview.net/pdf?id=cPpMl7Y2y3)

---

NLDisco trains shallow, overcomplete, sparse encoder-decoder (SED) neural network models, in which individual dictionary elements -- hidden layer neurons -- represent learned interpretable latents.

_**Note**_: The codebase is functional, but still a work in progress.

## Getting started

We recommend following our NeurIPS 2025 Data on the Brain & Mind Workshop tutorial, [here](notebooks/NLDisco_tutorial.ipynb). It walks through Python environment setup, downloading sample data, and step-by-step instructions and explanations for each stage of the pipeline.

## Development environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the
locked project environment:

    uv sync

Run commands inside the environment with uv run, for example:

    uv run pytest
    uv run jupyter lab

For VS Code or another Jupyter client, register the project kernel once:

    uv run ipython kernel install --user --env VIRTUAL_ENV "$(pwd)/.venv" --name=nldisco

Select the **nldisco** kernel when running the notebooks.

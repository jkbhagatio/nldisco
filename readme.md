# NLDisco

**Ne**ural **L**atent **Disco**very pipeline

<br>

<img width="755" height="153" alt="Screenshot 2025-09-28 at 16 35 19" src="https://github.com/user-attachments/assets/2414056b-2611-4fb9-a904-3abf7d221606" />

---

NLDisco trains shallow, overcomplete, sparse encoder-decoder (SED) neural network models, in which individual dictionary elements -- hidden layer neurons -- represent learned interpretable latents.

## Getting started

If this is your first time using NLDisco, we recommend following our tutorial, which you can find [here](notebooks/NLDisco_tutorial.ipynb). It walks you through environment setup, downloading the required data, and provides step-by-step instructions and explanations for each stage of the pipeline.

## Environment set-up

### With [pixi](https://pixi.sh/latest/tutorials/python) (recommended)

Prerequisites:

- An installed version of [pixi](https://pixi.sh/latest/)

After cloning this project, in its root directory, just run `pixi install --manifest-path ./pyproject.toml` -- this will create a Python environment located in the '.pixi' subdirectory within 'nldisco'. Then run `pixi run postinstall` to complete the setup required for running the notebooks.

### Other

All package dependencies are specified in the `pyproject.toml`. You can format them as required for your favorite python environment / package management tool, and install them using this tool (e.g. via uv, pip, poetry, conda, etc.)

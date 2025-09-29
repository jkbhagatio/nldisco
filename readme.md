# NLDisco

**Ne**ural **L**atent **Disco**very pipeline

<br>

<img width="755" height="153" alt="Screenshot 2025-09-28 at 16 35 19" src="https://github.com/user-attachments/assets/2414056b-2611-4fb9-a904-3abf7d221606" />

---

NLDisco trains shallow, overcomplete, sparse encoder-decoder (SED) neural network models, in which individual dictionary elements -- hidden layer neurons -- represent learned interpretable latents.

## Environment set-up

### With [pixi](https://pixi.sh/latest/tutorials/python) (recommended)

Prerequisites:

- An installed version of [pixi](https://pixi.sh/latest/)

After cloning this project, in its root directory, just run `pixi install --manifest-path ./pyproject.toml` --- this will create a Python environment located in the '.pixi' subdirectory within 'nldisco'.

### Other

All package dependencies are specified in the 'pyproject.toml'. You can format them as required for your favorite python environment / package management tool, and install them using this tool (e.g. via uv, pip, poetry, conda, etc.)

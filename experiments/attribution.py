"""Shared experiment source traceback: gradient-times-input through the SED encoder.

For a latent ``d`` with pre-activation ``z_d = f(x)``, each source entry ``x[s, i]`` (time bin
``s``, unit ``i``) is credited ``x[s, i] * dz_d / dx[s, i]``. For the linear encoders this is
exactly the encoder weight multiplied by the source activity; for transformer encoders it is the
first-order generalization of that product.
"""

import numpy as np
import torch as t
from beartype import beartype
from beartype.typing import Iterable
from jaxtyping import Float, jaxtyped
from torch import Tensor

from nldisco.model import Sed


@jaxtyped(typechecker=beartype)
def encoder_attribution(
    model: Sed, windows: Float[Tensor, "batch time unit"], latent: int
) -> Float[Tensor, "batch time unit"]:
    """Attribute one latent's endpoint pre-activation to every source entry of each window.

    The endpoint is the last time bin for shift-equivariant (temporal) codes and the single
    global code otherwise. Inputs must already be in the model's training representation.
    """
    if not 0 <= latent < model.cfg.n_features:
        raise ValueError(f"latent must lie in [0, {model.cfg.n_features})")
    with t.enable_grad():
        x = windows.detach().requires_grad_(True)
        pre_acts = model.encoder(x)
        target = pre_acts[:, -1, latent] if pre_acts.ndim == 3 else pre_acts[:, latent]
        (grad,) = t.autograd.grad(target.sum(), x)
    return x.detach() * grad


@beartype
def mean_encoder_attribution(
    model: Sed, batches: Iterable[Float[Tensor, "batch time unit"]], latent: int
) -> Float[np.ndarray, "time unit"]:
    """Average ``encoder_attribution`` over every window in a stream of batches."""
    model.eval()
    device = next(model.parameters()).device
    total, count = None, 0
    for batch in batches:
        contribution = encoder_attribution(model, batch.to(device), latent).double().sum(dim=0)
        total = contribution if total is None else total + contribution
        count += len(batch)
    if not count:
        raise ValueError("No windows were provided")
    return (total / count).cpu().numpy()

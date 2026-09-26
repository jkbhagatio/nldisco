"""Shared probability curves for the current Churchland figures."""
import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

@jaxtyped(typechecker=beartype)
def probability_curve(x: Float[np.ndarray, 'n'], active: Bool[np.ndarray, 'n'],  # noqa: F821
                      edges: Float[np.ndarray, 'edges']) -> Float[np.ndarray, 'bins']:  # noqa: F821
    """Pool windows in half-open plotting bins; omit cells with fewer than 20 rows."""
    index = np.searchsorted(edges, x, side='right') - 1
    good = np.isfinite(x) & (index >= 0) & (index < len(edges) - 1)
    n = np.bincount(index[good], minlength=len(edges) - 1)
    total = np.bincount(index[good], weights=active[good], minlength=len(edges) - 1)
    return np.divide(total, n, out=np.full(len(n), np.nan), where=n >= 20)


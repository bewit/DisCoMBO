import numpy as np
from spn.structure.Base import get_nodes_by_type
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf


class _MinVarianceLeaf:
    """Picklable ``leaves=`` callback for SPFlow's ``learn_parametric`` that
    floors every Gaussian leaf's stdev to a per-column lower bound.

    Module-level (not a closure) so it survives pickling for SPFlow's
    ``multiprocessing.Pool``-based leaf creation.
    """

    def __init__(self, floor: np.ndarray):
        self.floor = floor

    def __call__(self, slice_data, ds_context, scope):
        node = create_parametric_leaf(slice_data, ds_context, scope)
        for g in get_nodes_by_type(node, Gaussian):
            col = g.scope[0]
            if g.stdev < self.floor[col]:
                g.stdev = float(self.floor[col])
        return node


def make_min_variance_leaf(
    data: np.ndarray,
    *,
    epsilon_rel: float = 0.5,
    epsilon_abs: float = 0.1,
) -> _MinVarianceLeaf:
    """Build a ``leaves=`` callback for SPFlow's ``learn_parametric`` that
    floors the stdev of every Gaussian leaf to
    ``max(epsilon_abs, epsilon_rel * sigma_j)``, where ``sigma_j`` is the
    marginal stdev of column ``j`` over the full training data.

    The floor is computed once from ``data`` so it is a fixed prior, not a
    slice-dependent quantity. Non-Gaussian leaves (Bernoulli, Categorical)
    pass through untouched.
    """
    sigma = np.nanstd(data, axis=0)
    floor = np.maximum(epsilon_abs, epsilon_rel * sigma)
    return _MinVarianceLeaf(floor)

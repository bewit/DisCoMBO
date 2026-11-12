"""
    Bottom-up p-value inference on DisCoMBO circuits (formerly `membership_circuits.models.nodewise.spflow_pvalue`).

    Leaves:   Normal -> CDF tail probability, Categorical/Bernoulli -> PMF of the observation.
              Marginalized (NaN) values -> 1.
    Sums:     weighted arithmetic mean of the children's p-values, optionally multiplied by the
              correction factor min(2, 1 / max(weights)).
    Products: Fisher's method. With K children, T = -2 * sum(log p_k) is chi^2 distributed with 2K degrees of
              freedom under the null, hence p = P(chi^2_{2K} >= T) = Q(K, T / 2) (regularized upper incomplete
              gamma function). Marginalized children (p = 1) still count towards K, as in the legacy implementation.
"""
import numpy as np
import torch
from spflow.modules.leaves import Bernoulli, Categorical, Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.ops import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum

from .inference import _num_threads
from .leaves import DTYPE

# lower clip of children's p-values in Fisher's method (log(0) is undefined)
FISHER_MIN_PVALUE = 1e-300


def _normal_pvalue(leaf: Normal, x: torch.Tensor, side: str) -> torch.Tensor:
    z = (x - leaf.loc[0, 0, 0]) / leaf.scale[0, 0, 0]
    if side == 'left':
        return torch.special.ndtr(z)
    if side == 'right':
        return torch.special.ndtr(-z)
    return 2.0 * torch.special.ndtr(-torch.abs(z))


def _categorical_pvalue(leaf: Categorical, x: torch.Tensor) -> torch.Tensor:
    probs = leaf.probs[0, 0, 0]
    idx = torch.trunc(torch.nan_to_num(x, nan=0.0))
    valid = (idx >= 0) & (idx < leaf.K)
    return torch.where(valid, probs[idx.clamp(0, leaf.K - 1).long()], torch.zeros_like(x))


def _bernoulli_pvalue(leaf: Bernoulli, x: torch.Tensor) -> torch.Tensor:
    logit = leaf.logits[0, 0, 0]
    return torch.where(x == 1, torch.sigmoid(logit), torch.sigmoid(-logit))


def _leaf_pvalue(leaf, data: torch.Tensor, side: str) -> torch.Tensor:
    if len(leaf.scope.query) != 1:
        raise NotImplementedError("p-values are only implemented for univariate leaves.")
    x = data[:, leaf.scope.query[0]].contiguous()
    marginalized = torch.isnan(x)
    if isinstance(leaf, Normal):
        p = _normal_pvalue(leaf, x, side)
    elif isinstance(leaf, Categorical):
        p = _categorical_pvalue(leaf, x)
    elif isinstance(leaf, Bernoulli):
        p = _bernoulli_pvalue(leaf, x)
    else:
        raise NotImplementedError(f"p-values are not implemented for {type(leaf).__name__} leaves.")
    return torch.where(marginalized, torch.ones_like(p), p).reshape(-1, 1, 1)


def fishers_method(pvalues: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Combine independent p-values along `dim` with Fisher's method (keeps `dim` with size 1)."""
    pvalues = pvalues.clamp(FISHER_MIN_PVALUE, 1.0)
    k = torch.tensor(float(pvalues.shape[dim]), dtype=pvalues.dtype)
    statistic = -2.0 * torch.log(pvalues).sum(dim=dim, keepdim=True)
    return torch.special.gammaincc(k, statistic / 2.0)


def _pvalue(module: Module, data: torch.Tensor, side: str, corrected: bool, memo: dict) -> torch.Tensor:
    """p-values of a module with shape (N, features, channels)."""
    key = id(module)
    if key in memo:
        return memo[key]

    if isinstance(module, Product):
        result = fishers_method(_pvalue(module.inputs, data, side, corrected, memo), dim=1)
    elif isinstance(module, Sum):
        child_pvalues = _pvalue(module.inputs, data, side, corrected, memo)
        weights = module.weights[..., 0]  # (features, in_channels, out_channels)
        result = torch.einsum('bfi,fio->bfo', child_pvalues, weights)
        if corrected:
            result = result * torch.clamp(1.0 / weights.amax(dim=1), max=2.0)
    elif isinstance(module, Cat):
        result = torch.cat([_pvalue(m, data, side, corrected, memo) for m in module.inputs], dim=module.dim)
    elif isinstance(module, LeafModule):
        result = _leaf_pvalue(module, data, side)
    else:
        raise NotImplementedError(f"p-values are not implemented for {type(module).__name__} modules.")

    memo[key] = result
    return result


def pvalues(circuit: Module, data, side: str = 'both', corrected: bool = False) -> np.ndarray:
    """
        Compute the p-value of each row of `data` under the circuit.

        Args:
            circuit (Module): Circuit learned with `learn_circuit` (or conditioned with `condition`).
            data: Array of shape (N, D); NaN marks marginalized variables.
            side (str, optional): Tail of continuous leaves: 'left', 'right' or 'both'. Defaults to 'both'.
            corrected (bool, optional): Apply the correction factor min(2, 1 / max(weights)) at sum nodes.
                Defaults to False (plain weighted arithmetic mean).

        Returns:
            np.ndarray: p-values in [0, 1] of shape (N,).
    """
    if side not in ('left', 'right', 'both'):
        raise ValueError(f"Unknown side: {side}")
    data = torch.tensor(np.asarray(data, dtype=float), dtype=DTYPE)
    with torch.no_grad(), _num_threads(1):
        result = _pvalue(circuit, data, side, corrected, memo={})
    return result.reshape(data.shape[0]).clamp(0.0, 1.0).numpy()

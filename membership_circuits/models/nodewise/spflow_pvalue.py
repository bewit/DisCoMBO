"""
P-value inference functions compatible with SPFlow's eval_spn_bottom_up.

Provides leaf-level p-value computation (Gaussian, Categorical) and
merging functions (adaptive corrected arithmetic mean for Sum nodes,
Fisher's method for Product nodes) that can be passed directly to
SPFlow's bottom-up evaluation engine.

Usage:
    from membership_circuits.models.nodewise.spflow_pvalue import pvalue_spflow
    pvals = pvalue_spflow(spn_root, data, side="both")
"""
import numpy as np
from functools import partial
from scipy.stats import norm, chi2
from spn.structure.Base import Sum, Product
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical, Bernoulli
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Inference import eval_spn_bottom_up


def gaussian_pvalue(node, data=None, dtype=np.float64, side="both", **kwargs):
    """Compute p-values at a Gaussian leaf node via CDF."""
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    col = data[:, node.scope[0]]
    marg_ids = np.isnan(col)
    if np.all(marg_ids):
        return probs

    obs = col[~marg_ids]
    cdf_vals = norm.cdf(obs, loc=node.mean, scale=node.stdev)

    if side == "left":
        pvals = cdf_vals
    elif side == "right":
        pvals = 1.0 - cdf_vals
    else:  # "both"
        pvals = 2.0 * np.minimum(cdf_vals, 1.0 - cdf_vals)

    probs[~marg_ids, 0] = pvals
    return probs


def categorical_pvalue(node, data=None, dtype=np.float64, **kwargs):
    """Compute p-values at a Categorical leaf node via PMF."""
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    col = data[:, node.scope[0]]
    marg_ids = np.isnan(col)
    if np.all(marg_ids):
        return probs

    obs = col[~marg_ids].astype(np.int64)
    valid = obs < node.k
    p_array = np.array(node.p, dtype=dtype)
    pvals = np.where(valid, p_array[np.clip(obs, 0, node.k - 1)], 0.0)
    probs[~marg_ids, 0] = pvals
    return probs


def bernoulli_pvalue(node, data=None, dtype=np.float64, **kwargs):
    """Compute p-values at a Bernoulli leaf node via PMF."""
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    col = data[:, node.scope[0]]
    marg_ids = np.isnan(col)
    if np.all(marg_ids):
        return probs
    obs = col[~marg_ids].astype(np.int64)
    pvals = np.where(obs == 1, node.p, 1.0 - node.p)
    probs[~marg_ids, 0] = pvals
    return probs


def histogram_pvalue(node, data=None, dtype=np.float64, side="both", **kwargs):
    """Compute p-values at a Histogram leaf node.

    REAL histograms: CDF-based (same tail logic as gaussian_pvalue).
    DISCRETE/BINARY histograms: PMF-based (consistent with categorical_pvalue).
    """
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    col = data[:, node.scope[0]]
    marg_ids = np.isnan(col)
    if np.all(marg_ids):
        return probs

    obs = col[~marg_ids]
    breaks = np.array(node.breaks, dtype=dtype)
    densities = np.array(node.densities, dtype=dtype)
    bin_widths = np.diff(breaks)
    bin_probs = densities * bin_widths  # probability mass per bin

    if node.meta_type in (MetaType.DISCRETE, MetaType.BINARY):
        # bin_width == 1 for integer categories, so bin_probs[i] == P(X=i)
        indices = np.clip(obs.astype(np.int64), 0, len(bin_probs) - 1)
        pvals = bin_probs[indices]
    else:
        cum_probs = np.concatenate([[0.0], np.cumsum(bin_probs)])
        bin_idx = np.clip(np.searchsorted(breaks, obs, side='right') - 1, 0, len(densities) - 1)
        frac = np.where(
            bin_widths[bin_idx] > 0,
            (obs - breaks[bin_idx]) / bin_widths[bin_idx],
            0.0,
        )
        cdf_vals = np.clip(cum_probs[bin_idx] + frac * bin_probs[bin_idx], 0.0, 1.0)
        if side == "left":
            pvals = cdf_vals
        elif side == "right":
            pvals = 1.0 - cdf_vals
        else:  # "both"
            pvals = 2.0 * np.minimum(cdf_vals, 1.0 - cdf_vals)

    probs[~marg_ids, 0] = pvals
    return probs


def sum_corrected_arithmetic_mean(node, children, corrected=False, dtype=np.float64, **kwargs):
    """Adaptive corrected weighted arithmetic mean of children p-values."""
    child_pvals = np.concatenate(children, axis=1)  # (N, K)
    weights = np.array(node.weights, dtype=dtype)
    if corrected:
        correction = min(2.0, 1.0 / max(weights))
    else:
        correction = 1.0
    combined = correction * (child_pvals @ weights)
    return combined.reshape(-1, 1)


def prod_fishers_method(node, children, dtype=np.float64, **kwargs):
    """Fisher's method: combine independent p-values via chi-squared."""
    child_pvals = np.concatenate(children, axis=1)  # (N, K)
    child_pvals = np.clip(child_pvals, 1e-300, 1.0)
    K = child_pvals.shape[1]
    T = -2.0 * np.sum(np.log(child_pvals), axis=1)
    combined = 1.0 - chi2.cdf(T, df=2 * K)
    return combined.reshape(-1, 1)


def pvalue_spflow(root, data, side="both", dtype=np.float64, corrected=False):
    """
    Compute p-values on an SPFlow PC via bottom-up evaluation.

    Args:
        root: SPFlow SPN root node (learned via learn_parametric / learn_mspn).
        data: numpy array of shape (N, D). NaN = marginalized variable.
        side: "left", "right", or "both" — controls tail direction at Gaussian leaves.
        dtype: numpy dtype for computation.
        corrected: If True, apply the adaptive correction factor
            (min(2, 1/max(weights))) at Sum nodes. Default False (plain
            weighted arithmetic mean).

    Returns:
        numpy array of shape (N, 1) with p-values in [0, 1].
    """
    eval_functions = {
        Gaussian: partial(gaussian_pvalue, side=side),
        Categorical: categorical_pvalue,
        Bernoulli: bernoulli_pvalue,
        Histogram: partial(histogram_pvalue, side=side),
        Sum: partial(sum_corrected_arithmetic_mean, corrected=corrected),
        Product: prod_fishers_method,
    }
    result = eval_spn_bottom_up(root, eval_functions, data=data, dtype=dtype)
    return np.clip(result, 0.0, 1.0)

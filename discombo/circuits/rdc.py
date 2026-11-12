"""
    Randomized dependence coefficient (RDC) column partitioning, ported from legacy SPFlow
    (`spn.algorithms.splitting.RDC.getIndependentRDCGroups_py`).

    spflow>=1.0 ships its own RDC, but it breaks rank ties arbitrarily (which distorts discrete
    hyperparameter columns), does not one-hot encode discrete columns and draws from the global torch RNG.
"""
import itertools
import warnings
from typing import List

import numpy as np
import scipy.stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cross_decomposition import CCA

from .leaves import DISCRETE

CCA_MAX_ITER = 100


def _ecdf(x: np.ndarray) -> np.ndarray:
    """Empirical CDF of a 1d array (ties share the maximal rank, NaNs map to 0)."""
    mv_ids = np.isnan(x)
    ranks = np.zeros(x.shape[0])
    ranks[~mv_ids] = scipy.stats.rankdata(x[~mv_ids], method='max') / np.sum(~mv_ids)
    return ranks


def _one_hot(x: np.ndarray, domain: np.ndarray) -> np.ndarray:
    encoded = np.zeros((x.shape[0], len(domain)))
    encoded[x[:, None] == domain[None, :]] = 1
    return encoded


def _rdc_features(local_data: np.ndarray, kinds: List[str], domains: List[np.ndarray], rand_gen: np.random.RandomState,
                  k: int = 10, s: float = 1.0 / 6.0, non_linearity=np.sin) -> List[np.ndarray]:
    """Copula transformation, random gaussian projection and non-linearity per column (legacy `rdc_transformer`)."""
    features = []
    for f in range(local_data.shape[1]):
        if kinds[f] == DISCRETE:
            features.append(_one_hot(local_data[:, f], domains[f]))
        else:
            features.append(local_data[:, f].reshape(-1, 1))

    ones = np.ones((local_data.shape[0], 1))
    features = [np.concatenate((np.apply_along_axis(_ecdf, 0, f), ones), axis=1) for f in features]
    features = [np.nan_to_num(f) for f in features]

    gaussians = [rand_gen.normal(size=(f.shape[1], k)) for f in features]
    projected = [non_linearity(s / f.shape[1] * np.dot(f, g)) for f, g in zip(features, gaussians)]
    return [np.concatenate((f, ones), axis=1) for f in projected]


def _rdc_cca(x: np.ndarray, y: np.ndarray) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x_cca, y_cca = CCA(n_components=1, max_iter=CCA_MAX_ITER).fit_transform(x, y)
            return np.corrcoef(x_cca.T, y_cca.T)[0, 1]
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def rdc_partition(local_data: np.ndarray, kinds: List[str], domains: List[np.ndarray], threshold: float,
                  rand_gen: np.random.RandomState) -> np.ndarray:
    """
        Partition the columns of `local_data` into groups that are (approximately) independent of each other.

        Returns:
            np.ndarray: component label per column, labels ordered by their smallest column index.
    """
    n_features = local_data.shape[1]
    features = _rdc_features(local_data, kinds, domains, rand_gen)

    adjacency = np.eye(n_features)
    for i, j in itertools.combinations(range(n_features), 2):
        adjacency[i, j] = adjacency[j, i] = _rdc_cca(features[i], features[j])

    adjacency[np.isnan(adjacency)] = 0
    adjacency[adjacency < threshold] = 0
    _, labels = connected_components(csr_matrix(adjacency), directed=False)
    return labels

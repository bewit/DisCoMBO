"""
    LearnSPN structure learning for DisCoMBO circuits.

    Mirrors legacy SPFlow's `learn_parametric` / `learn_mspn` (`spn.algorithms.StructureLearning`):
    breadth-first task queue, clustering first at the root, separate leaves for zero-variance columns,
    KMeans row clustering on the scope-local data, RDC column partitioning, univariate leaves and legacy
    pruning. The learned structure is converted into spflow>=1.0 `Sum`/`Product` modules.
"""
import warnings
from collections import deque
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.products import Product
from spflow.modules.sums import Sum

from .leaves import DTYPE, VarSpec, compute_domains, fit_histogram_leaf, fit_parametric_leaf
from .rdc import rdc_partition

PARAMETRIC = 'parametric'
HISTOGRAM = 'histogram'


class _Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLUMNS = 2
    SPLIT_ROWS = 3
    NAIVE_FACTORIZATION = 4
    REMOVE_UNINFORMATIVE_FEATURES = 5


class _SumNode:
    def __init__(self):
        self.children = []
        self.weights = []


class _ProductNode:
    def __init__(self):
        self.children = []


def _next_operation(local_data, scope, min_instances_slice, no_clusters, no_independencies, is_first):
    """Legacy `get_next_operation` with min_features_slice=1 and univariate leaves."""
    if len(scope) == 1:
        return _Operation.CREATE_LEAF, None

    uninformative = np.var(local_data, axis=0) == 0
    if np.any(uninformative):
        if np.all(uninformative):
            return _Operation.NAIVE_FACTORIZATION, None
        return _Operation.REMOVE_UNINFORMATIVE_FEATURES, np.flatnonzero(uninformative).tolist()

    if local_data.shape[0] <= min_instances_slice or (no_clusters and no_independencies):
        return _Operation.NAIVE_FACTORIZATION, None
    if no_independencies:
        return _Operation.SPLIT_ROWS, None
    if no_clusters:
        return _Operation.SPLIT_COLUMNS, None
    if is_first:
        return _Operation.SPLIT_ROWS, None
    return _Operation.SPLIT_COLUMNS, None


def _cluster_rows(local_data: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """KMeans row clustering (legacy `get_split_rows_KMeans`); too few rows yield a single cluster."""
    if local_data.shape[0] < n_clusters:
        return np.zeros(local_data.shape[0], dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(local_data)


def _prune(node):
    """
        Legacy `Prune`: contract internal nodes with a single child and merge children of the same type
        into their parent (multiplying sum weights). Merging nested products matters for Fisher's method,
        which is not associative.
    """
    internal = []
    queue = deque([node])
    while queue:
        n = queue.popleft()
        if isinstance(n, (_SumNode, _ProductNode)):
            internal.append(n)
            queue.extend(n.children)

    while internal:
        n = internal.pop()
        is_sum = isinstance(n, _SumNode)
        i = 0
        while i < len(n.children):
            c = n.children[i]
            if not isinstance(c, LeafModule) and len(c.children) == 1:
                n.children[i] = c.children[0]
                continue
            if type(n) == type(c):
                del n.children[i]
                n.children.extend(c.children)
                if is_sum:
                    w = n.weights[i]
                    del n.weights[i]
                    n.weights.extend([cw * w for cw in c.weights])
                continue
            i += 1
        if is_sum and i > 0:
            n.weights[0] = 1.0 - np.sum(n.weights[1:])

    if isinstance(node, (_SumNode, _ProductNode)) and len(node.children) == 1:
        node = node.children[0]
    return node


def _to_module(node) -> Module:
    if isinstance(node, LeafModule):
        return node
    inputs = [_to_module(c) for c in node.children]
    if isinstance(node, _ProductNode):
        return Product(inputs=inputs)
    weights = torch.tensor(node.weights, dtype=DTYPE)
    return Sum(inputs=inputs, weights=weights / weights.sum())


def learn_circuit(data: np.ndarray, specs: List[VarSpec], *, family: str = PARAMETRIC, min_instances_slice: int = 200,
                  threshold: float = 0.3, n_clusters: int = 2, alpha: float = 1.0, seed: int = 17,
                  domains: Optional[List[np.ndarray]] = None) -> Module:
    """
        Learn a circuit over all columns of `data` with LearnSPN.

        Args:
            data (np.ndarray): Training data of shape (N, D), discrete columns index-encoded.
            specs (List[VarSpec]): Statistical type of each of the D columns.
            family (str, optional): 'parametric' (Normal/Categorical/Bernoulli leaves, legacy `learn_parametric`)
                or 'histogram' (histogram leaves, legacy `learn_mspn`). Defaults to 'parametric'.
            min_instances_slice (int, optional): Slices with at most this many rows are factorized. Defaults to 200.
            threshold (float, optional): RDC threshold for column independence. Defaults to 0.3.
            n_clusters (int, optional): Number of row clusters per sum node. Defaults to 2.
            alpha (float, optional): Laplace smoothing of histogram leaves. Defaults to 1.0.
            seed (int, optional): Seed of KMeans and the RDC projections (legacy uses 17). Defaults to 17.
            domains (List[np.ndarray], optional): Global per-column domains (see `compute_domains`), e.g. of a
                larger dataset than `data`. Defaults to the domains of `data`.

        Returns:
            Module: spflow circuit (float64) with univariate leaves.
    """
    data = np.asarray(data, dtype=float)
    if data.shape[1] != len(specs):
        raise ValueError(f"Got {len(specs)} variable specs for {data.shape[1]} data columns.")
    if family not in (PARAMETRIC, HISTOGRAM):
        raise ValueError(f"Unknown leaf family: {family}")

    if domains is None:
        domains = compute_domains(data, specs)
    kinds = [spec.kind for spec in specs]
    rand_gen = np.random.RandomState(seed)

    def create_leaf(x, col):
        if family == HISTOGRAM:
            return fit_histogram_leaf(specs[col], col, x, domains[col], alpha=alpha)
        return fit_parametric_leaf(specs[col], col, x, domains[col])

    root = _ProductNode()
    root.children.append(None)
    tasks = deque([(data, root, 0, list(range(data.shape[1])), False, False)])
    while tasks:
        local_data, parent, children_pos, scope, no_clusters, no_independencies = tasks.popleft()
        operation, op_params = _next_operation(local_data, scope, min_instances_slice, no_clusters, no_independencies,
                                               is_first=parent is root)

        if operation == _Operation.REMOVE_UNINFORMATIVE_FEATURES:
            node = _ProductNode()
            parent.children[children_pos] = node
            for col in op_params:
                node.children.append(None)
                tasks.append((local_data[:, [col]], node, len(node.children) - 1, [scope[col]], True, True))
            rest_cols = [col for col in range(len(scope)) if col not in op_params]
            next_final = len(rest_cols) == 1
            node.children.append(None)
            tasks.append((local_data[:, rest_cols], node, len(node.children) - 1, [scope[col] for col in rest_cols],
                          next_final, next_final))

        elif operation == _Operation.SPLIT_ROWS:
            labels = _cluster_rows(local_data, n_clusters, seed)
            clusters = np.unique(labels)
            if len(clusters) == 1:
                tasks.append((local_data, parent, children_pos, scope, True, False))
                continue
            node = _SumNode()
            parent.children[children_pos] = node
            for cluster in clusters:
                cluster_data = local_data[labels == cluster, :]
                node.children.append(None)
                node.weights.append(cluster_data.shape[0] / local_data.shape[0])
                tasks.append((cluster_data, node, len(node.children) - 1, scope, False, False))

        elif operation == _Operation.SPLIT_COLUMNS:
            labels = rdc_partition(local_data, [kinds[s] for s in scope], [domains[s] for s in scope], threshold,
                                   rand_gen)
            clusters = np.unique(labels)
            if len(clusters) == 1:
                tasks.append((local_data, parent, children_pos, scope, False, True))
                continue
            node = _ProductNode()
            parent.children[children_pos] = node
            for cluster in clusters:
                cols = np.flatnonzero(labels == cluster)
                node.children.append(None)
                tasks.append((local_data[:, cols], node, len(node.children) - 1, [scope[c] for c in cols],
                              False, False))

        elif operation == _Operation.NAIVE_FACTORIZATION:
            node = _ProductNode()
            parent.children[children_pos] = node
            node.children.extend(create_leaf(local_data[:, col], scope[col]) for col in range(len(scope)))

        else:  # CREATE_LEAF
            parent.children[children_pos] = create_leaf(local_data[:, 0], scope[0])

    return _to_module(_prune(root.children[0]))

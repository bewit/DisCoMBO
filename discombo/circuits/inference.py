# Conditioning (`condition`) adapted from SPFlow (legacy branch, https://github.com/SPFlow/SPFlow/tree/legacy):
# spn/algorithms/Condition.py.
# Copyright 2018 Authors and Contributors (https://github.com/alejandromolinaml/SPFlow/).
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Modified by the DisCoMBO authors: ported to the PyTorch modules of SPFlow >= 1.0.
"""
    Inference on DisCoMBO circuits: log-likelihoods, conditional sampling and conditioning on evidence.

    All functions take and return numpy arrays (NaN marks unobserved/marginalized values) and evaluate the
    spflow circuit in float64 without gradients.
"""
import math
from contextlib import contextmanager
from typing import Dict

import numpy as np
import torch
from spflow.modules.leaves import Bernoulli, Categorical
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.ops import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.utils.cache import Cache
from torch import nn

from .leaves import DTYPE, PROB_FLOOR


class ConditioningError(RuntimeError):
    """Raised if a circuit cannot be conditioned on the given evidence (zero likelihood)."""


def children(module: Module) -> tuple:
    """Child modules of a circuit node (the inputs of a `Cat` are returned as individual children)."""
    if isinstance(module, LeafModule):
        return ()
    inputs = module.inputs
    if isinstance(inputs, Cat):
        inputs = inputs.inputs
    return tuple(inputs) if isinstance(inputs, nn.ModuleList) else (inputs,)


@contextmanager
def _num_threads(n: int):
    """Circuits are evaluated on small tensors, for which a single thread is faster than torch's default."""
    previous = torch.get_num_threads()
    if previous != n:
        torch.set_num_threads(n)
    try:
        yield
    finally:
        if previous != n:
            torch.set_num_threads(previous)


def _discrete_domains(circuit: Module) -> Dict[int, int]:
    """Number of categories per column for columns modeled by Categorical/Bernoulli leaves."""
    domains = {}
    for module in circuit.modules():
        if isinstance(module, Categorical):
            for col in module.scope.query:
                domains[col] = module.K
        elif isinstance(module, Bernoulli):
            for col in module.scope.query:
                domains[col] = 2
    return domains


def _evidence(circuit: Module, data) -> torch.Tensor:
    """
        Float64 copy of `data` in which observed values of Categorical/Bernoulli columns are truncated to
        integers (as legacy SPFlow did) and clipped to the leaf's support (spflow validates its inputs).
    """
    evidence = torch.tensor(np.asarray(data, dtype=float), dtype=DTYPE)
    if evidence.dim() == 1:
        evidence = evidence.reshape(1, -1)
    for col, n_categories in _discrete_domains(circuit).items():
        evidence[:, col] = torch.trunc(evidence[:, col]).clamp(0, n_categories - 1)
    return evidence


def log_likelihood(circuit: Module, data) -> np.ndarray:
    """Log-likelihood of each row of `data` (shape (N, D)), NaNs are marginalized."""
    evidence = _evidence(circuit, data)
    with torch.no_grad(), _num_threads(1):
        ll = circuit.log_likelihood(evidence)
    return ll.reshape(evidence.shape[0]).numpy()


def sample(circuit: Module, evidence, seed: int, n_repeats: int = 1) -> np.ndarray:
    """
        Sample the NaN entries of each row of `evidence` from the circuit conditioned on the row's observed entries.

        Args:
            circuit (Module): Circuit to sample from.
            evidence: Array of shape (N, D); NaN entries are sampled, the others are kept as evidence.
            seed (int): Seed for this call. The global torch RNG state is left untouched.
            n_repeats (int, optional): Sample every evidence row this many times (rows are tiled). Defaults to 1.

        Returns:
            np.ndarray: Array of shape (N * n_repeats, D).
    """
    data = _evidence(circuit, evidence)
    if n_repeats > 1:
        data = data.repeat(n_repeats, 1)
    if data.shape[0] == 0:
        return data.numpy()
    with torch.no_grad(), torch.random.fork_rng(devices=[]), _num_threads(1):
        torch.default_generator.manual_seed(int(seed))
        samples = circuit.sample_with_evidence(data)
    return samples.numpy()


def condition(circuit: Module, evidence_row, legacy_underflow: bool = True) -> Module:
    """
        Circuit of the conditional distribution given the observed entries of a single evidence row
        (legacy `spn.algorithms.Condition.condition`): sum weights are replaced by their posterior given the
        evidence, leaves of observed variables are removed and the structure is pruned like legacy `Prune`.
        Untouched sub-circuits are shared with `circuit`.

        Args:
            circuit (Module): Circuit to condition.
            evidence_row: Array of shape (D,) or (1, D), NaN for unobserved variables.
            legacy_underflow (bool, optional): Raise ConditioningError whenever legacy SPFlow failed, i.e. if
                sum[w_i * exp(ll_i)] underflows to 0 at a sum node. Defaults to True.

        Raises:
            ConditioningError: The evidence has zero (or underflowing) likelihood.
    """
    evidence = _evidence(circuit, evidence_row)
    if evidence.shape[0] != 1:
        raise ValueError("condition expects a single evidence row.")
    observed = {col for col in range(evidence.shape[1]) if not torch.isnan(evidence[0, col])}
    cache = Cache()

    def rec(module):
        scope = set(module.scope.query)
        if not observed & scope:
            return module
        if scope <= observed:
            return None
        if isinstance(module, Product):
            kept = []
            for child in children(module):
                result = rec(child)
                if result is None:
                    continue
                kept.extend(children(result) if isinstance(result, Product) else [result])
            return kept[0] if len(kept) == 1 else Product(inputs=kept)
        if isinstance(module, Sum):
            child_ll = cache['log_likelihood'][module.inputs][0, 0, :, 0]
            prior = module.log_weights[0, :, 0, 0]
            posterior = prior + child_ll
            if not torch.isfinite(torch.logsumexp(posterior, dim=0)) or \
                    (legacy_underflow and torch.sum(prior.exp() * child_ll.exp()) == 0):
                raise ConditioningError("Evidence has zero likelihood under a sum node.")
            posterior = torch.log_softmax(posterior, dim=0)

            new_children, new_log_weights = [], []
            for child, log_weight in zip(children(module), posterior):
                if log_weight == -math.inf:
                    continue
                result = rec(child)
                if isinstance(result, Sum):
                    new_children.extend(children(result))
                    new_log_weights.extend(log_weight + result.log_weights[0, :, 0, 0])
                else:
                    new_children.append(result)
                    new_log_weights.append(log_weight)
            if len(new_children) == 1:
                return new_children[0]
            log_weights = torch.stack(new_log_weights).clamp_min(math.log(PROB_FLOOR))
            return Sum(inputs=new_children, weights=torch.softmax(log_weights, dim=0))
        raise NotImplementedError(f"Cannot condition {type(module).__name__} modules.")

    with torch.no_grad(), _num_threads(1):
        circuit.log_likelihood(evidence, cache=cache)
        conditioned = rec(circuit)
    if conditioned is None:
        raise ConditioningError("Evidence covers the whole scope of the circuit.")
    return conditioned

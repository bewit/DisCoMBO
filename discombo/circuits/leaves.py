# Adapted from SPFlow (legacy branch, https://github.com/SPFlow/SPFlow/tree/legacy):
# spn/structure/leaves/parametric/Parametric.py, spn/structure/leaves/parametric/MLE.py and
# spn/structure/Base.py (@author: Alejandro Molina).
# Copyright 2018 Authors and Contributors (https://github.com/alejandromolinaml/SPFlow/).
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.
# Modified by the DisCoMBO authors: ported to the PyTorch modules of SPFlow >= 1.0.
"""
    Leaf construction for DisCoMBO circuits.

    Leaves are fitted exactly like legacy SPFlow's `learn_parametric` (Gaussian/Categorical/Bernoulli),
    but are built as spflow>=1.0 leaf modules in float64. Zero probabilities (Categorical/Bernoulli),
    which legacy SPFlow produced but spflow turns into NaN/-inf, are floored.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from spflow.meta import Scope
from spflow.modules.leaves import Bernoulli, Categorical, Normal

DTYPE = torch.float64

REAL = 'real'
DISCRETE = 'discrete'
BINARY = 'binary'

# legacy SPFlow replaces a zero Gaussian stdev by this value
STD_FLOOR = 1e-8
# smallest probability representable in log space without spflow producing -inf/NaN
PROB_FLOOR = 1e-300
MAX_LOGIT = 690.0


@dataclass(frozen=True)
class VarSpec:
    """
        Statistical type of one data column.

        kind: one of REAL, DISCRETE, BINARY.
        n_categories: size of the (index-encoded) domain if known, e.g. len(allowed).
    """
    kind: str
    n_categories: Optional[int] = None


def _n_allowed(hp_def: dict) -> Optional[int]:
    allowed = hp_def.get('allowed')
    return len(allowed) if allowed is not None else None


def parametric_specs(search_space_definition: dict, int_real_threshold: int) -> List[VarSpec]:
    """
        Map a search space to leaf types for parametric circuits (performance column appended as REAL):
          float        -> REAL     (Normal)
          int          -> BINARY if 2 values, REAL if >= int_real_threshold values, else DISCRETE
          str/cat      -> DISCRETE (Categorical)
          bool         -> BINARY   (Bernoulli)
    """
    specs = []
    for val in search_space_definition.values():
        dtype = val.get('dtype', 'float')
        n_allowed = _n_allowed(val)
        if dtype == 'float':
            specs.append(VarSpec(REAL))
        elif dtype == 'int':
            n_unique = n_allowed if n_allowed is not None else int(val['max']) - int(val['min']) + 1
            if n_unique == 2:
                specs.append(VarSpec(BINARY, 2))
            elif n_unique >= int_real_threshold:
                specs.append(VarSpec(REAL))
            else:
                specs.append(VarSpec(DISCRETE, n_allowed))
        elif dtype in ('str', 'cat'):
            specs.append(VarSpec(DISCRETE, n_allowed))
        elif dtype == 'bool':
            specs.append(VarSpec(BINARY, 2))
        else:
            specs.append(VarSpec(REAL))
    specs.append(VarSpec(REAL))  # performance column
    return specs


def compute_domains(data: np.ndarray, specs: List[VarSpec]) -> List[np.ndarray]:
    """Global per-column domains, as computed by legacy `Context.add_domains`."""
    domains = []
    for col, spec in enumerate(specs):
        min_val = np.nanmin(data[:, col])
        max_val = np.nanmax(data[:, col])
        if spec.kind == DISCRETE:
            domains.append(np.arange(min_val, max_val + 1, 1))
        else:
            domains.append(np.array([min_val, max_val]))
    return domains


def make_normal(col: int, loc: float, scale: float) -> Normal:
    return Normal(
        scope=Scope([col]),
        loc=torch.full((1, 1, 1), loc, dtype=DTYPE),
        scale=torch.full((1, 1, 1), scale, dtype=DTYPE),
    )


def make_categorical(col: int, probs: np.ndarray) -> Categorical:
    probs = torch.as_tensor(probs, dtype=DTYPE)
    logits = torch.log(probs.clamp_min(PROB_FLOOR)).reshape(1, 1, 1, -1)
    return Categorical(scope=Scope([col]), K=logits.shape[-1], logits=logits)


def make_bernoulli(col: int, p: float) -> Bernoulli:
    with np.errstate(divide='ignore'):
        logit = np.log(p) - np.log1p(-p)
    logit = float(np.clip(logit, -MAX_LOGIT, MAX_LOGIT))
    return Bernoulli(scope=Scope([col]), logits=torch.full((1, 1, 1), logit, dtype=DTYPE))


def fit_parametric_leaf(spec: VarSpec, col: int, x: np.ndarray, domain: np.ndarray):
    """Maximum-likelihood leaf for one column (legacy `create_parametric_leaf` + MLE)."""
    x = x[~np.isnan(x)]
    if spec.kind == REAL:
        if x.size == 0:
            return make_normal(col, 0.0, 1.0)
        std = float(np.std(x))
        if np.isclose(std, 0):
            std = STD_FLOOR
        return make_normal(col, float(np.mean(x)), std)
    if spec.kind == DISCRETE:
        # spflow cannot impute marginalized values for a single category, hence K >= 2
        k = max(int(np.max(domain)) + 1, spec.n_categories or 0, 2)
        counts = np.array([np.sum(x == i) for i in range(k)], dtype=float)
        probs = counts / counts.sum() if counts.sum() > 0 else np.full(k, 1.0 / k)
        return make_categorical(col, probs)
    if spec.kind == BINARY:
        p = float(x.sum() / x.shape[0]) if x.size > 0 else 0.5
        return make_bernoulli(col, p)
    raise ValueError(f"Unknown variable kind: {spec.kind}")

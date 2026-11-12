"""
    Leaf construction for DisCoMBO circuits.

    Leaves are fitted exactly like legacy SPFlow's `learn_parametric` (Gaussian/Categorical/Bernoulli)
    and `learn_mspn` (histograms), but are built as spflow>=1.0 leaf modules in float64.
    Legacy edge cases that spflow does not tolerate are handled here:
        - zero probabilities (Categorical/Bernoulli) are floored, since spflow produces NaN/-inf otherwise,
        - histograms use the local `LegacyHistogram` leaf (raw legacy densities, float64 bin edges and an
          EPSILON floor instead of -inf outside the support).
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from spflow.exceptions import UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules.leaves import Bernoulli, Categorical, Normal
from spflow.modules.leaves.leaf import LeafModule

DTYPE = torch.float64

REAL = 'real'
DISCRETE = 'discrete'
BINARY = 'binary'

# legacy SPFlow replaces a zero Gaussian stdev by this value
STD_FLOOR = 1e-8
# smallest probability representable in log space without spflow producing -inf/NaN
PROB_FLOOR = 1e-300
MAX_LOGIT = 690.0
# legacy SPFlow floors histogram likelihoods at machine epsilon
EPSILON = float(np.finfo(float).eps)


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


def parametric_specs(search_space_definition: dict, int_real_threshold: int, only_real: bool = False) -> List[VarSpec]:
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
        if only_real or dtype == 'float':
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


def metatype_specs(search_space_definition: dict) -> List[VarSpec]:
    """
        Map a search space to leaf types via the `type` (MetaType) entry of each hyperparameter, as legacy
        `learn_mspn` did. The performance column is appended as REAL.
    """
    kinds = {'REAL': REAL, 'DISCRETE': DISCRETE, 'BINARY': BINARY}
    specs = [VarSpec(kinds[val['type'].name], _n_allowed(val)) for val in search_space_definition.values()]
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


def fit_parametric_leaf(spec: VarSpec, col: int, x: np.ndarray, domain: np.ndarray) -> LeafModule:
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


def fit_histogram_leaf(spec: VarSpec, col: int, x: np.ndarray, domain: np.ndarray, alpha: float = 1.0) -> 'LegacyHistogram':
    """
        Histogram leaf for one column (legacy `create_histogram_leaf`, numpy binning, Laplace smoothing).

        Like legacy, the smoothing is applied to densities, so real-valued histograms are not normalized.
        Unlike legacy, a constant global domain does not raise but is widened.
    """
    x = x[~np.isnan(x)]
    if spec.kind == REAL:
        minx, maxx = float(np.min(domain)), float(np.max(domain))
        if np.isclose(maxx, minx):
            minx, maxx = minx - 0.5, maxx + 0.5
        if x.size == 0 or np.var(x) == 0:
            breaks = np.array([minx, maxx])
            densities = np.array([1 / (maxx - minx)])
        else:
            densities, breaks = np.histogram(x, bins='auto', density=True)
    else:
        values = np.unique(domain)
        breaks = np.append(values, values[-1] + 1)
        if x.size == 0:
            densities = np.zeros(len(breaks) - 1)
        else:
            densities, breaks = np.histogram(x, bins=breaks, density=True)

    if alpha > 0:
        n_samples = x.shape[0]
        n_bins = len(breaks) - 1
        counts = densities * n_samples
        densities = (counts + alpha) / (n_samples + n_bins * alpha)

    return LegacyHistogram(Scope([col]), bin_edges=breaks, densities=densities, discrete=spec.kind != REAL)


class LegacyHistogramDist:
    """
        Distribution object of `LegacyHistogram`, implementing the interface spflow's `LeafModule` needs
        (`log_prob`, `sample`, `mode`) with shapes (N, 1, 1, 1).
    """

    def __init__(self, bin_edges: torch.Tensor, densities: torch.Tensor, discrete: bool) -> None:
        self.bin_edges = bin_edges
        self.densities = densities
        self.discrete = discrete

    @property
    def bin_widths(self) -> torch.Tensor:
        return self.bin_edges[1:] - self.bin_edges[:-1]

    @property
    def bin_probs(self) -> torch.Tensor:
        """Normalized bin probabilities used for sampling (legacy: `rv_histogram` / `rand_gen.choice`)."""
        mass = self.densities * self.bin_widths
        return mass / mass.sum()

    def bin_index(self, x: torch.Tensor):
        """Bin index of each value (half-open bins) and whether the value lies inside the support."""
        edges = self.bin_edges.to(x.dtype)
        idx = (torch.bucketize(x, edges, right=True) - 1).clamp(0, len(edges) - 2)
        inside = (x >= edges[0]) & (x < edges[-1])
        return idx, inside

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        idx, inside = self.bin_index(x)
        dens = torch.where(inside, self.densities.to(x.dtype)[idx], torch.zeros_like(x))
        return dens.clamp_min(EPSILON).log()

    def sample(self, sample_shape) -> torch.Tensor:
        n_samples = int(sample_shape[0]) if len(sample_shape) else 1
        idx = torch.distributions.Categorical(probs=self.bin_probs).sample((n_samples,))
        left = self.bin_edges[idx]
        if self.discrete:
            # legacy samples the bin representative points, i.e. the domain values
            samples = left
        else:
            samples = left + torch.rand_like(left) * (self.bin_edges[idx + 1] - left)
        return samples.reshape(n_samples, 1, 1, 1)

    @property
    def mode(self) -> torch.Tensor:
        idx = torch.argmax(self.bin_probs)
        left = self.bin_edges[idx]
        value = left if self.discrete else (left + self.bin_edges[idx + 1]) / 2
        return value.reshape(1, 1, 1)


class LegacyHistogram(LeafModule):
    """
        Univariate histogram leaf reproducing legacy SPFlow histogram semantics:
            - bins are half-open [b_i, b_{i+1}), values outside the support get density EPSILON (not -inf),
            - densities are stored as fitted by legacy (not renormalized),
            - discrete histograms sample domain values (left bin edges), real ones uniformly within the bin.
        Bin edges and densities are float64 buffers (spflow's `Histogram` stores float32 edges).
    """

    def __init__(self, scope: Scope, *, bin_edges, densities, discrete: bool = False) -> None:
        super().__init__(scope=scope, out_channels=1, num_repetitions=1)
        if len(self.scope.query) != 1:
            raise ValueError("LegacyHistogram is univariate.")
        self.register_buffer('bin_edges', torch.as_tensor(np.asarray(bin_edges, dtype=float), dtype=DTYPE))
        self.register_buffer('densities', torch.as_tensor(np.asarray(densities, dtype=float), dtype=DTYPE))
        self.discrete = bool(discrete)

    @property
    def _torch_distribution_class(self):
        return None

    def distribution(self, with_differentiable_sampling: bool = False) -> LegacyHistogramDist:
        if with_differentiable_sampling:
            raise UnsupportedOperationError("LegacyHistogram does not support differentiable sampling.")
        return LegacyHistogramDist(self.bin_edges, self.densities, self.discrete)

    @property
    def _supported_value(self) -> torch.Tensor:
        return self.bin_edges[0]

    def params(self) -> dict:
        return {'bin_edges': self.bin_edges, 'densities': self.densities}

    def _compute_parameter_estimates(self, data, weights, bias_correction):
        raise UnsupportedOperationError("LegacyHistogram is fitted by fit_histogram_leaf.")

    def marginalize(self, marg_rvs, prune=True, cache=None):
        return None if any(rv in marg_rvs for rv in self.scope.query) else self

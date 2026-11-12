import itertools

import numpy as np
import pytest
import torch
from scipy import stats
from spflow.meta import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.products import Product
from spflow.modules.sums import Sum

from circuits import (BINARY, DISCRETE, HISTOGRAM, PARAMETRIC, REAL, ConditioningError, LegacyHistogram, VarSpec,
                      children, condition, fishers_method, learn_circuit, log_likelihood, mixture, pvalues, sample)
from circuits.leaves import EPSILON, make_bernoulli, make_categorical, make_normal


def mixed_data(n=200, seed=0):
    """Two clusters over (real, categorical, binary, perf); the perf column identifies the cluster."""
    rng = np.random.RandomState(seed)
    z = rng.rand(n) < 0.5
    x0 = np.where(z, rng.normal(3, 0.5, n), rng.normal(-3, 0.5, n))
    x1 = np.where(z, rng.choice(3, n, p=[0.1, 0.1, 0.8]), rng.choice(3, n, p=[0.8, 0.1, 0.1]))
    x2 = (rng.rand(n) < np.where(z, 0.9, 0.2)).astype(float)
    perf = np.where(z, rng.normal(1.0, 0.1, n), rng.normal(0.0, 0.1, n))
    specs = [VarSpec(REAL), VarSpec(DISCRETE, 3), VarSpec(BINARY, 2), VarSpec(REAL)]
    return np.column_stack([x0, x1, x2, perf]), specs


def hand_built_circuit():
    """Sum(0.3, 0.7) over two products of Normal(x0), Categorical(x1), Normal(x2)."""
    p1 = Product(inputs=[make_normal(0, 0.0, 1.0), make_categorical(1, np.array([0.2, 0.8])), make_normal(2, 1.0, 2.0)])
    p2 = Product(inputs=[make_normal(0, 3.0, 0.5), make_categorical(1, np.array([0.6, 0.4])), make_normal(2, -1.0, 1.0)])
    return Sum(inputs=[p1, p2], weights=torch.tensor([0.3, 0.7], dtype=torch.float64))


def legacy_pvalues(x, corrected=False):
    """Reference p-values of `hand_built_circuit` computed like legacy `pvalue_spflow`."""
    def gauss(col, mean, std):
        cdf = stats.norm.cdf(col, mean, std)
        return np.where(np.isnan(col), 1.0, 2 * np.minimum(cdf, 1 - cdf))

    def cat(col, p):
        return np.where(np.isnan(col), 1.0, np.asarray(p)[np.nan_to_num(col).astype(int)])

    def fisher(*ps):
        ps = np.clip(np.column_stack(ps), 1e-300, 1)
        return stats.chi2.sf(-2 * np.log(ps).sum(1), 2 * ps.shape[1])

    c1 = fisher(gauss(x[:, 0], 0, 1), cat(x[:, 1], [0.2, 0.8]), gauss(x[:, 2], 1, 2))
    c2 = fisher(gauss(x[:, 0], 3, 0.5), cat(x[:, 1], [0.6, 0.4]), gauss(x[:, 2], -1, 1))
    correction = min(2.0, 1 / 0.7) if corrected else 1.0
    return correction * (0.3 * c1 + 0.7 * c2)


def assert_pruned(module):
    """No sum directly below a sum and no product directly below a product (legacy Prune invariant)."""
    for child in children(module):
        for grandchild in children(child) if type(child).__name__ == 'Cat' else [child]:
            assert not (isinstance(module, Product) and isinstance(grandchild, Product))
            assert not (isinstance(module, Sum) and isinstance(grandchild, Sum))
            assert_pruned(grandchild)


# --- Fisher's method ---

def test_fishers_method_single_pvalue_is_identity():
    p = torch.tensor([0.01, 0.3, 0.99], dtype=torch.float64).reshape(-1, 1, 1)
    assert torch.allclose(fishers_method(p).flatten(), p.flatten())


def test_fishers_method_matches_scipy():
    rng = np.random.RandomState(1)
    for k in (2, 3, 7, 20):
        p = rng.rand(50, k) ** 3
        ours = fishers_method(torch.tensor(p).unsqueeze(-1)).flatten().numpy()
        ref = np.array([stats.combine_pvalues(row, method='fisher')[1] for row in p])
        assert np.allclose(ours, ref, rtol=1e-10, atol=1e-300)


def test_fishers_method_counts_marginalized_children():
    p = torch.tensor([[0.05, 1.0, 1.0]], dtype=torch.float64).unsqueeze(-1)
    expected = stats.chi2.sf(-2 * np.log(0.05), 6)
    assert np.isclose(fishers_method(p).item(), expected)
    assert np.isclose(fishers_method(torch.ones(1, 4, 1, dtype=torch.float64)).item(), 1.0)


# --- p-values ---

@pytest.mark.parametrize('corrected', [False, True])
def test_pvalues_match_legacy_formulas(corrected):
    rng = np.random.RandomState(2)
    x = np.column_stack([rng.normal(1, 2, 100), rng.randint(0, 2, 100), rng.normal(0, 2, 100)])
    x[::7, 2] = np.nan  # marginalized column still counts in Fisher's method
    x[::11, 1] = np.nan
    ours = pvalues(hand_built_circuit(), x, corrected=corrected)
    assert np.allclose(ours, np.clip(legacy_pvalues(x, corrected), 0, 1), rtol=1e-10, atol=1e-14)


def test_leaf_pvalues():
    nan = np.nan
    cat = make_categorical(0, np.array([0.5, 0.0, 0.5]))
    assert np.allclose(pvalues(cat, [[0], [1], [2.7], [3], [-1], [nan]]), [0.5, 1e-300, 0.5, 0, 0, 1])
    bern = make_bernoulli(0, 0.25)
    assert np.allclose(pvalues(bern, [[1], [0], [nan]]), [0.25, 0.75, 1])
    normal = make_normal(0, 0.0, 1.0)
    z = np.array([[-1.0], [0.0], [2.0]])
    assert np.allclose(pvalues(normal, z), 2 * stats.norm.sf(np.abs(z.flatten())))
    assert np.allclose(pvalues(normal, z, side='left'), stats.norm.cdf(z.flatten()))
    assert np.allclose(pvalues(normal, z, side='right'), stats.norm.sf(z.flatten()))

    hist = LegacyHistogram(Scope([0]), bin_edges=[0.0, 1.0, 3.0], densities=[0.2, 0.4])
    # cdf(0.5) = 0.1, cdf(2) = 0.2 + 0.4 = 0.6
    assert np.allclose(pvalues(hist, [[0.5], [2.0], [-1.0], [5.0]]), [0.2, 0.8, 0.0, 0.0])
    disc = LegacyHistogram(Scope([0]), bin_edges=[1.0, 2.0, 3.0, 4.0], densities=[0.5, 0.3, 0.2], discrete=True)
    assert np.allclose(pvalues(disc, [[1], [3], [0], [4], [nan]]), [0.5, 0.2, 0, 0, 1])


# --- leaves ---

def test_legacy_histogram_likelihood_and_sampling():
    hist = LegacyHistogram(Scope([0]), bin_edges=[0.0, 0.5, 1.0 + 1e-12], densities=[0.6, 1.2])
    assert hist.bin_edges.dtype == torch.float64 and hist.bin_edges[-1].item() == 1.0 + 1e-12
    ll = log_likelihood(hist, [[0.25], [0.75], [1.0 + 1e-12], [-3.0], [np.nan]])
    assert np.allclose(ll, np.log([0.6, 1.2, EPSILON, EPSILON, 1.0]))

    samples = sample(hist, np.full((5000, 1), np.nan), seed=0)
    # sampling probabilities are proportional to density * width: 0.3 / (0.3 + 0.6)
    assert np.isclose(np.mean(samples < 0.5), 1 / 3, atol=0.03)

    disc = LegacyHistogram(Scope([0]), bin_edges=[2.0, 3.0, 4.0], densities=[0.25, 0.75], discrete=True)
    samples = sample(disc, np.full((2000, 1), np.nan), seed=0)
    assert set(np.unique(samples)) <= {2.0, 3.0} and np.isclose(np.mean(samples == 3.0), 0.75, atol=0.04)


def test_degenerate_parametric_leaves_have_finite_likelihoods():
    bern = make_bernoulli(0, 0.0)
    cat = make_categorical(0, np.array([1.0, 0.0]))
    for leaf, x in ((bern, [[0.0], [1.0]]), (cat, [[0.0], [1.0]])):
        assert np.all(np.isfinite(log_likelihood(leaf, x)))


# --- structure learning ---

@pytest.mark.parametrize('family', [PARAMETRIC, HISTOGRAM])
def test_learn_circuit_mixed_data(family):
    data, specs = mixed_data()
    circuit = learn_circuit(data, specs, family=family, min_instances_slice=10, threshold=0.3)
    assert set(circuit.scope.query) == set(range(data.shape[1]))
    assert np.all(np.isfinite(log_likelihood(circuit, data)))
    assert all(len(leaf.scope.query) == 1 for leaf in circuit.modules() if isinstance(leaf, LeafModule))
    assert_pruned(circuit)


@pytest.mark.parametrize('family', [PARAMETRIC, HISTOGRAM])
@pytest.mark.parametrize('n_rows', [1, 2, 3, 30])
def test_learn_circuit_degenerate_data(family, n_rows):
    rng = np.random.RandomState(n_rows)
    data = np.column_stack([
        np.full(n_rows, 1.5),                    # constant real column
        np.zeros(n_rows),                        # constant category
        np.ones(n_rows),                         # constant binary column
        np.repeat(rng.normal(size=1), n_rows),   # duplicate rows
        rng.normal(size=n_rows),
    ])
    specs = [VarSpec(REAL), VarSpec(DISCRETE, 1), VarSpec(BINARY, 2), VarSpec(REAL), VarSpec(REAL)]
    circuit = learn_circuit(data, specs, family=family, min_instances_slice=2)
    assert set(circuit.scope.query) == set(range(data.shape[1]))
    assert np.all(np.isfinite(log_likelihood(circuit, data)))
    assert np.all(np.isfinite(pvalues(circuit, data)))


def test_discrete_circuit_is_normalized():
    rng = np.random.RandomState(3)
    a = rng.randint(0, 3, 300)
    b = (a + rng.randint(0, 2, 300)) % 4
    c = (rng.rand(300) < 0.3).astype(float)
    circuit = learn_circuit(np.column_stack([a, b, c]), [VarSpec(DISCRETE, 3), VarSpec(DISCRETE, 4), VarSpec(BINARY, 2)],
                            min_instances_slice=20)
    grid = np.array(list(itertools.product(range(3), range(4), range(2))), dtype=float)
    assert np.isclose(np.exp(log_likelihood(circuit, grid)).sum(), 1.0)


def test_learn_circuit_is_deterministic():
    data, specs = mixed_data()
    c1 = learn_circuit(data, specs, min_instances_slice=10)
    c2 = learn_circuit(data, specs, min_instances_slice=10)
    assert np.allclose(log_likelihood(c1, data), log_likelihood(c2, data))


# --- sampling and conditioning ---

def test_sample_respects_evidence_and_is_reproducible():
    data, specs = mixed_data()
    circuit = learn_circuit(data, specs, min_instances_slice=10)
    evidence = np.full((500, 4), np.nan)
    evidence[:, -1] = 1.0
    evidence_copy = evidence.copy()
    torch_state = torch.get_rng_state()

    s1 = sample(circuit, evidence, seed=3)
    s2 = sample(circuit, evidence, seed=3)
    assert np.array_equal(s1, s2) and not np.isnan(s1).any()
    assert np.array_equal(evidence, evidence_copy, equal_nan=True)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.all(s1[:, -1] == 1.0)
    # high performance identifies the cluster with x0 ~ N(3, 0.5)
    assert np.isclose(s1[:, 0].mean(), 3.0, atol=0.2)

    tiled = sample(circuit, evidence[:2], seed=0, n_repeats=3)
    assert tiled.shape == (6, 4)


def test_condition_matches_likelihood_ratio():
    data, specs = mixed_data()
    circuit = learn_circuit(data, specs, min_instances_slice=10)
    perf = 0.8
    conditioned = condition(circuit, [np.nan, np.nan, np.nan, perf])
    assert set(conditioned.scope.query) == {0, 1, 2}
    assert_pruned(conditioned)

    query = data[:50].copy()
    query[:, -1] = perf
    marginal = np.array([[np.nan, np.nan, np.nan, perf]])
    expected = log_likelihood(circuit, query) - log_likelihood(circuit, marginal)
    query[:, -1] = np.nan
    assert np.allclose(log_likelihood(conditioned, query), expected, atol=1e-10)


def test_condition_underflow():
    data, specs = mixed_data()
    circuit = learn_circuit(data, specs, min_instances_slice=10)
    evidence = [np.nan, np.nan, np.nan, 1e4]
    with pytest.raises(ConditioningError):
        condition(circuit, evidence)
    conditioned = condition(circuit, evidence, legacy_underflow=False)
    assert set(conditioned.scope.query) == {0, 1, 2}


def test_mixture():
    data, specs = mixed_data()
    c1 = learn_circuit(data[:100], specs, min_instances_slice=10)
    c2 = learn_circuit(data[100:], specs, min_instances_slice=10)
    mix = mixture([c1, c2], [1, 3])
    expected = np.logaddexp(np.log(0.25) + log_likelihood(c1, data), np.log(0.75) + log_likelihood(c2, data))
    assert np.allclose(log_likelihood(mix, data), expected)

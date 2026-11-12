import logging

import numpy as np
import pytest
import torch

from discombo import BenchQueryResult, MCOptimizer, SimpleSearchSpace

DEFINITION = {
    'lr': {'dtype': 'float', 'min': 1e-4, 'max': 1e-1, 'is_log': True},
    'momentum': {'dtype': 'float', 'min': 0.0, 'max': 1.0},
    'layers': {'dtype': 'int', 'allowed': [1, 2, 3, 4]},
    'optimizer': {'dtype': 'str', 'allowed': ['sgd', 'adam', 'rmsprop']},
    'nesterov': {'dtype': 'bool', 'allowed': [False, True]},
}


def objective(cfg):
    score = (-(np.log10(cfg['lr']) + 2) ** 2 - (cfg['momentum'] - 0.9) ** 2 - 0.1 * abs(cfg['layers'] - 3)
             + 0.2 * (cfg['optimizer'] == 'adam') + 0.05 * cfg['nesterov'])
    return BenchQueryResult(score, score, score)


def make_optimizer(seed=0, **kwargs):
    return MCOptimizer(SimpleSearchSpace(DEFINITION), objective, iterations=8, num_samples=5, initial_samples=10,
                       exploration_method='stratified', stratified_pool_size=200, seed=seed, **kwargs)


def assert_valid(cfg):
    assert set(cfg) == set(DEFINITION)
    for name, hp_def in DEFINITION.items():
        if hp_def['dtype'] == 'float':
            assert hp_def['min'] <= cfg[name] <= hp_def['max'] * (1 + 1e-9)
        else:
            assert cfg[name] in hp_def['allowed']


def test_simple_search_space_samples_valid_configs():
    space = SimpleSearchSpace(DEFINITION)
    for cfg in space.sample(size=50):
        assert_valid(cfg)
    for cfg in space.sample(size=10, keep_domain=False):
        assert cfg['optimizer'] in range(3)
    with pytest.raises(ValueError):
        SimpleSearchSpace({'x': {'dtype': 'int'}})


@pytest.mark.parametrize('normalize_inputs', [False, True])
def test_mc_optimizer_runs(normalize_inputs):
    opt = make_optimizer(normalize_inputs=normalize_inputs)
    cfg, res = opt.optimize()
    assert len(opt.history) == 10 + 8 * 5
    for evaluated_cfg, _, _ in opt.history:
        assert_valid(evaluated_cfg)
    assert res.val_performance == max(r.val_performance for _, r, _ in opt.history)
    assert_valid(cfg)


def test_mc_optimizer_interventions():
    opt = make_optimizer(intervention_pval_corrected=True)
    opt.intervene([{'momentum': {'dist': 'gauss', 'parameters': [0.9, 0.05]}}, {'optimizer': 1}], [2, 5])
    opt.optimize()
    log = opt.get_intervention_pval_log()
    assert [rec['iter'] for rec in log] == [2, 5]
    assert all(0.0 <= rec['pval_mean'] <= 1.0 for rec in log)


def run_history(seed):
    opt = make_optimizer(seed=seed)
    opt.intervene([{'momentum': {'dist': 'gauss', 'parameters': [0.9, 0.05]}}, {'optimizer': 1}], [2, 5])
    opt.optimize()
    return [(cfg, res.val_performance, it) for cfg, res, it in opt.history], opt.get_intervention_pval_log()


def test_seed_makes_runs_reproducible_without_touching_global_rngs():
    np.random.seed(123)
    torch.manual_seed(123)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    first = run_history(seed=7)
    # the global random states are neither used nor changed by DisCoMBO
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    # perturbing the global random states does not change the run
    np.random.seed(999)
    torch.manual_seed(999)
    assert run_history(seed=7) == first
    assert run_history(seed=8)[0] != first[0]


def test_progress_is_logged_not_printed(capsys, caplog):
    with caplog.at_level(logging.INFO, logger='discombo'):
        make_optimizer().optimize()
    assert capsys.readouterr().out == ''
    assert any('best observed value' in rec.getMessage() for rec in caplog.records)

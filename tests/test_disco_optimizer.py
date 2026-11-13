import logging

import numpy as np
import pytest
import torch

from discombo import DisCoOptimizer, EvaluationResult, SimpleSearchSpace

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
    return EvaluationResult(score, score, score)


def make_optimizer(seed=0, **kwargs):
    return DisCoOptimizer(SimpleSearchSpace(DEFINITION), objective, iterations=8, num_samples=5, initial_samples=10,
                          stratified_pool_size=200, seed=seed, **kwargs)


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
def test_disco_optimizer_runs(normalize_inputs):
    opt = make_optimizer(normalize_inputs=normalize_inputs)
    cfg, res = opt.optimize()
    assert len(opt.history) == 10 + 8 * 5
    for evaluated_cfg, _, _ in opt.history:
        assert_valid(evaluated_cfg)
    assert res.val_performance == max(r.val_performance for _, r, _ in opt.history)
    assert_valid(cfg)


def test_disco_optimizer_interventions():
    opt = make_optimizer(disco_score_corrected=True)
    opt.intervene([{'momentum': {'dist': 'gauss', 'parameters': [0.9, 0.05]}}, {'optimizer': 1}], [2, 5])
    opt.optimize()
    records = opt.get_disco_scores()
    assert [rec['iter'] for rec in records] == [2, 5]
    assert [rec['intervention_idx'] for rec in records] == [0, 1]
    assert all(0.0 <= rec['disco_score'] <= 1.0 for rec in records)
    assert all(len(rec['sample_disco_scores']) == 5 for rec in records)


def run_history(seed):
    opt = make_optimizer(seed=seed)
    opt.intervene([{'momentum': {'dist': 'gauss', 'parameters': [0.9, 0.05]}}, {'optimizer': 1}], [2, 5])
    opt.optimize()
    return [(cfg, res.val_performance, it) for cfg, res, it in opt.history], opt.get_disco_scores()


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


def test_ask_tell_loop_equals_optimize():
    reference = make_optimizer(seed=3)
    reference.intervene([{'momentum': {'dist': 'gauss', 'parameters': [0.9, 0.05]}}], [2])
    reference.optimize()

    opt = make_optimizer(seed=3)
    opt.intervene([{'momentum': {'dist': 'gauss', 'parameters': [0.9, 0.05]}}], [2])
    configs = opt.ask()
    assert len(configs) == 10 and opt.iteration == 0  # initial random design
    opt.tell(configs, [objective(cfg) for cfg in configs])
    while opt.iteration < 8:
        configs = opt.ask()
        assert len(configs) == 5
        for cfg in configs:  # results may also be reported one by one, as plain numbers
            opt.tell([cfg], [objective(cfg).val_performance])
    as_tuples = [[(cfg, r.val_performance, it) for cfg, r, it in o.history] for o in (reference, opt)]
    assert as_tuples[0] == as_tuples[1]
    assert opt.get_disco_scores() == reference.get_disco_scores()
    assert opt.best()[1].val_performance == reference.best()[1].val_performance


def test_tell_without_ask_warm_starts_and_ignores_failures():
    opt = make_optimizer()
    observed = SimpleSearchSpace(DEFINITION).sample(size=12, rng=1)
    values = [objective(cfg).val_performance for cfg in observed]
    opt.tell(observed + [observed[0]], values + [None])
    assert len(opt.history) == 12
    configs = opt.ask()  # no initial random design: the first ask is already a DisCoMBO iteration
    assert len(configs) == 5 and opt.iteration == 1
    for cfg in configs:
        assert_valid(cfg)
    assert opt.best()[1].val_performance == max(values)
    with pytest.raises(ValueError):
        opt.tell(configs, [1.0])


def test_deprecated_names_still_work():
    import pickle
    import discombo
    import discombo.optimizers
    import discombo.result
    deprecated = pytest.warns(DeprecationWarning, match='removed in DisCoMBO 2.0')

    # classes
    with deprecated:
        from discombo import MCOptimizer
    assert MCOptimizer is DisCoOptimizer
    with deprecated:
        assert discombo.optimizers.MCOptimizer is DisCoOptimizer
    with deprecated:
        from discombo.optimizers.mc_optimizer import MCOptimizer as from_old_module
    assert from_old_module is DisCoOptimizer
    with deprecated:
        assert discombo.BenchQueryResult is EvaluationResult
    with deprecated:
        assert discombo.result.BenchQueryResult is EvaluationResult
    with pytest.raises(AttributeError):
        discombo.NoSuchName
    with pytest.raises(AttributeError):
        discombo.optimizers.BenchQueryResult  # not in this module, neither under its new name

    # deprecated constructor arguments, also positionally
    old_arguments = {'interaction_dist_sample_decay': 0.5, 'mc_min_instances': 3, 'mc_rdc_threshold': 0.2,
                     'mc_k_clusters': 3, 'log_hpo_runtime': True, 'intervention_pval_corrected': True}
    with deprecated as warned:
        opt = make_optimizer(**old_arguments)
    assert {str(w.message).split()[2] for w in warned} == set(old_arguments)  # "The argument <name> is deprecated..."
    assert (opt._intervention_decay, opt._circuit_min_instances, opt._circuit_rdc_threshold, opt._circuit_k_clusters,
            opt._log_runtime, opt._disco_score_corrected) == tuple(old_arguments.values())
    with pytest.raises(TypeError, match='both'):
        make_optimizer(mc_k_clusters=3, circuit_k_clusters=3)
    with pytest.raises(TypeError):
        make_optimizer(mc_no_such_argument=1)
    with pytest.warns(DeprecationWarning, match='exploration_method'):
        positional = DisCoOptimizer(SimpleSearchSpace(DEFINITION), objective, 8, 5, 10, 'stratified', 200, 7)
    assert positional._stratified_bins == 7
    with deprecated:
        make_optimizer(mc_type='mixed')
    with deprecated, pytest.raises(ValueError):
        make_optimizer(exploration_method='rejection')

    # methods and attributes
    opt.intervene([{'optimizer': 1}], [2])
    opt.optimize()
    with pytest.warns(DeprecationWarning, match=r'use get_disco_scores\(\)'):
        old_records = opt.get_intervention_pval_log()
    assert old_records == [{'iter': r['iter'], 'intervention_idx': r['intervention_idx'], 'pval_mean': r['disco_score'],
                            'pval_vec': r['sample_disco_scores']} for r in opt.get_disco_scores()]
    with pytest.warns(DeprecationWarning, match='use circuit'):
        assert opt.curr_mc is opt.circuit
    with pytest.warns(DeprecationWarning, match='use runtimes'):
        assert opt.hpo_runtimes is opt.runtimes and len(opt.runtimes) == 8
    with deprecated:
        opt.set_search_space(opt.search_space)
    with deprecated, pytest.raises(NotImplementedError):
        opt.adapt_to_search_space({})

    # pickled optimizers with deprecated attributes are migrated
    state = pickle.loads(pickle.dumps(make_optimizer())).__dict__
    old_state = {'curr_mc': None, 'hpo_runtimes': [0.1], '_mc_type': 'mixed', '_exploration_method': 'stratified',
                 '_intervention_pval_log': [{'iter': 2, 'intervention_idx': 0, 'pval_mean': 0.5, 'pval_vec': [0.5]}]}
    restored = DisCoOptimizer.__new__(DisCoOptimizer)
    restored.__setstate__({**{k: v for k, v in state.items() if k not in ('circuit', 'runtimes', '_disco_scores')},
                           **old_state})
    assert set(restored.__dict__) == set(state)
    assert restored.runtimes == [0.1]
    assert restored.get_disco_scores() == [{'iter': 2, 'intervention_idx': 0, 'disco_score': 0.5,
                                            'sample_disco_scores': [0.5]}]

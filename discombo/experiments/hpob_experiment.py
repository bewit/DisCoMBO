
from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
from copy import deepcopy
from spn.structure.StatisticalTypes import MetaType
import os
import numpy as np

class HPOBExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, search_space_id, dataset_id, seed=0) -> None:
        self.benchmark_name = 'hpob'
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id
        self.seed = seed
        benchmark_cfg = {
            'search_space_id': self.search_space_id,
            'dataset_id': self.dataset_id,
            'surrogate_path': './benchmark_data/hpob-surrogates/',
            'dataset_path': './benchmark_data/hpob-data/'
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        benchmark_cfg)

        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config) 
        super().__init__(benchmark, optimizer)
        super().__init__(benchmark, optimizer, seed)

    def run(self):
        # perform optimization on second search space
        config, performance = self.optimizer.optimize()
        print((config, performance))


    def evaluate_config(self, cfg, budget=None):
        test_cfg = cfg
        test_cfg['booster'] = 'INVALID'
        if self.benchmark is not None:
            res = self.benchmark.query(test_cfg, budget)
            return res

        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 260,
                'log_hpo_runtime': False,
                'seed': self.seed
            }
        elif self._optimizer_name == 'pc':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 25,
                'num_samples': 10,
                'num_self_consistency_samplings': 10,
                'initial_samples': 10,
                'use_ei': False,
                'num_ei_repeats': 20,
                'pc_type': 'mixed',
                'log_hpo_runtime': True,
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 10000,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 200
            }
        elif self._optimizer_name == 'optunabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 260
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 260,
                'surrogate': 'rf'
            }
        elif self._optimizer_name == 'mc_stratified':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 25,
                'num_samples': 10,
                'initial_samples': 10,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_cycle_length': 20,
                'stratified_gamma_start': 3.0,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'cmaes':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 25,
                'num_samples': 10,
                'initial_samples': 10,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'hebo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 25,
                'num_samples': 10,
                'initial_samples': 10,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'dynabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 260,
                'seed': self.seed,
                'initial_design_size': 10,
                'prior_std_denominator': 5,
                'prior_decay': 'quadratic',
                'prior_decay_enumerator': 20,
                'prior_decay_denominator': 1,
                'remove_old_priors': False,
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
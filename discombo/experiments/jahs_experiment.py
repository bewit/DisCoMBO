from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
from copy import deepcopy
import os
import json

class JAHSBenchExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='cifar10', seed=0) -> None:
        self.benchmark_name = 'jahs'
        self.benchmark_config = {
            'task': task
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)

        self._optimizer_name = optimizer_name
        self.seed = seed
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        super().__init__(benchmark, optimizer)     

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        test_cfg = deepcopy(cfg)
        if self.benchmark is not None:
            test_cfg['Optimizer'] = 'SGD'
            if budget is None and 'epoch' in cfg:
                budget = cfg['epoch']
                test_cfg.pop('epoch', None)
            res = self.benchmark.query(test_cfg, budget)
            return res
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 1220,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 1220
            }
        elif self._optimizer_name == 'bopro':
            self.setup_bopro_json(benchmark)
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'exp_json': self._bopro_file_name
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'min_budget': 1,
                'max_budget': 200,
                'n_trials': 2000
            }
        elif self._optimizer_name == 'pc':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 60,
                'num_samples': 20,
                'use_ei': False,
                'num_ei_repeats': 20,
                'log_hpo_runtime': True,
                #'conditioning_value_quantile': 0.25
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 2000,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 150
            }
        elif self._optimizer_name == 'optunabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 1220
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 1220,
                'surrogate': 'rf'
            }
        elif self._optimizer_name == 'mc_stratified':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 60,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_cycle_length': 50,
                'stratified_gamma_start': 3.0,
                'log_hpo_runtime': False,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_c2':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_gamma_start': 2.0,
                'stratified_gamma_end': 0.25,
                'stratified_cycle_length': 2,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_c20':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_gamma_start': 2.0,
                'stratified_gamma_end': 0.25,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_cmax':
            # cycle_length = max_iterations = 50: one half-period, explore→exploit monotonically
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_gamma_start': 2.0,
                'stratified_gamma_end': 0.25,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_uniform':
            # Uniform bins: gamma=1.0 always → equal bin widths, no exploration/exploitation bias
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_gamma_start': 1.0,
                'stratified_gamma_end': 1.0,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_p100':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 100,
                'stratified_bins': 10,
                'stratified_gamma_start': 2.0,
                'stratified_gamma_end': 0.25,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_p10000':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 10000,
                'stratified_bins': 10,
                'stratified_gamma_start': 2.0,
                'stratified_gamma_end': 0.25,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_gamma_weak':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_gamma_start': 1.5,
                'stratified_gamma_end': 0.5,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_gamma_strong':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_gamma_start': 4.0,
                'stratified_gamma_end': 0.1,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.05,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'mc_stratified_annealed_sine_noball':
            # No ball perturbation: tests contribution of local refinement
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': 20,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_gamma_start': 2.0,
                'stratified_gamma_end': 0.25,
                'stratified_cycle_length': 50,
                'max_rel_ball_size': 0.0,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'cmaes':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 60,
                'num_samples': 20,
                'initial_samples': 20,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'hebo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 60,
                'num_samples': 20,
                'initial_samples': 20,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'dynabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 1220,
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

    def setup_bopro_json(self, benchmark):
        search_space = benchmark.search_space
        borpo_search_space = search_space.to_hypermapper()
        json_dict = {
            "application_name": "JAHS Interactive",
            "optimization_objectives": ["value"],
            "design_of_experiment": {
                "number_of_samples": 3,
            },
            "optimization_iterations": 2000,
            "optimization_method": "prior_guided_optimization",
            "number_of_cpus": 16,
            "models": {
                "model": "random_forest"
            },
            "input_parameters": borpo_search_space,
            "local_search_starting_points": 10,
            "local_search_random_points": 50,
            "local_search_evaluation_limit": 200
        }
        if not os.path.exists('./bopro_experiments/'):
            os.mkdir('./bopro_experiments')
        self._bopro_file_name =  './bopro_experiments/jahs_interactive.json'
        with open(self._bopro_file_name, 'w+') as f:
            json.dump(json_dict, f)

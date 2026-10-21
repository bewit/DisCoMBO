from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory

class NASBench201Experiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='cifar10', seed=0) -> None:
        self.benchmark_name = 'nas201'
        self.benchmark_config = {
            'task': task
        }
        self.seed = seed
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)

        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)  
        super().__init__(benchmark, optimizer)

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        test_cfg = cfg
        for key, val in cfg.items():
            cfg[key] = int(val)
        if self.benchmark is not None:
            res = self.benchmark.query(test_cfg, budget)
            return res
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 1220,
                'seed': self.seed
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'min_budget': 1,
                'max_budget': 200,
                'n_trials': 500
            }
        elif self._optimizer_name == 'pc':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 60,
                'num_samples': 20,
                'use_ei': False,
                'num_ei_repeats': 20,
                'pc_type': 'mixed',
                'log_hpo_runtime': True,
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 2000
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

from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import warnings


class PVExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='AgNP', seed=0) -> None:
        self.benchmark_name = 'pv'
        self.benchmark_config = {
            'task': task,
            'save_dir': './benchmark_data/pv/'
        }
        benchmark = BenchmarkFactory.get_benchmark(
            self.benchmark_name,
            self.benchmark_config,
        )

        self._optimizer_name = optimizer_name
        self.seed = seed
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        super().__init__(benchmark, optimizer)

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        if budget is not None:
            warnings.warn('PV benchmark does not support multiple fidelities.')
        return self.benchmark.query(cfg, budget)

    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 510,
                'log_hpo_runtime': False,
                'seed': self.seed,
            }
        elif self._optimizer_name == 'pc':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'num_samples': 5,
                'initial_samples': 10,
                'num_self_consistency_samplings': 20,
                'use_ei': False,
                'num_ei_repeats': 20,
                'pc_type': 'mspn',
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 100,
            }
        elif self._optimizer_name == 'optunabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 510,
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 510,
                'surrogate': 'rf',
            }
        elif self._optimizer_name == 'dynabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 510,
                'seed': self.seed,
                'initial_design_size': 10,
                'prior_std_denominator': 5,
                'prior_decay': 'quadratic',
                'prior_decay_enumerator': 20,
                'prior_decay_denominator': 1,
                'remove_old_priors': False,
            }
        elif self._optimizer_name == 'mc_stratified':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'num_samples': 5,
                'initial_samples': 10,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 5,
                'stratified_cycle_length': 10,
                'stratified_gamma_start': 3.0,
                'stratified_gamma_end': 0.05,
                'max_rel_ball_size_explore': 0.2,
                'max_rel_ball_size_exploit': 0.02,
                'normalize_inputs': True,
                'log_hpo_runtime': False,
                'seed': self.seed
            }
        elif self._optimizer_name == 'cmaes':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'num_samples': 5,
                'initial_samples': 10,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        elif self._optimizer_name == 'hebo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'num_samples': 5,
                'initial_samples': 10,
                'log_hpo_runtime': True,
                'seed': self.seed
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')

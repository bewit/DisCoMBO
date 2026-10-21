from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import os
import json
import warnings
from scipy.special import softmax
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, NormalFloatHyperparameter, UniformIntegerHyperparameter

# See note in jahs_interactive_experiment.py: align DynaBO's eval budget with MC.
MC_INITIAL_SAMPLES = 10
MC_NUM_SAMPLES = 5


class FlorisInteractiveExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, interaction_idx=0, seed=0,
                 iterations=None, abort_after_intervention=False,
                 interventions_path='./interventions/floris_balanced.json',
                 intervention_pval_corrected=False) -> None:
        self.benchmark_name = 'floris'
        self.benchmark_config = {
            'save_dir': './benchmark_data/floris/default/'
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, self.benchmark_config)

        self._optimizer_name = optimizer_name
        self._seed = seed
        self._interaction_idx = interaction_idx
        self._iterations_override = iterations
        self._abort_after_intervention = abort_after_intervention
        self._interventions_path = interventions_path
        self._intervention_pval_corrected = intervention_pval_corrected
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        super().__init__(benchmark, optimizer)

    def run(self):
        if self._optimizer_name in ('pc', 'rs', 'mc_stratified'):
            interventions, iterations = self.get_pc_interventions()
            self.optimizer.intervene(interventions, iterations)
        elif self._optimizer_name == 'bopro':
            intervention = self.get_bopro_intervention()
            self.optimizer.intervene(intervention)
        elif self._optimizer_name == 'pibo':
            intervention = self.get_pibo_intervention()
            self.optimizer.intervene(intervention)
        else:
            interventions, iterations = self.get_dynabo_interventions()
            self.optimizer.intervene(interventions, iterations)
        config, performance = self.optimizer.optimize()
        print((config, performance))
        return config, performance

    def evaluate_config(self, cfg, budget=None):
        if budget is not None:
            warnings.warn('Floris benchmark does not support multiple fidelities.')
        return self.benchmark.query(cfg, budget)

    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 250,
                'log_hpo_runtime': False,
                'seed': self._seed,
            }
        elif self._optimizer_name == 'pc':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 50,
                'num_samples': MC_NUM_SAMPLES,
                'initial_samples': MC_INITIAL_SAMPLES,
                'num_self_consistency_samplings': 10,
                'use_ei': False,
                'num_ei_repeats': 20,
                'pc_type': 'mixed',
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 250,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 250,
            }
        elif self._optimizer_name == 'mc_stratified':
            iters = self._iterations_override if self._iterations_override is not None else 50
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': iters,
                'num_samples': MC_NUM_SAMPLES,
                'initial_samples': MC_INITIAL_SAMPLES,
                'exploration_method': 'stratified',
                'stratified_pool_size': 1000,
                'stratified_bins': 10,
                'stratified_cycle_length': min(50, iters),
                'stratified_gamma_start': 3.0,
                'interaction_dist_sample_decay': 0.9,
                'seed': self._seed,
                'abort_after_intervention': self._abort_after_intervention,
                'intervention_pval_corrected': self._intervention_pval_corrected,
            }
        if self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 250,
                'seed': self._seed,
            }
        elif self._optimizer_name == 'dynabo':
            user_iters = self._iterations_override if self._iterations_override is not None else 50
            n_trials = MC_INITIAL_SAMPLES + user_iters * MC_NUM_SAMPLES
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': n_trials,
                'seed': self._seed,
                'initial_design_size': MC_INITIAL_SAMPLES,
                'prior_std_denominator': 5,
                'prior_decay': 'quadratic',
                'prior_decay_enumerator': 20,
                'prior_decay_denominator': 1,
                'remove_old_priors': False,
                'abort_after_intervention': self._abort_after_intervention,
            }
        elif self._optimizer_name == 'bopro':
            self.setup_bopro_json(benchmark)
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'exp_json': self._bopro_file_name,
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')

    def get_pc_interventions(self):
        with open(self._interventions_path, 'r') as f:
            ints_json = json.load(f)

        interaction_iters, interactions = [], []
        for interaction in ints_json:
            interaction_iters.append(interaction['iteration'])
            interactions.append(interaction['intervention'])

        if len(self._interaction_idx) == 1:
            idx = self._interaction_idx[0]
            if idx == -1:
                return interactions, interaction_iters
            else:
                return [interactions[idx]], [interaction_iters[idx]]
        else:
            final_interactions, final_iterations = [], []
            for i in self._interaction_idx:
                final_interactions.append(interactions[i])
                final_iterations.append(interaction_iters[i])
            return final_interactions, final_iterations

    def get_pibo_intervention(self):
        print('Getting PiBO interaction. Iteration will be set to 0 by default.')
        with open('./interventions/floris_balanced.json', 'r') as f:
            ints_json = list(json.load(f))
        interaction = ints_json[self._interaction_idx[0]]
        if interaction['kind'] != 'dist':
            raise ValueError('PiBO only supports distributions as user input!')
        return self._convert_distribution_interaction(interaction['intervention'])

    def get_dynabo_interventions(self):
        with open(self._interventions_path, 'r') as f:
            ints_json = list(json.load(f))

        selected_idxs = self._interaction_idx
        if len(selected_idxs) == 1 and selected_idxs[0] == -1:
            selected_idxs = list(range(len(ints_json)))

        interventions, iterations = [], []
        for idx in selected_idxs:
            interaction = ints_json[idx]
            interaction_spec = interaction['intervention']
            if interaction_spec is None:
                continue

            if interaction.get('kind') == 'dist':
                intervention_dict = self._convert_distribution_interaction(interaction_spec)
            else:
                intervention_dict = interaction_spec

            interventions.append(intervention_dict)
            iterations.append(MC_INITIAL_SAMPLES + interaction['iteration'] * MC_NUM_SAMPLES)

        return interventions, iterations

    def _convert_distribution_interaction(self, interaction_spec):
        interaction_dict = {}
        for k, v in interaction_spec.items():
            if v['dist'] == 'cat':
                weights = softmax(v['parameters'])
                hp = CategoricalHyperparameter(k, v['values'], weights=weights)
            elif v['dist'] == 'uniform':
                mi, ma = v['parameters']
                hp = UniformFloatHyperparameter(k, mi, ma)
            elif v['dist'] == 'int_uniform':
                mi, ma = v['parameters']
                hp = UniformIntegerHyperparameter(k, mi, ma, default_value=mi)
            elif v['dist'] == 'gauss':
                mean, std = v['parameters']
                hp = NormalFloatHyperparameter(k, mean, std)
            else:
                continue

            interaction_dict[k] = hp

        return interaction_dict

    def get_bopro_intervention(self):
        print('Getting BOPrO interaction. Iteration will be set to 0 by default.')
        with open('./interventions/floris_balanced.json', 'r') as f:
            ints_json = list(json.load(f))
        interaction = ints_json[self._interaction_idx[0]]
        if interaction['kind'] != 'dist':
            raise ValueError('BOPrO only supports distributions as user input!')
        interaction_spec = interaction['intervention']
        interaction_dict = {}
        for k, v in interaction_spec.items():
            if v['dist'] == 'cat':
                weights = list(softmax(v['parameters']))
                hp = {'prior': weights}
            elif v['dist'] == 'uniform' or v['dist'] == 'int_uniform':
                mi, ma = v['parameters']
                hp = {'prior': 'uniform', 'values': [mi, ma]}
            elif v['dist'] == 'gauss':
                mean, std = v['parameters']
                hp = {
                    'prior': 'custom_gaussian',
                    'custom_gaussian_prior_means': [mean],
                    'custom_gaussian_prior_stds': [std],
                }
            else:
                continue

            interaction_dict[k] = hp

        return interaction_dict

    def setup_bopro_json(self, benchmark):
        search_space = benchmark.search_space
        borpo_search_space = search_space.to_hypermapper()
        json_dict = {
            'application_name': 'Floris Interactive',
            'optimization_objectives': ['value'],
            'design_of_experiment': {
                'number_of_samples': 3,
            },
            'optimization_iterations': 100,
            'optimization_method': 'prior_guided_optimization',
            'number_of_cpus': 0,
            'models': {
                'model': 'random_forest'
            },
            'input_parameters': borpo_search_space,
            'local_search_starting_points': 10,
            'local_search_random_points': 50,
            'local_search_evaluation_limit': 200,
        }
        if not os.path.exists('./bopro_experiments/'):
            os.mkdir('./bopro_experiments')
        self._bopro_file_name = './bopro_experiments/floris_interactive.json'
        with open(self._bopro_file_name, 'w+') as f:
            json.dump(json_dict, f)

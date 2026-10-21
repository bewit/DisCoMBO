from copy import deepcopy

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from syne_tune.config_space import Float

from .search_space import SearchSpace
from ..consts.dtypes import MetaType


class PVSearchSpace(SearchSpace):

    def __init__(self, search_space_definition) -> None:
        super().__init__()
        self.search_space_definition = search_space_definition

    def sample(self, size=1, keep_domain=True, exclude_fidelities=False, **kwargs):
        samples = []
        for _ in range(size):
            x_sample = {}
            for name, val in self.search_space_definition.items():
                x = np.random.uniform(val['min'], val['max'])
                x_sample[name] = float(x)
            samples.append(x_sample)
        return samples

    def get_neighbors(self, config: dict, num_cont_neighbors=6):
        neighbors = []
        if config is None:
            return self.sample()

        for name in self.search_space_definition.keys():
            curr_value = config[name]
            min_val = self.search_space_definition[name]['min']
            max_val = self.search_space_definition[name]['max']
            dist = max_val - min_val
            rnd_left = np.random.uniform(0, 0.2 * dist, size=(num_cont_neighbors // 2))
            rnd_right = np.random.uniform(0, -0.2 * dist, size=(num_cont_neighbors // 2))
            rnd = np.concatenate((rnd_left, rnd_right))
            rnd += curr_value
            ns = np.clip(rnd, min_val, max_val)
            for neighbor in ns.tolist():
                curr_config = deepcopy(config)
                curr_config[name] = float(neighbor)
                neighbors.append(curr_config)
        return neighbors

    def to_configspace(self):
        config_space = ConfigurationSpace()
        for key, val in self.search_space_definition.items():
            hp = UniformFloatHyperparameter(key, val['min'], val['max'], log=val['is_log'])
            config_space.add_hyperparameter(hp)
        return config_space

    def to_synetune(self):
        config_space = {}
        for key, val in self.search_space_definition.items():
            hp = Float(val['min'], val['max'])
            config_space[key] = hp
        return config_space

    def to_hypermapper(self):
        config_space = {}
        for key, val in self.search_space_definition.items():
            hp = {
                'parameter_type': 'real',
                'values': [val['min'], val['max']]
            }
            config_space[key] = hp
        return config_space

    def is_valid(self, config):
        keys_match = set(config.keys()) == set(self.search_space_definition.keys())
        if not keys_match:
            return False

        for key, val in config.items():
            if not isinstance(val, (float, int, np.floating, np.integer)):
                return False
            if val < self.search_space_definition[key]['min'] or val > self.search_space_definition[key]['max']:
                return False
        return True

    def get_search_space_definition(self):
        return self.search_space_definition

    @classmethod
    def from_feature_bounds(cls, feature_bounds):
        search_space_definition = {}
        for feature_name, bounds in feature_bounds.items():
            search_space_definition[feature_name] = {
                'type': MetaType.REAL,
                'min': float(bounds[0]),
                'max': float(bounds[1]),
                'dtype': 'float',
                'is_log': False,
            }
        return cls(search_space_definition)

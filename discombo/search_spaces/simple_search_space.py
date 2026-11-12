import numpy as np
from .search_space import SearchSpace


class SimpleSearchSpace(SearchSpace):

    """
        Search space given directly by its definition, a dict mapping each hyperparameter name to
            {'dtype': 'float', 'min': ..., 'max': ..., 'is_log': bool}   for continuous hyperparameters, or
            {'dtype': 'int' | 'str' | 'cat' | 'bool', 'allowed': [...]}   for discrete hyperparameters.
        Continuous hyperparameters are sampled uniformly (log-uniformly if 'is_log'),
        discrete ones uniformly from their 'allowed' values.
    """

    def __init__(self, definition: dict) -> None:
        super().__init__()
        self.search_space_definition = {}
        for name, hp_def in definition.items():
            hp_def = dict(hp_def)
            if hp_def['dtype'] == 'float':
                hp_def.setdefault('is_log', False)
            elif 'allowed' not in hp_def:
                raise ValueError(f"Discrete hyperparameter '{name}' needs a list of 'allowed' values.")
            self.search_space_definition[name] = hp_def

    def sample(self, size=1, keep_domain=True, **kwargs):
        """
            Sample random configurations. With keep_domain=False, discrete hyperparameters
            are returned as indices into their 'allowed' values.
        """
        configurations = []
        for _ in range(size):
            cfg = {}
            for name, hp_def in self.search_space_definition.items():
                if hp_def['dtype'] == 'float':
                    if hp_def['is_log']:
                        cfg[name] = float(np.exp(np.random.uniform(np.log(hp_def['min']), np.log(hp_def['max']))))
                    else:
                        cfg[name] = float(np.random.uniform(hp_def['min'], hp_def['max']))
                else:
                    idx = np.random.randint(len(hp_def['allowed']))
                    cfg[name] = hp_def['allowed'][idx] if keep_domain else idx
            configurations.append(cfg)
        return configurations

    def get_search_space_definition(self):
        return self.search_space_definition

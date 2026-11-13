from typing import Dict, List
from ..search_spaces import SearchSpace
from .._deprecation import warn_deprecated


class Optimizer:
    """Interface of DisCoMBO's optimizers, see DisCoOptimizer."""

    def __init__(self, search_space, objective, seed=0) -> None:
        self.search_space = search_space
        self.objective = objective
        self.evaluations = []
        self.seed = seed

    def optimize(self):
        raise NotImplementedError('Optimizer base class has no optimize implementation')

    def ask(self) -> List[Dict]:
        raise NotImplementedError('Optimizer base class has no ask implementation')

    def tell(self, configs: List[Dict], results) -> None:
        raise NotImplementedError('Optimizer base class has no tell implementation')

    def best(self):
        raise NotImplementedError('Optimizer base class has no best implementation')

    def intervene(self, interventions: List[Dict], iters):
        raise NotImplementedError('Optimizer base class has no intervene implementation')

    def adapt_to_search_space(self, new_search_space: Dict):
        """Deprecated."""
        warn_deprecated("adapt_to_search_space()")
        raise NotImplementedError('Optimizer base class has no adapt_to_search_space implementation')

    def set_search_space(self, search_space: SearchSpace):
        """Deprecated."""
        warn_deprecated("set_search_space()")
        self.search_space = search_space
        self.evaluations = []

    @property
    def history(self):
        raise NotImplementedError('Optimizer base class does not implement history')

import logging
from importlib.metadata import PackageNotFoundError, version

from .result import EvaluationResult
from .optimizers import DisCoOptimizer, Optimizer
from .search_spaces import SearchSpace, SimpleSearchSpace
from ._deprecation import module_getattr

try:
    __version__ = version("discombo")
except PackageNotFoundError:  # not installed, e.g. imported from a source checkout
    __version__ = "unknown"

# DisCoMBO reports its progress via logging; the application decides whether and where it is shown.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):  # deprecated names
    return module_getattr(globals(), name)

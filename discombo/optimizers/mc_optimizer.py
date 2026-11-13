"""
    Deprecated module path of DisCoOptimizer.
"""
from .disco_optimizer import DisCoOptimizer
from .._deprecation import module_getattr


def __getattr__(name):  # deprecated names
    return module_getattr(globals(), name)

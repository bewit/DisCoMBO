"""
    Deprecated names, which keep working with a DeprecationWarning.
"""
import functools
import warnings
from typing import Dict, Optional

# module-level names: old name -> new name
RENAMED = {"MCOptimizer": "DisCoOptimizer", "BenchQueryResult": "EvaluationResult"}


def warn_deprecated(what: str, instead: Optional[str] = None, stacklevel: int = 3) -> None:
    """Warn that `what` is deprecated; by default, the warning points at the caller of the deprecated name."""
    message = f"{what} is deprecated and will be removed in DisCoMBO 2.0"
    message += f"; use {instead} instead." if instead else "."
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def module_getattr(module_globals: dict, name: str):
    """Module `__getattr__` (PEP 562) resolving the deprecated module-level names whose new name is in the module."""
    if RENAMED.get(name) not in module_globals:
        raise AttributeError(f"module {module_globals['__name__']!r} has no attribute {name!r}")
    warn_deprecated(name, RENAMED[name], stacklevel=4)
    return module_globals[RENAMED[name]]


def renamed_arguments(renamed: Dict[str, str]):
    """Decorator accepting the deprecated keyword arguments in `renamed` (old name -> new name) of a function."""
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for old, new in renamed.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"{function.__qualname__}() got both {old!r} and its new name {new!r}.")
                    warn_deprecated(f"The argument {old}", new)
                    kwargs[new] = kwargs.pop(old)
            return function(*args, **kwargs)
        return wrapper
    return decorator

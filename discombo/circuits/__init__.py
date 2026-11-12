"""
    Probabilistic circuits of DisCoMBO, built on spflow (>= 1.0).

    Only relative imports are used inside this package, so it does not pull in the rest of `discombo`.
"""
from .leaves import (BINARY, DISCRETE, REAL, LegacyHistogram, VarSpec, compute_domains, metatype_specs,
                     parametric_specs)
from .learning import HISTOGRAM, PARAMETRIC, learn_circuit
from .inference import ConditioningError, children, condition, log_likelihood, mixture, sample
from .pvalue import fishers_method, pvalues

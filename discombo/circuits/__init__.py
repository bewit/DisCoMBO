"""
    Probabilistic circuits of DisCoMBO, built on spflow (>= 1.0).
"""
from .leaves import BINARY, DISCRETE, REAL, VarSpec, compute_domains, parametric_specs
from .learning import learn_circuit
from .inference import ConditioningError, children, condition, log_likelihood, sample
from .pvalue import fishers_method, pvalues

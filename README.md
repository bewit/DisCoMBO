# DisCoMBO: Steering Expert-in-the-Loop Black Box Optimization via Distributional Conformance
This repository contains the code of `Steering Expert-in-the-Loop Black Box Optimization via Distributional Conformance` (https://arxiv.org/pdf/2609.36472).

## Abstract
Sequential Model-Based Optimization (SMBO) traditionally relies on Bayesian or ensembling surrogates for uncertainty quantification. While historically treated as fully data-driven, SMBO increasingly integrates external domain expertise to accelerate discovery. To overcome the opaque guidance and diminished integration fidelity of standard acquisition re-weighting, Probabilistic Circuits (PCs) have emerged as a generative surrogate alternative, enabling direct knowledge injection via conditional sampling. However, these generative routines lack the formal exploration-exploitation semantics required for rigorous optimization. We introduce the \textbf{Dis}tributional \textbf{Co}nformance Score (DisCo), a novel metric that unifies the flexibility and efficiency of PCs with a formal uncertainty framework. DisCo provides a bounded, $[0, 1]$-normalized measure of model "surprise" that (1) recovers properties comparable to kernel-based uncertainty known from, e.g., Gaussian Processes, while maintaining linear-time inference, and (2) enables accurate assessment of conformance of external knowledge w.r.t. model evidence. We then present DisCoMBO, a framework leveraging these properties for robust, knowledge-aware optimization. We prove that DisCoMBO is a zero-regret algorithm and demonstrate its effectiveness across diverse benchmarks from AutoML, material optimization, and wind park optimization.

## Branches
| Branch | Content |
|---|---|
| `main` | DisCoMBO only, with minimal dependencies (Python >= 3.10, SPFlow v1.1.0). |
| `paper` | The code and environment that produced the results of the paper: DisCoMBO, all baselines and benchmarks (legacy SPFlow, Python 3.9, Docker). |
| `new_spflow_benchmarking` | The benchmarking environment of the paper migrated to SPFlow v1.1.0 (Python 3.10, Docker). |

## Installation
DisCoMBO requires Python >= 3.10 and is built on [SPFlow](https://github.com/SPFlow/SPFlow) v1.1.0, PyTorch, NumPy, SciPy and scikit-learn.

```
pip install -e .
```

## Usage
DisCoMBO maximizes an objective over a search space. The search space is defined by a dictionary mapping each hyperparameter to its type and domain;
the objective takes a configuration (a dictionary) and returns a `BenchQueryResult`, whose validation performance is maximized.

```python
from discombo import BenchQueryResult, MCOptimizer, SimpleSearchSpace

space = SimpleSearchSpace({
    'lr': {'dtype': 'float', 'min': 1e-4, 'max': 1e-1, 'is_log': True},
    'layers': {'dtype': 'int', 'allowed': [1, 2, 3, 4]},
    'optimizer': {'dtype': 'str', 'allowed': ['sgd', 'adam']},
})

def objective(cfg):
    score = ...  # evaluate cfg
    return BenchQueryResult(score, score, score)  # train, validation and test performance

optimizer = MCOptimizer(space, objective, iterations=50, num_samples=10, initial_samples=20, seed=0)
best_cfg, best_result = optimizer.optimize()
```

Continuous hyperparameters use `'dtype': 'float'` with `'min'`, `'max'` and optionally `'is_log'`; discrete hyperparameters use `'dtype'` `'int'`, `'str'`, `'cat'` or `'bool'` with a list of `'allowed'` values.

### Expert knowledge
Expert beliefs are injected as interventions in given iterations, either as fixed values (point interventions) or as distributions over values.
Discrete hyperparameters are referred to by the index of the value in `'allowed'`.

```python
optimizer.intervene(
    [{'lr': {'dist': 'gauss', 'parameters': [-4.6, 0.5]}},    # distributions: 'gauss', 'uniform', 'int_uniform', 'cat'
     {'optimizer': 1}],                                         # point intervention: 'adam'
    [10, 25],                                                   # iterations in which the interventions are applied
)
```

Continuous hyperparameters with `'is_log': True` are modeled in log space, so their intervention values are given in log space as well.
After optimization, `optimizer.get_intervention_pval_log()` returns the distributional conformance (p-values) of each intervention w.r.t. the model.

## Structure
- `discombo/optimizers/mc_optimizer.py`: the DisCoMBO optimizer.
- `discombo/circuits/`: the probabilistic circuits on top of SPFlow: LearnSPN structure learning, conditioning, conditional sampling and the bottom-up p-value inference (weighted arithmetic mean at sum nodes, Fisher's method at product nodes).
- `discombo/search_spaces/`: the search space interface and `SimpleSearchSpace`.

## Tests
```
pip install -e ".[test]"
pytest tests/
```

# DisCoMBO: Steering Expert-in-the-Loop Black-Box Optimization via Distributional Conformance
This repository contains the code of [DisCoMBO: Steering Expert-in-the-Loop Black-Box Optimization via Distributional Conformance](https://arxiv.org/abs/2609.36472) (NeurIPS 2026).

## Abstract
Sequential Model-Based Optimization (SMBO) traditionally relies on Bayesian or ensembling surrogates for uncertainty quantification. While historically treated as fully data-driven, SMBO increasingly integrates external domain expertise to accelerate discovery. To overcome the opaque guidance and diminished integration fidelity of standard acquisition re-weighting, Probabilistic Circuits (PCs) have emerged as a generative surrogate alternative, enabling direct knowledge injection via conditional sampling. However, these generative routines lack the formal exploration-exploitation semantics required for rigorous optimization. We introduce the **Dis**tributional **Co**nformance Score (DisCo), a novel metric that unifies the flexibility and efficiency of PCs with a formal uncertainty framework. DisCo provides a bounded, [0, 1]-normalized measure of model "surprise" that (1) recovers properties comparable to kernel-based uncertainty known from, e.g., Gaussian Processes, while maintaining linear-time inference, and (2) enables accurate assessment of conformance of external knowledge w.r.t. model evidence. We then present DisCoMBO, a framework leveraging these properties for robust, knowledge-aware optimization. We prove that DisCoMBO is a zero-regret algorithm and demonstrate its effectiveness across diverse benchmarks from AutoML, material optimization, and wind park optimization.

## Branches
| Branch | Content |
|---|---|
| [`main`](https://github.com/bewit/DisCoMBO/tree/main) | DisCoMBO only, with minimal dependencies (Python >= 3.10, SPFlow 1.1). This is the version released on PyPI. |
| [`paper`](https://github.com/bewit/DisCoMBO/tree/paper) | The code and environment that produced the results of the paper: DisCoMBO, all baselines and benchmarks (legacy SPFlow, Python 3.9, Docker). |
| [`new_spflow_benchmarking`](https://github.com/bewit/DisCoMBO/tree/new_spflow_benchmarking) | The benchmarking environment of the paper migrated to SPFlow 1.1.0 (Python 3.10, Docker). |

## Installation
DisCoMBO requires Python >= 3.10 and is built on [SPFlow](https://github.com/SPFlow/SPFlow) 1.1, PyTorch, NumPy, SciPy and scikit-learn.

**From PyPI:**
```
pip install discombo
```

**From the repository:**
```
pip install git+https://github.com/bewit/DisCoMBO.git
```
or, for development, as an editable install of a clone (including the test dependencies):
```
git clone https://github.com/bewit/DisCoMBO.git
cd DisCoMBO
pip install -e ".[test]"
```

**For Optuna:** DisCoMBO is available as an Optuna sampler on [OptunaHub](https://hub.optuna.org/). The sampler is loaded at runtime with `optunahub` (see *Usage with Optuna* below); its dependencies, including DisCoMBO from PyPI, are installed with
```
pip install optuna optunahub
pip install -r https://hub.optuna.org/samplers/discombo/requirements.txt
```

**CPU-only PyTorch:** DisCoMBO runs on the CPU. On Linux, `pip` installs the CUDA build of PyTorch by default, which takes several GB.
To get the much smaller CPU build, install PyTorch from the PyTorch CPU index before installing DisCoMBO with any of the options above:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install discombo
```
On macOS and Windows, the default PyTorch build from PyPI is already CPU-only.

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

Runs are reproducible: all randomness of DisCoMBO is drawn from a random state seeded with `seed`, and the global random states of NumPy and PyTorch are left untouched.
Custom search spaces (subclasses of `SearchSpace`) should draw from the `rng` passed to their `sample` method.

DisCoMBO reports its progress (the best observed value per iteration and the applied interventions) via Python's `logging` under the logger name `discombo`.
To show it, configure logging in your application, e.g.
```python
import logging
logging.basicConfig(level=logging.INFO)
```

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

### Usage with Optuna
```python
import optuna
import optunahub

module = optunahub.load_module("samplers/discombo")
study = optuna.create_study(direction="maximize", sampler=module.DisCoMBOSampler(seed=0))
study.optimize(objective, n_trials=100)  # objective(trial) as usual in Optuna
```

## Structure
- `discombo/optimizers/mc_optimizer.py`: the DisCoMBO optimizer.
- `discombo/circuits/`: the probabilistic circuits on top of SPFlow: LearnSPN structure learning, conditioning, conditional sampling and the bottom-up p-value inference (weighted arithmetic mean at sum nodes, Fisher's method at product nodes).
- `discombo/search_spaces/`: the search space interface and `SimpleSearchSpace`.

## Tests
```
pip install -e ".[test]"
pytest tests/
```

## Citation
If you use DisCoMBO, please cite our paper ([CITATION.bib](https://github.com/bewit/DisCoMBO/blob/main/CITATION.bib)):
```bibtex
@inproceedings{seng2026discombo,
  title         = {{DisCoMBO}: Steering Expert-in-the-Loop Black-Box Optimization via Distributional Conformance},
  author        = {Seng, Jonas and Wittelsbach, Bennet and Kersting, Kristian},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {40},
  year          = {2026},
  eprint        = {2609.36472},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.36472}
}
```

import numpy as np
import pandas as pd
from time import time
from .optimizer import Optimizer
from ..search_spaces import SearchSpace


class HEBOOptimizer(Optimizer):
    """
    HEBO optimizer (Heteroscedastic Evolutionary Bayesian Optimization).

    Builds a HEBO DesignSpace from the project's SearchSpace definition:
      - float HPs  -> {'type': 'num', 'lb': min, 'ub': max}
      - int HPs    -> {'type': 'int', 'lb': 0, 'ub': len(allowed)-1}
      - str/bool   -> {'type': 'cat', 'categories': allowed}

    `initial_samples` random configs are evaluated and observed upfront (same
    as MC/PC), giving the GP real data before BO begins. `rand_sample` is set
    equal to `initial_samples` so HEBO's own random phase is bypassed and the
    GP is used from iteration 0.

    HEBO minimizes, so val_performance is negated in observe().
    Categorical/int HPs are stored as index in the HEBO space and
    converted back to original values via the allowed list.
    """

    def __init__(self, search_space: SearchSpace, objective,
                 iterations=50, num_samples=20, initial_samples=20, log_hpo_runtime=False, seed=0) -> None:
        super().__init__(search_space, objective)
        self.search_space = search_space
        self.objective = objective
        self._iterations = iterations
        self._num_samples = num_samples
        self._initial_samples = initial_samples
        self._log_hpo_runtime = log_hpo_runtime
        self._seed = seed
        self._history = []
        self._ssd = search_space.get_search_space_definition()
        self._hp_names = list(self._ssd.keys())

    def _build_design_space(self):
        from hebo.design_space.design_space import DesignSpace
        params = []
        for name, val in self._ssd.items():
            if val['dtype'] == 'float':
                params.append({'name': name, 'type': 'num',
                                'lb': float(val['min']), 'ub': float(val['max'])})
            elif val['dtype'] == 'int':
                params.append({'name': name, 'type': 'int',
                                'lb': 0, 'ub': len(val['allowed']) - 1})
            else:
                # str, bool — use categorical with original values as strings
                categories = [str(c) for c in val['allowed']]
                params.append({'name': name, 'type': 'cat', 'categories': categories})
        return DesignSpace().parse(params)

    def _suggestion_to_config(self, rec: pd.DataFrame) -> dict:
        row = rec.iloc[0]
        config = {}
        for name, val in self._ssd.items():
            if val['dtype'] == 'float':
                config[name] = float(row[name])
            elif val['dtype'] == 'int':
                idx = int(round(float(row[name])))
                idx = max(0, min(idx, len(val['allowed']) - 1))
                config[name] = val['allowed'][idx]
            else:
                # categorical: HEBO returns the string value directly
                raw = row[name]
                # match back to original type (e.g. bool)
                for choice in val['allowed']:
                    if str(choice) == str(raw):
                        config[name] = choice
                        break
                else:
                    config[name] = raw
        return config

    def _config_to_rec(self, config: dict) -> pd.DataFrame:
        """Convert a config dict to a HEBO-compatible DataFrame row for observe()."""
        row = {}
        for name, val in self._ssd.items():
            if val['dtype'] == 'float':
                row[name] = float(config[name])
            elif val['dtype'] == 'int':
                # HEBO int space uses 0-based index, not the actual value
                idx = val['allowed'].index(config[name])
                row[name] = float(idx)
            else:
                # str, bool — HEBO stores as string
                row[name] = str(config[name])
        return pd.DataFrame([row])

    def optimize(self):
        from hebo.optimizers.hebo import HEBO
        space = self._build_design_space()
        # rand_sample=initial_samples: HEBO switches from random to GP-based
        # suggestions once it has this many observations. Since we pre-observe
        # exactly initial_samples configs, the GP is used from iteration 0.
        opt = HEBO(space, rand_sample=self._initial_samples, scramble_seed=self._seed)

        # --- Initial random samples (same as MC/PC warm-up) ---
        print(f"[HEBO] Evaluating {self._initial_samples} initial random samples...")
        init_configs = self.search_space.sample(size=self._initial_samples)
        for cfg in init_configs:
            result = self.objective(cfg)
            self._history.append((cfg, result, -1))
            rec = self._config_to_rec(cfg)
            opt.observe(rec, np.array([[-result.val_performance]]))
        print(f"[HEBO] Initial best: {max(r.val_performance for _, r, _ in self._history):.4f}")

        total_proposal_time = 0.0
        for iteration in range(self._iterations):
            if iteration % 10 == 0:
                print(f"[HEBO] iter {iteration}/{self._iterations}")
            t0 = time()
            recs = opt.suggest(n_suggestions=self._num_samples)
            if self._log_hpo_runtime:
                total_proposal_time += time() - t0
            ys = []
            for j in range(len(recs)):
                config = self._suggestion_to_config(recs.iloc[[j]])
                result = self.objective(config)
                self._history.append((config, result, iteration))
                ys.append(-result.val_performance)
            opt.observe(recs, np.array(ys).reshape(-1, 1))
            print(f"BEST: {max(r.val_performance for _, r, _ in self._history)}")
        self._proposal_time = total_proposal_time

        val_perfs = [r.val_performance for _, r, _ in self._history]
        best_idx = int(np.argmax(val_perfs))
        return self._history[best_idx][:2]

    @property
    def history(self):
        if self._log_hpo_runtime:
            optim_cost = getattr(self, '_proposal_time', 0.0)
            for cfg, res, it in self._history:
                res.set_optim_cost(optim_cost)
        return self._history

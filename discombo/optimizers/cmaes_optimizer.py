import numpy as np
import cma
from time import time
from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..utils import ConfigurationNumericalTransform


class CMAESOptimizer(Optimizer):
    """
    CMA-ES optimizer wrapping the `cma` library (cma.CMAEvolutionStrategy).

    Iteration semantics match MCOptimizer: each iteration samples and evaluates
    exactly `num_samples` new HP vectors, then updates the CMA-ES model. So
    `iterations=50, num_samples=20` performs 1000 total evaluations (plus
    `initial_samples` random warm-up evaluations), matching MC's budget when
    configured identically.

    `initial_samples` random configs are evaluated first (same as MC/PC). The
    best among them is used as x0 for CMA-ES, giving a data-driven warm start
    instead of the geometric center of bounds.

    `num_samples` sets CMA's population size (popsize/lambda). The natural
    CMA-ES default would be `4 + floor(3*ln(N_dims))`, but overriding it to
    match MC allows fair budget comparison.

    The search space is encoded to a continuous float vector using
    ConfigurationNumericalTransform (same as MC/PC optimizers):
      - float HPs: raw value (or log-value if is_log)
      - discrete/categorical/binary HPs: integer index into allowed list

    CMA-ES minimizes, so val_performance is negated when passed to tell().
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
        self.transform = ConfigurationNumericalTransform(search_space)
        self._ssd = search_space.get_search_space_definition()
        self._lb, self._ub, self._is_discrete = self._build_bounds()

    def _build_bounds(self):
        lb, ub, is_discrete = [], [], []
        for key, val in self._ssd.items():
            if val['dtype'] == 'float':
                if val.get('is_log', False):
                    lb.append(np.log(val['min']))
                    ub.append(np.log(val['max']))
                else:
                    lb.append(val['min'])
                    ub.append(val['max'])
                is_discrete.append(False)
            else:
                lb.append(0)
                ub.append(len(val['allowed']) - 1)
                is_discrete.append(True)
        return np.array(lb), np.array(ub), np.array(is_discrete)

    def _clamp_and_round(self, x):
        x = np.clip(x, self._lb, self._ub)
        for i, disc in enumerate(self._is_discrete):
            if disc:
                x[i] = round(x[i])
        return x

    def optimize(self):
        # --- Initial random samples (same as MC/PC warm-up) ---
        print(f"[CMA-ES] Evaluating {self._initial_samples} initial random samples...")
        init_configs = self.search_space.sample(size=self._initial_samples)
        for cfg in init_configs:
            result = self.objective(cfg)
            self._history.append((cfg, result, -1))

        # Warm-start x0 from best initial sample instead of center of bounds
        best_init_idx = int(np.argmax([r.val_performance for _, r, _ in self._history]))
        best_init_cfg = self._history[best_init_idx][0]
        x0 = np.array(self.transform.transform_configuration(best_init_cfg), dtype=float)
        x0 = self._clamp_and_round(x0)
        print(f"[CMA-ES] Warm-start x0 from best initial sample "
              f"(val_perf={self._history[best_init_idx][1].val_performance:.4f})")

        ranges = self._ub - self._lb
        # Per-dimension step sizes: sigma * stds[i] ≈ range[i] * 0.3.
        # Using sigma0=1 with CMA_stds avoids the single-sigma-for-all-dims problem
        # that occurs when dims have very different scales (e.g. epoch range=199
        # vs Resolution range=1 on JAHS).
        sigma0 = 1.0
        stds = np.where(ranges > 0, ranges * 0.3, 1.0)

        opts = {
            'bounds': [self._lb.tolist(), self._ub.tolist()],
            'CMA_stds': stds.tolist(),
            'popsize': self._num_samples,
            # maxiter and maxfevals are not set — we drive the loop ourselves for
            # exactly self._iterations generations. CMA's own budget counters would
            # fire es.stop() at unexpected times, breaking the iteration alignment.
            'maxiter': int(1e9),
            'verbose': -9,
            'seed': self._seed,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        # Each iteration = one full CMA generation of num_samples evaluations,
        # matching the MCOptimizer convention (iterations x num_samples total evals).
        total_proposal_time = 0.0
        for iteration in range(self._iterations):
            if iteration % 10 == 0:
                print(f"Iteration {iteration + 1}/{self._iterations}")
            t0 = time()
            solutions = es.ask()
            if self._log_hpo_runtime:
                total_proposal_time += time() - t0
            fitnesses = []
            for x in solutions:
                x_clamped = self._clamp_and_round(x.copy())
                config = self.transform.inv_transform_configuration(list(x_clamped))
                result = self.objective(config)
                self._history.append((config, result, iteration))
                fitnesses.append(-result.val_performance)
            t0 = time()
            es.tell(solutions, fitnesses)
            if self._log_hpo_runtime:
                total_proposal_time += time() - t0
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

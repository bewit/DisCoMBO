from typing import Dict, List
from .optimizer import Optimizer
from ..result import EvaluationResult
from ..circuits import ConditioningError, condition, learn_circuit, parametric_specs, pvalues, sample
from ..search_spaces import SearchSpace
from ..utils import ConfigurationNumericalTransform
from ..utils.sampling import sample_from_ball
from .._deprecation import renamed_arguments, warn_deprecated
import logging
import numpy as np
from time import time

logger = logging.getLogger(__name__)

"""
    Implementation of the DisCoMBO optimizer.
    Uses LearnSPN circuits on spflow (discombo.circuits) for structure learning and the distributional
    conformance (DisCo) score of configurations, their p-value under the circuit (Fisher's method at product
    nodes, weighted corrected adaptive arithmetic mean at sum nodes).
"""


# deprecated constructor arguments -> their names
_RENAMED_ARGUMENTS = {
    'interaction_dist_sample_decay': 'intervention_decay',
    'mc_min_instances': 'circuit_min_instances',
    'mc_rdc_threshold': 'circuit_rdc_threshold',
    'mc_k_clusters': 'circuit_k_clusters',
    'log_hpo_runtime': 'log_runtime',
    'intervention_pval_corrected': 'disco_score_corrected',
}
# deprecated attributes -> their names, to unpickle optimizers with deprecated attributes
_RENAMED_ATTRIBUTES = {
    'curr_mc': 'circuit',
    'hpo_runtimes': 'runtimes',
    '_log_hpo_runtime': '_log_runtime',
    '_mc_min_instances': '_circuit_min_instances',
    '_mc_rdc_threshold': '_circuit_rdc_threshold',
    '_mc_k_clusters': '_circuit_k_clusters',
    '_intervention_pval_corrected': '_disco_score_corrected',
    '_interaction_dist_sample_decay': '_intervention_decay',
    '_prob_interaction_pc': '_intervention_acceptance_prob',
}


class DisCoOptimizer(Optimizer):

    @renamed_arguments(_RENAMED_ARGUMENTS)
    def __init__(self, search_space: SearchSpace, objective=None, iterations=100,
                 num_samples=20, initial_samples=20, exploration_method=None,
                 stratified_pool_size=1000, stratified_bins=10,
                 stratified_gamma_start=2.0, stratified_gamma_end=0.25,
                 stratified_cycle_length=20,
                 intervention_decay=0.9, max_rel_ball_size=0.05,
                 max_rel_ball_size_explore=None, max_rel_ball_size_exploit=None,
                 normalize_inputs=False,
                 circuit_min_instances=10, circuit_rdc_threshold=0.3, circuit_k_clusters=2,
                 mc_type=None, log_runtime=False, seed=0,
                 abort_after_intervention=False,
                 disco_score_corrected=False,
                 hybrid_pool_uniform_frac=0.5,
                 condition_perf_quantile_lo=0.7) -> None:
        """
        DisCoMBO optimizer.

        Args:
            search_space: SearchSpace object defining the hyperparameter domain.
            objective: Objective function taking a config dict, returning an EvaluationResult (or a number),
                both maximized. Only needed for `optimize`; with `ask`/`tell` it can be None.
            iterations: Number of optimization iterations of `optimize`.
            num_samples: Number of candidate configs per iteration.
            initial_samples: Number of random configurations of the initial design.
            exploration_method: Deprecated.
            stratified_pool_size: Number of candidates in the large pool for stratified
                DisCo score sampling.
            stratified_bins: Number of DisCo score bins for stratified selection.
                Selected candidates are drawn uniformly from each bin to ensure
                approximately uniform DisCo score coverage.
            stratified_gamma_start: Power-law exponent at iteration 0 (>1 → exploration bias:
                tiny bins at the low-score end → high sampling density at low DisCo scores).
            stratified_gamma_end: Power-law exponent at exploitation phase (<1 → tiny bins at
                the high-score end → high sampling density at high DisCo scores).
            stratified_cycle_length: Half-period (in iterations) of the sinusoidal γ schedule.
                γ(t) = γ_mid + γ_amp·cos(π·t/cycle_length), cycling exploration→exploitation
                every cycle_length iterations (default 20: explore at t=0, exploit at t=20,
                explore again at t=40, …).
            intervention_decay: Per-iteration decay of the probability (initially 1) that an active intervention
                conditions the candidate pool.
            max_rel_ball_size: Relative perturbation radius for ball sampling.
                Used as the constant per-iteration radius unless `max_rel_ball_size_explore`
                and `max_rel_ball_size_exploit` are both set (γ-coupled schedule).
            max_rel_ball_size_explore: Per-dim relative ball radius at the explore
                end of the γ cycle (γ = stratified_gamma_start). When both this and
                `max_rel_ball_size_exploit` are set, the radius is interpolated each
                iteration along the same cosine γ schedule used by stratified
                bin power-law. Either both or neither must be set.
            max_rel_ball_size_exploit: Per-dim relative ball radius at the exploit
                end of the γ cycle (γ = stratified_gamma_end). See above.
            normalize_inputs: If True, min-max scale float HP columns to [0, 1]
                using the search-space bounds (log-bounds when is_log=True) before
                fitting the circuit. Categorical/discrete dims and the performance
                column are left untouched.
            circuit_min_instances: Minimum number of instances per leaf in the circuit's structure learning
                (LearnSPN).
            circuit_rdc_threshold: RDC independence threshold for column partitioning in structure learning.
            circuit_k_clusters: Number of row clusters in structure learning.
            mc_type: Deprecated.
            log_runtime: Whether to record the optimizer's own runtime per iteration (`runtimes`, reported as
                `optim_cost` in `history`).
            seed: Random seed. All randomness of the optimizer is drawn from a random state seeded with it;
                numpy's and torch's global random states are left untouched.
            abort_after_intervention: Stop the optimization after computing the DisCo score of the first
                intervention.
            disco_score_corrected: Multiply the DisCo scores of interventions at sum nodes by the correction
                factor min(2, 1 / max(weights)).
        """
        if exploration_method is not None:
            warn_deprecated("The argument exploration_method", stacklevel=4)
            if exploration_method != 'stratified':
                raise ValueError(f"Only exploration_method='stratified' is supported, got {exploration_method!r}.")
        if mc_type is not None:
            warn_deprecated("The argument mc_type", stacklevel=4)
            if mc_type != 'mixed':
                raise ValueError(f"Only mc_type='mixed' is supported, got {mc_type!r}.")
        super().__init__(search_space, objective, seed=seed)
        self.search_space = search_space
        self.objective = objective
        self._num_iterations = iterations
        self._samples_per_iter = num_samples
        self._initial_samples = initial_samples
        self._stratified_pool_size = stratified_pool_size
        self._stratified_bins = stratified_bins
        self._stratified_gamma_start = stratified_gamma_start
        self._stratified_gamma_end = stratified_gamma_end
        if stratified_cycle_length is None:
            self._stratified_cycle_length = iterations
        else:
            self._stratified_cycle_length = min(stratified_cycle_length, iterations)
        self._intervention_decay = intervention_decay
        self._max_rel_ball_size = max_rel_ball_size
        if (max_rel_ball_size_explore is None) != (max_rel_ball_size_exploit is None):
            raise ValueError(
                "max_rel_ball_size_explore and max_rel_ball_size_exploit must be "
                "either both set or both None (they only make sense as a pair)."
            )
        self._max_rel_ball_size_explore = max_rel_ball_size_explore
        self._max_rel_ball_size_exploit = max_rel_ball_size_exploit
        self._normalize_inputs = bool(normalize_inputs)
        self._circuit_min_instances = circuit_min_instances
        self._circuit_rdc_threshold = circuit_rdc_threshold
        self._circuit_k_clusters = circuit_k_clusters
        self._log_runtime = log_runtime
        self._seed = seed
        self._rand_state = np.random.RandomState(seed)
        self._abort_after_intervention = abort_after_intervention
        self._disco_score_corrected = disco_score_corrected
        self._hybrid_pool_uniform_frac = float(hybrid_pool_uniform_frac)
        self._condition_perf_quantile_lo = float(condition_perf_quantile_lo)

        self.circuit = None  # the circuit learned in the last iteration
        self.evaluations = []
        self.runtimes = []  # the optimizer's runtime per iteration, if log_runtime
        self.data = None  # Joint (HPs + performance) numpy array, shape (N, D+1)
        self._curr_iter = 0
        self._iteration = 0  # number of DisCoMBO iterations proposed by ask()
        self._pending = None  # (configs, circuit-space rows or None, iteration) of the last ask()
        self._stopped = False  # set when abort_after_intervention stops the optimization
        self._intervene = False  # an intervention is active (its influence decays over the iterations)

        self.transform = ConfigurationNumericalTransform(search_space)
        self._build_normalizer()
        self._construct_search_space_metadata()

        # Interventions (see `intervene`)
        self._interventions = None
        self._intervention_iters = None
        self._intervention = None
        self._intervention_acceptance_prob = 1  # decays by intervention_decay per intervened iteration
        self._exploit_override_remaining = 0  # iterations left in post-intervention exploitation clamp
        self._disco_scores: List[Dict] = []  # one record per applied intervention, see `get_disco_scores`

    def optimize(self):
        """
        Main optimization loop: the initial random design (skipped if observations were already passed to
        `tell`) followed by `iterations` DisCoMBO iterations, each evaluating `num_samples` configurations.
        Selects candidates by their DisCo scores (discombo.circuits.pvalues).

        Returns:
            tuple: (best_config_dict, best_EvaluationResult)
        """
        if self.data is None:
            configs = self.ask()
            self.tell(configs, [self.objective(cfg) for cfg in configs])
        while self._iteration < self._num_iterations:
            configs = self.ask()
            if not configs:  # stopped by abort_after_intervention
                break
            self.tell(configs, [self.objective(cfg) for cfg in configs])
            logger.info("Iteration %d/%d: best observed value %.6g", self._iteration, self._num_iterations,
                        self.data[:, -1].max())
        return self.best()

    def ask(self) -> List[Dict]:
        """
        Propose the next batch of configurations to evaluate (ask-and-tell interface).

        The first call returns `initial_samples` random configurations, unless observations were already passed
        to `tell`. Every further call performs one DisCoMBO iteration: it learns a circuit on all observations
        told so far, applies the intervention scheduled for this iteration (see `intervene`) and returns
        `num_samples` configurations. Returns an empty list once `abort_after_intervention` stopped the
        optimization.

        Returns:
            list: Configurations (dicts mapping hyperparameter names to values).
        """
        if self._stopped:
            return []
        if self.data is None:
            configs = self.search_space.sample(size=self._initial_samples, rng=self._rand_state)
            self._pending = (configs, None, self._iteration)
            return list(configs)

        i = self._iteration
        self._curr_iter = i
        start_time = time() if self._log_runtime else None
        self._learn_circuit()

        if self._intervention_iters is not None and self._interventions is not None:
            if i in self._intervention_iters:
                int_idx = self._intervention_iters.index(i)
                logger.info("Iteration %d: applying intervention %d: %s", i + 1, int_idx,
                            self._interventions[int_idx])
                self._intervention = self._interventions[int_idx]
                self._intervene = self._intervention is not None
                if self._intervene:
                    evidence = self._apply_intervention(size=self._samples_per_iter)

                    # Condition on best observed performance: we want
                    # P(HPs | perf = best), not the joint marginal P(HPs).
                    # `condition` removes the perf leaves and replaces
                    # sum-node weights by their posterior given best_perf.
                    n_dims = self.data.shape[1]
                    perf_evidence = np.full((1, n_dims), np.nan)
                    perf_evidence[0, -1] = self._get_conditioning_value()
                    try:
                        conditioned = condition(self.circuit, perf_evidence)
                    except ConditioningError:
                        logger.warning(
                            "Iteration %d: conditioning on the best observed value failed (sum-node "
                            "normalizer underflow); computing the DisCo score of the intervention w.r.t. the "
                            "joint distribution instead.", i + 1
                        )
                        conditioned = self.circuit

                    scores = pvalues(
                        conditioned, evidence, side="both",
                        corrected=self._disco_score_corrected,
                    )
                    disco_score = float(np.clip(np.mean(scores), 0.0, 1.0))
                    self._exploit_override_remaining = 5
                    self._disco_scores.append({
                        "iter": i,
                        "intervention_idx": int_idx,
                        "disco_score": disco_score,
                        "sample_disco_scores": np.atleast_1d(scores).astype(float).tolist(),
                    })
                    logger.info("Iteration %d: DisCo score of intervention %d: %.4g", i + 1, int_idx, disco_score)
                    if self._abort_after_intervention:
                        logger.info("Iteration %d: abort_after_intervention is set, stopping after "
                                    "computing the DisCo score of the intervention.", i + 1)
                        self._stopped = True
                        return []

        sampled = self._sample(intervene=self._intervene)
        if self._log_runtime:
            self.runtimes.append(time() - start_time)
        if self._exploit_override_remaining > 0:
            self._exploit_override_remaining -= 1
        self._iteration += 1

        # `sampled` is in the circuit's (possibly normalized) space; keep these rows for `tell` and
        # inverse-scale them to obtain the configurations.
        rows = [list(row) for row in sampled]
        configs = []
        for row in rows:
            raw = list(self._scale_from_hp(np.array([row], dtype=float))[0]) if self._normalize_inputs else row
            configs.append(self.transform.inv_transform_configuration(raw))
        self._pending = (configs, rows, i)
        return list(configs)

    def tell(self, configs: List[Dict], results) -> None:
        """
        Report evaluation results (ask-and-tell interface).

        Args:
            configs (list): Evaluated configurations, either as returned by `ask` or any other configurations of
                the search space (e.g. evaluated before the optimization).
            results (list): One result per configuration: an EvaluationResult (its validation performance is
                maximized), a number (maximized), or None for a failed evaluation, which is ignored.
        """
        if len(configs) != len(results):
            raise ValueError(f"Got {len(configs)} configurations but {len(results)} results.")
        pending_configs, pending_rows, pending_iter = self._pending or ([], None, self._iteration)
        rows = []
        for cfg, result in zip(configs, results):
            if result is None:
                continue
            if not isinstance(result, EvaluationResult):
                result = EvaluationResult(float(result), float(result), float(result))
            k = next((k for k, proposed in enumerate(pending_configs) if proposed is cfg), None)
            if k is not None and pending_rows is not None:
                rows.append(pending_rows[k] + [result.val_performance])
            else:
                features = self.transform.transform_configuration(cfg) + [result.val_performance]
                rows.append(list(self._scale_to_full(np.array([features], dtype=float))[0]))
            self.evaluations.append((cfg, result, pending_iter if k is not None else self._iteration))
        if rows:
            rows = np.array(rows, dtype=float)
            self.data = rows if self.data is None else np.concatenate((self.data, rows))

    def best(self):
        """
        Returns:
            tuple: The best observed (configuration, EvaluationResult).
        """
        if not self.evaluations:
            raise RuntimeError("No observations yet.")
        val_performances = [e.val_performance for _, e, _ in self.evaluations]
        max_idx = int(np.argmax(val_performances))
        cfg, res = self.evaluations[max_idx][:2]
        return cfg, res

    @property
    def iteration(self) -> int:
        """Number of DisCoMBO iterations proposed by `ask` so far (the initial random design not counted)."""
        return self._iteration

    def intervene(self, interventions: List[Dict], iters):
        """
        Schedule interventions, i.e. expert beliefs about good configurations.

        Args:
            interventions (list): One dict per intervention, mapping hyperparameter names to a value (a point
                intervention; discrete hyperparameters by the index of the value in 'allowed', log-scaled ones in
                log space) or to a distribution {'dist': 'gauss' | 'uniform' | 'int_uniform' | 'cat',
                'parameters': [...]}.
            iters (list): The DisCoMBO iteration in which each intervention is applied.
        """
        self._interventions = interventions
        self._intervention_iters = iters

    def get_disco_scores(self) -> List[Dict]:
        """
        Return the distributional conformance (DisCo) score of each applied intervention.

        Returns:
            list: One record per applied intervention with the keys "iter" (the DisCoMBO iteration),
                "intervention_idx" (the index of the intervention passed to `intervene`), "disco_score" (the mean
                DisCo score of the beliefs w.r.t. the circuit conditioned on the best observed value; low scores
                indicate that the beliefs contradict the observations) and "sample_disco_scores" (the DisCo scores
                of the individual samples of the beliefs).
        """
        return [dict(record) for record in self._disco_scores]

    def get_intervention_pval_log(self) -> List[Dict]:
        """Deprecated alias of `get_disco_scores` with the record keys "pval_mean" and "pval_vec"."""
        warn_deprecated("get_intervention_pval_log()", "get_disco_scores()")
        return [{"iter": r["iter"], "intervention_idx": r["intervention_idx"], "pval_mean": r["disco_score"],
                 "pval_vec": r["sample_disco_scores"]} for r in self._disco_scores]

    @property
    def curr_mc(self):
        """Deprecated alias of `circuit`."""
        warn_deprecated("curr_mc", "circuit")
        return self.circuit

    @property
    def hpo_runtimes(self):
        """Deprecated alias of `runtimes`."""
        warn_deprecated("hpo_runtimes", "runtimes")
        return self.runtimes

    def __setstate__(self, state):
        """Restore a pickled optimizer, migrating deprecated attributes."""
        state = dict(state)
        for old, new in _RENAMED_ATTRIBUTES.items():
            if old in state:
                state[new] = state.pop(old)
        if '_intervention_pval_log' in state:
            state['_disco_scores'] = [
                {"iter": r["iter"], "intervention_idx": r["intervention_idx"], "disco_score": r["pval_mean"],
                 "sample_disco_scores": r["pval_vec"]} for r in state.pop('_intervention_pval_log')]
        state.pop('_exploration_method', None)
        state.pop('_mc_type', None)
        self.__dict__.update(state)

    def _learn_circuit(self):
        """Learn a circuit with per-variable parametric leaves.

        Maps each HP's dtype to a leaf distribution (see circuits.parametric_specs):
          float        -> Normal
          int          -> Bernoulli (2 values), Normal (>= 10 values), else Categorical
          str/cat      -> Categorical
          bool         -> Bernoulli
        The performance column is always Normal.
        """
        specs = parametric_specs(self.search_space.get_search_space_definition(), int_real_threshold=10)
        min_inst = min(self._circuit_min_instances, max(1, self.data.shape[0] // 3))
        try:
            self.circuit = learn_circuit(
                self.data,
                specs,
                min_instances_slice=min_inst,
                threshold=self._circuit_rdc_threshold,
                n_clusters=self._circuit_k_clusters,
            )
        except Exception:
            self.circuit = learn_circuit(
                self.data,
                specs,
                min_instances_slice=max(1, self.data.shape[0] // 2),
                threshold=0.5,
                n_clusters=self._circuit_k_clusters,
            )

    def _get_conditioning_value(self):
        """Return the best observed performance value."""
        return self.data[:, -1].max()

    def _uniform_random_hp_pool(self, size):
        """Sample `size` configs uniformly at random from the search space and
        transform them into the circuit's training feature space.

        Returns: np.ndarray of shape (size, n_dims - 1) — HP cols only,
        scaled to match the circuit's input space when normalize_inputs=True.
        """
        configs = self.search_space.sample(size=size, rng=self._rand_state)
        rows = [self.transform.transform_configuration(cfg) for cfg in configs]
        arr = np.asarray(rows, dtype=float)
        # Build a (size, n_dims) matrix with NaN perf so _scale_to_full
        # can apply the same min-max scaling used for the training data.
        # NaN propagates harmlessly through the scaling for the perf col.
        full = np.concatenate([arr, np.full((size, 1), np.nan)], axis=1)
        full = self._scale_to_full(full)
        return full[:, :-1]

    def _apply_point_intervention(self, data):
        """Apply a point intervention to conditioning data."""
        names = list(self.search_space.get_search_space_definition())
        keys = list(self._intervention.keys())
        if self._intervention is not None:
            for idx, key in enumerate(names):
                if key in keys:
                    data[idx] = self._intervention[key]
        return data

    def _apply_intervention(self, size=None):
        """Sample conditioning data from the active intervention (values and distributions)."""
        if size is None:
            size = self._samples_per_iter

        rng = self._rand_state

        def sample_from_condition_prior(intervention_key):
            if self._intervention[intervention_key]['dist'] == 'gauss':
                mu, std = self._intervention[intervention_key]['parameters']
                samples = rng.normal(mu, std, size)
            elif self._intervention[intervention_key]['dist'] == 'uniform':
                s, e = self._intervention[intervention_key]['parameters']
                samples = rng.uniform(s, e, size)
            elif self._intervention[intervention_key]['dist'] == 'int_uniform':
                s, e = self._intervention[intervention_key]['parameters']
                samples = rng.randint(s, e, size)
            elif self._intervention[intervention_key]['dist'] == 'cat':
                weights = self._intervention[intervention_key]['parameters']
                probs = np.array(weights) / np.sum(weights)
                samples = rng.choice(np.arange(len(probs)), size, p=probs)
            return samples

        names = list(self.search_space.get_search_space_definition())
        keys = list(self._intervention.keys())
        cond_samples = np.array(
            [np.nan] * size * (len(names) + 1)
        ).reshape(-1, len(names) + 1)
        if self._intervention is not None:
            for idx, key in enumerate(names):
                if key in keys:
                    if isinstance(self._intervention[key], dict):
                        cond_samples[:, idx] = sample_from_condition_prior(key)
                    else:
                        cond_samples[:, idx] = np.repeat(self._intervention[key], size)
        # Intervention values are specified in raw HP space; scale to match
        # the circuit's space when input normalization is on. NaNs propagate.
        if self._normalize_inputs:
            cond_samples = self._scale_to_full(cond_samples)
        return cond_samples

    def _sample_stratified(self, n_samples, intervene=False):
        """
        Stratified DisCo score sampling from a performance-conditioned circuit pool.

        Draws a large pool from the circuit conditioned on the best observed
        performance, computes DisCo scores on the HP dimensions only, then
        selects candidates via equal-count bin stratification.

        Equal-count (quantile) bins are used rather than equal-width bins.
        This is robust to conservative DisCo scores in deep circuits:
        regardless of whether the pool's scores cluster at [0.4, 1.0] or
        [0.3, 0.95], each bin always contains the same number of candidates
        and the selected set is approximately uniform over the *relative*
        novelty spectrum of the pool.

        Returns:
            numpy array of shape (n_samples, D) — HP configs only.
        """
        n_dims = self.data.shape[1]
        best_perf = self._get_conditioning_value()
        pool_size = self._stratified_pool_size

        # Hybrid pool: split between circuit-conditioned (with range conditioning)
        # and uniform-random over the search space. The uniform half guarantees
        # the explore-end of γ stratification has access to genuine global
        # outliers regardless of how localised the circuit has become.
        # During an active intervention we keep the full pool conditioned so
        # intervention semantics are preserved.
        if intervene:
            accept_intervention = self._rand_state.choice(
                [1, 0], p=[self._intervention_acceptance_prob, 1 - self._intervention_acceptance_prob])
            self._intervention_acceptance_prob *= self._intervention_decay
        else:
            accept_intervention = 0
        intervention_active = bool(intervene and accept_intervention == 1)

        n_uniform = 0
        if not intervention_active and self._hybrid_pool_uniform_frac > 0.0:
            n_uniform = int(round(pool_size * self._hybrid_pool_uniform_frac))
            n_uniform = max(0, min(n_uniform, pool_size - 1))
        n_circuit = pool_size - n_uniform

        # --- circuit-conditioned half (range-conditioning on perf) ---
        cond_array = np.full((n_circuit, n_dims), np.nan)
        # Range conditioning: per-row uniform draw from [q_lo, max(perf)]. This
        # lets every leaf whose perf-mean lies in [q_lo, max] contribute a
        # share of pool samples, instead of collapsing onto the single leaf
        # closest to max.
        q_lo = float(np.quantile(self.data[:, -1], self._condition_perf_quantile_lo))
        if q_lo >= best_perf:
            cond_array[:, -1] = best_perf
        else:
            cond_array[:, -1] = self._rand_state.uniform(q_lo, best_perf, size=n_circuit)

        if intervention_active:
            cond_array = self._apply_intervention(size=n_circuit)
            # Re-apply (random) perf condition; _apply_intervention left perf as NaN.
            if q_lo >= best_perf:
                cond_array[:, -1] = best_perf
            else:
                cond_array[:, -1] = self._rand_state.uniform(q_lo, best_perf, size=n_circuit)

        if not np.isnan(cond_array).any():
            circuit_candidates = cond_array
        else:
            circuit_candidates = sample(self.circuit, cond_array, seed=self._rand_state.randint(2**31 - 1))
        circuit_hp = circuit_candidates[:, :-1]

        # --- Uniform-random half (only when not intervening) ---
        if n_uniform > 0:
            uniform_hp = self._uniform_random_hp_pool(n_uniform)
            hp_pool = np.concatenate([circuit_hp, uniform_hp], axis=0)
        else:
            hp_pool = circuit_hp

        # --- Compute DisCo scores on HP dimensions (marginalize performance) ---
        full_evidence = np.full((pool_size, n_dims), np.nan)
        full_evidence[:, :-1] = hp_pool
        scores = pvalues(self.circuit, full_evidence, side="both")

        # --- Stratified selection: B power-law bins by DisCo score rank ---
        # Bin b covers pool positions [q_b, q_{b+1}] where q_b = (b/B)^gamma.
        # gamma > 1: tiny bins at the low-score end → high sampling density there (exploration).
        # gamma < 1: tiny bins at the high-score end → high sampling density there (exploitation).
        # gamma follows a sinusoidal cycle: starts at gamma_start (explore), reaches gamma_end
        # (exploit) after cycle_length iterations, then returns to gamma_start, etc.
        B = min(self._stratified_bins, pool_size // 2, n_samples)
        B = max(B, 1)
        sorted_idx = np.argsort(scores)

        gamma = self._current_gamma()
        boundaries = [int((b / B) ** gamma * pool_size) for b in range(B + 1)]
        boundaries[-1] = pool_size  # clamp final boundary

        per_bin = n_samples // B
        remainder = n_samples % B

        selected = []
        for b in range(B):
            bin_idx = sorted_idx[boundaries[b]:boundaries[b + 1]]
            if len(bin_idx) == 0:
                remainder += per_bin  # empty bin: roll quota to remainder pool
                continue
            n_pick = per_bin + (1 if b < remainder else 0)
            n_pick = min(n_pick, len(bin_idx))
            if n_pick > 0:
                picks = self._rand_state.choice(bin_idx, size=n_pick, replace=False)
                selected.extend(picks)

        # Pad if empty bins consumed too many quotas
        if len(selected) < n_samples:
            already = set(selected)
            remaining = [i for i in range(pool_size) if i not in already]
            n_pad = n_samples - len(selected)
            if remaining:
                pad = self._rand_state.choice(remaining, size=min(n_pad, len(remaining)), replace=False)
                selected.extend(pad)

        selected = np.array(selected[:n_samples])
        return hp_pool[selected]

    def _sample(self, intervene=False):
        """
        Sample candidates by stratified DisCo score selection (see `_sample_stratified`) and apply ball sampling
        perturbation and clamping to all candidates.
        """
        sampled_configs_np = self._sample_stratified(self._samples_per_iter, intervene)

        # Ball sampling perturbation. When the explore/exploit pair is set,
        # the per-dim radius tracks the same cosine γ schedule that drives
        # bin power-law selection: large jitter while exploring, tight
        # jitter while exploiting.
        if self._max_rel_ball_size_explore is not None:
            gamma = self._current_gamma()
            g_min = self._stratified_gamma_end
            g_max = self._stratified_gamma_start
            g_norm = (gamma - g_min) / max(g_max - g_min, 1e-9)   # 0=exploit, 1=explore
            rel_t = (
                self._max_rel_ball_size_exploit
                + g_norm * (self._max_rel_ball_size_explore - self._max_rel_ball_size_exploit)
            )
            active_radius = self._ball_radius_for(rel_t)
        else:
            active_radius = self.ball_radius
        sampled_configs_np = sample_from_ball(
            sampled_configs_np, self.search_space, active_radius,
            sampled_configs_np.shape[0], rng=self._rand_state
        )

        # Clamp to search space bounds
        for i, (_, v) in enumerate(self.search_space.get_search_space_definition().items()):
            if v['dtype'] == 'float':
                if self._normalize_inputs:
                    # Float HP cols are in [0,1] scaled space.
                    lo, hi = 0.0, 1.0
                else:
                    # Data is stored log-transformed for log HPs, so bounds must also
                    # be in log space to avoid collapsing samples to exp(v['min']).
                    lo = np.log(v['min']) if v.get('is_log') else v['min']
                    hi = np.log(v['max']) if v.get('is_log') else v['max']
                for j in range(sampled_configs_np.shape[0]):
                    sampled_configs_np[j, i] = max(sampled_configs_np[j, i], lo)
                    sampled_configs_np[j, i] = min(sampled_configs_np[j, i], hi)
            elif 'allowed' in v:
                n = len(v['allowed'])
                sampled_configs_np[:, i] = np.clip(np.round(sampled_configs_np[:, i]), 0, n - 1)

        return sampled_configs_np

    def _construct_search_space_metadata(self):
        """Build ball radius metadata for perturbation sampling."""
        # Cached per-iter-constant radius for the legacy flat-radius code path.
        # γ-coupled callers recompute via _ball_radius_for(rel_t).
        self.ball_radius = self._ball_radius_for(self._max_rel_ball_size)

    def _ball_radius_for(self, rel):
        """Per-dim ball radius vector for the given relative value(s).

        rel: scalar or list/array of length n_hp_dims.
        When `_normalize_inputs` is True, float dims have unit width in
        scaled space, so the per-dim radius equals `rel` directly.
        Otherwise the radius is `rel × (hi - lo)` per float dim, with
        log-bounds when is_log=True.
        """
        search_space_definition = self.search_space.get_search_space_definition()
        radii = []
        for idx, (_, hp_def) in enumerate(search_space_definition.items()):
            rel_rad = rel[idx] if isinstance(rel, (list, np.ndarray)) else rel
            if hp_def['dtype'] == 'float':
                if self._normalize_inputs:
                    radii.append(float(rel_rad))
                elif hp_def.get('is_log'):
                    radii.append(float((np.log(hp_def['max']) - np.log(hp_def['min'])) * rel_rad))
                else:
                    radii.append(float((hp_def['max'] - hp_def['min']) * rel_rad))
            else:
                radii.append(float(rel_rad))
        return radii

    def _current_gamma(self):
        """Current γ value from the cosine cycle, with the post-intervention exploit clamp."""
        if self._exploit_override_remaining > 0:
            return self._stratified_gamma_end
        gamma_mid = (self._stratified_gamma_start + self._stratified_gamma_end) / 2
        gamma_amp = (self._stratified_gamma_start - self._stratified_gamma_end) / 2
        return gamma_mid + gamma_amp * np.cos(
            np.pi * self._curr_iter / self._stratified_cycle_length
        )

    def _build_normalizer(self):
        """Per-column min-max bounds for the joint (HPs + perf) matrix.

        Float HP cols get [lo, hi] from search-space bounds (log when is_log=True).
        Categorical/discrete cols and the perf col get identity (lo=0, hi=1, is_float=False).
        """
        search_space_definition = self.search_space.get_search_space_definition()
        n_dims = len(search_space_definition) + 1   # + perf col
        lo = np.zeros(n_dims, dtype=float)
        hi = np.ones(n_dims, dtype=float)
        is_float = np.zeros(n_dims, dtype=bool)
        for idx, (_, hp_def) in enumerate(search_space_definition.items()):
            if hp_def.get('dtype') == 'float':
                is_float[idx] = True
                if hp_def.get('is_log'):
                    lo[idx] = float(np.log(hp_def['min']))
                    hi[idx] = float(np.log(hp_def['max']))
                else:
                    lo[idx] = float(hp_def['min'])
                    hi[idx] = float(hp_def['max'])
        self._scale_lo = lo
        self._scale_hi = hi
        self._scale_is_float = is_float

    def _scale_to_full(self, arr):
        """Min-max scale float HP cols of (N, n_dims) matrix to [0,1]. Other cols untouched."""
        if not self._normalize_inputs or not self._scale_is_float.any():
            return arr
        out = arr.copy()
        cols = self._scale_is_float
        width = self._scale_hi[cols] - self._scale_lo[cols]
        out[:, cols] = (arr[:, cols] - self._scale_lo[cols]) / np.maximum(width, 1e-12)
        return out

    def _scale_from_hp(self, hp_arr):
        """Inverse-scale float HP cols of (N, D) HP-only matrix back to raw bounds."""
        if not self._normalize_inputs:
            return hp_arr
        is_float_hp = self._scale_is_float[:-1]
        if not is_float_hp.any():
            return hp_arr
        out = hp_arr.copy()
        lo_hp = self._scale_lo[:-1][is_float_hp]
        hi_hp = self._scale_hi[:-1][is_float_hp]
        out[:, is_float_hp] = hp_arr[:, is_float_hp] * (hi_hp - lo_hp) + lo_hp
        return out

    @property
    def history(self):
        if self._log_runtime:
            evaluations = []
            optim_cost = np.sum(self.runtimes)
            for cfg, eval_res, it in self.evaluations:
                eval_res.set_optim_cost(optim_cost)
                evaluations.append((cfg, eval_res, it))
            return evaluations
        return self.evaluations

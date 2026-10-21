from typing import Dict, List
from .optimizer import Optimizer
from .pc_optimizer import sample_from_ball
from ..search_spaces import SearchSpace
from ..utils import ConfigurationNumericalTransform
import numpy as np
import warnings
from time import time

from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric
from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Condition import condition
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical, Bernoulli

_METATYPE_TO_PARAMETRIC = {
    MetaType.REAL: Gaussian,
    MetaType.DISCRETE: Categorical,
    MetaType.BINARY: Bernoulli,
}
from membership_circuits.models.nodewise.spflow_pvalue import pvalue_spflow

warnings.filterwarnings("ignore")

"""
    Implementation of the MC-HPO optimizer.
    Uses SPFlow's learn_mspn for structure learning and membership_circuits'
    pvalue_spflow for p-value computation (Fisher's method at product nodes,
    weighted corrected adaptive arithmetic mean at sum nodes).
"""


class MCOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100,
                 num_samples=20, initial_samples=20, exploration_method='none',
                 stratified_pool_size=1000, stratified_bins=10,
                 stratified_gamma_start=2.0, stratified_gamma_end=0.25,
                 stratified_cycle_length=20,
                 interaction_dist_sample_decay=0.9, max_rel_ball_size=0.05,
                 max_rel_ball_size_explore=None, max_rel_ball_size_exploit=None,
                 normalize_inputs=False,
                 mc_min_instances=10, mc_rdc_threshold=0.3, mc_k_clusters=2,
                 mc_type='mixed', log_hpo_runtime=False, seed=0,
                 abort_after_intervention=False,
                 intervention_pval_corrected=False,
                 hybrid_pool_uniform_frac=0.5,
                 condition_perf_quantile_lo=0.7) -> None:
        """
        MC-HPO Optimizer.

        Args:
            search_space: SearchSpace object defining the hyperparameter domain.
            objective: Objective function taking a config dict, returning BenchQueryResult.
            iterations: Number of optimization iterations.
            num_samples: Number of candidate configs per iteration.
            initial_samples: Number of random samples for initializing the MC.
            stratified_pool_size: Number of candidates in the large pool for stratified
                p-value sampling (exploration_method='stratified').
            stratified_bins: Number of p-value bins for stratified selection.
                Selected candidates are drawn uniformly from each bin to ensure
                approximately uniform p-value coverage.
            stratified_gamma_start: Power-law exponent at iteration 0 (>1 → exploration bias:
                tiny bins at low-p end → high sampling density at low p-values).
            stratified_gamma_end: Power-law exponent at exploitation phase (<1 → tiny bins at
                high-p end → high sampling density at high p-values).
            stratified_cycle_length: Half-period (in iterations) of the sinusoidal γ schedule.
                γ(t) = γ_mid + γ_amp·cos(π·t/cycle_length), cycling exploration→exploitation
                every cycle_length iterations (default 10: explore at t=0, exploit at t=10,
                explore again at t=20, …).
            interaction_dist_sample_decay: Decay factor for user interaction recovery.
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
                fitting the SPN. Categorical/discrete dims and the performance
                column are left untouched. The variance-floor for `make_min_variance_leaf`
                is also re-tuned (epsilon_abs=1e-6, epsilon_rel=0.1) so leaves can
                concentrate in the smaller normalized scale.
            mc_min_instances: Minimum instances per leaf in SPN structure learning.
            mc_rdc_threshold: RDC independence threshold for column partitioning.
            mc_k_clusters: Number of row clusters for structure learning.
            mc_type: SPN learning variant. One of:
                - 'mixed': Per-variable parametric leaves via learn_parametric
                  (Gaussian/Categorical/Bernoulli per MetaType). Handles mixed
                  search spaces correctly with parametric distributions.
                - 'mspn': Histogram leaves via learn_mspn — non-parametric,
                  also handles mixed types but less interpretable.
                - 'parametric': All-Gaussian leaves via learn_parametric — only
                  suitable for fully continuous search spaces.
            log_hpo_runtime: Whether to log optimizer runtime.
            seed: Random seed.
        """
        super().__init__(search_space, objective)
        self.search_space = search_space
        self.objective = objective
        self._num_iterations = iterations
        self._samples_per_iter = num_samples
        self._initial_samples = initial_samples
        self._exploration_method = exploration_method
        self._stratified_pool_size = stratified_pool_size
        self._stratified_bins = stratified_bins
        self._stratified_gamma_start = stratified_gamma_start
        self._stratified_gamma_end = stratified_gamma_end
        if stratified_cycle_length is None:
            self._stratified_cycle_length = iterations
        else:
            self._stratified_cycle_length = min(stratified_cycle_length, iterations)
        self._interaction_dist_sample_decay = interaction_dist_sample_decay
        self._max_rel_ball_size = max_rel_ball_size
        if (max_rel_ball_size_explore is None) != (max_rel_ball_size_exploit is None):
            raise ValueError(
                "max_rel_ball_size_explore and max_rel_ball_size_exploit must be "
                "either both set or both None (they only make sense as a pair)."
            )
        self._max_rel_ball_size_explore = max_rel_ball_size_explore
        self._max_rel_ball_size_exploit = max_rel_ball_size_exploit
        self._normalize_inputs = bool(normalize_inputs)
        self._mc_min_instances = mc_min_instances
        self._mc_rdc_threshold = mc_rdc_threshold
        self._mc_k_clusters = mc_k_clusters
        self._mc_type = mc_type
        self._log_hpo_runtime = log_hpo_runtime
        self._seed = seed
        self._rand_state = np.random.RandomState(seed)
        self._abort_after_intervention = abort_after_intervention
        self._intervention_pval_corrected = intervention_pval_corrected
        self._hybrid_pool_uniform_frac = float(hybrid_pool_uniform_frac)
        self._condition_perf_quantile_lo = float(condition_perf_quantile_lo)

        self.curr_mc = None
        self.evaluations = []
        self.hpo_runtimes = []
        self.data = None  # Joint (HPs + performance) numpy array, shape (N, D+1)
        self._curr_iter = 0

        self.transform = ConfigurationNumericalTransform(search_space)
        self._build_normalizer()
        self._construct_search_space_metadata()

        # Intervention support (same interface as PCOptimizer)
        self._interventions = None
        self._intervention_iters = None
        self._intervention = None
        self._prob_interaction_pc = 1
        self._exploit_override_remaining = 0  # iterations left in post-intervention exploitation clamp
        self._intervention_pval_log: List[Dict] = []

    def optimize(self):
        """
        Main optimization loop. Mirrors PCOptimizer's Algorithm 1 structure,
        using SPFlow SPNs and pvalue_spflow for p-value rejection sampling.

        Returns:
            tuple: (best_config_dict, best_BenchQueryResult)
        """
        intervene = False
        for i in range(self._num_iterations):
            self._curr_iter = i
            if i % 10 == 0:
                print(f"Iteration {i+1}/{self._num_iterations}")

            start_time = time() if self._log_hpo_runtime else None

            if self.curr_mc is None:
                self._learn_init_mc()
            else:
                self._learn_mc()

            if self._intervention_iters is not None and self._interventions is not None:
                if i in self._intervention_iters:
                    int_idx = self._intervention_iters.index(i)
                    print(f"[Optimizer][MC] Using Intervention {int_idx}: {self._interventions[int_idx]}")
                    self._intervention = self._interventions[int_idx]
                    intervene = True and self._intervention is not None
                    if intervene:
                        evidence = self._apply_intervention(size=self._samples_per_iter)

                        # Condition on best observed performance: we want
                        # P(HPs | perf = best), not the joint marginal P(HPs).
                        # SPFlow's `condition` consumes the perf leaves and
                        # rewrites sum-node weights by leaf likelihoods at
                        # best_perf.
                        n_dims = self.data.shape[1]
                        perf_evidence = np.full((1, n_dims), np.nan)
                        perf_evidence[0, -1] = self._get_conditioning_value()
                        try:
                            cond_mc = condition(self.curr_mc, perf_evidence)
                        except AssertionError:
                            print(
                                "[Optimizer][MC] condition() failed "
                                "(sum-node normalizer underflow); "
                                "falling back to joint marginal for p-value"
                            )
                            cond_mc = self.curr_mc

                        pvals = pvalue_spflow(
                            cond_mc, evidence, side="both",
                            corrected=self._intervention_pval_corrected,
                        ).squeeze()
                        pval_mean = float(np.clip(np.mean(pvals), 0.0, 1.0))
                        self._exploit_override_remaining = 5
                        self._intervention_pval_log.append({
                            "iter": i,
                            "intervention_idx": int_idx,
                            "pval_mean": pval_mean,
                            "pval_vec": np.atleast_1d(pvals).astype(float).tolist(),
                        })
                        if self._abort_after_intervention:
                            print(
                                f"[Optimizer][MC] abort_after_intervention set; "
                                f"stopping at iter {i} after capturing p-value."
                            )
                            break

            configs = self._sample(intervene=intervene)

            if self._log_hpo_runtime:
                self.hpo_runtimes.append(time() - start_time)

            self._evaluate(configs)
            if self._exploit_override_remaining > 0:
                self._exploit_override_remaining -= 1
            print(f"BEST: {self.data[:, -1].max()}")

        val_performances = [e.val_performance for _, e, _ in self.evaluations]
        max_idx = int(np.argmax(val_performances))
        cfg, res = self.evaluations[max_idx][:2]
        return cfg, res

    def intervene(self, interventions: List[Dict], iters):
        """Set interventions (same interface as PCOptimizer)."""
        self._interventions = interventions
        self._intervention_iters = iters

    def get_intervention_pval_log(self) -> List[Dict]:
        """Return per-intervention p-value records captured during optimize()."""
        return list(self._intervention_pval_log)

    def _build_dataset(self, configs, scores):
        """Build joint (HPs + performance) data matrix.

        Float HP cols are min-max scaled to [0,1] when `_normalize_inputs=True`.
        """
        data_matrix = []
        for cfg, score in zip(configs, scores):
            features = self.transform.transform_configuration(cfg)
            features.append(score.val_performance)
            data_matrix.append(features)
            self.evaluations.append((cfg, score, self._curr_iter))
        arr = np.array(data_matrix, dtype=float)
        return self._scale_to_full(arr)

    def _learn_init_mc(self):
        """Sample initial random configs, evaluate, and learn the first SPN."""
        configs = self.search_space.sample(size=self._initial_samples)
        scores = [self.objective(cfg) for cfg in configs]
        self.data = self._build_dataset(configs, scores)
        self._learn_mc()

    def _learn_mc(self):
        """Dispatch to the appropriate SPN learning variant based on mc_type."""
        if self._mc_type == 'mixed':
            self._learn_mc_mixed()
        else:
            raise ValueError(f"Only learn mixed SPNs right now, but got passed _mc_type: {self._mc_type}")

    def _learn_mc_mixed(self):
        """Learn a parametric SPN with per-variable leaf types via learn_parametric.

        Maps each HP's dtype to the appropriate MetaType and parametric distribution:
          float        -> REAL     -> Gaussian
          int/str/cat  -> DISCRETE -> Categorical
          bool         -> BINARY   -> Bernoulli
        The performance column is always Gaussian (REAL).

        Note: derives types from val['dtype'] (a plain string) rather than
        val['type'] (a MetaType enum that may come from a different module
        instance), ensuring the MetaType objects match _METATYPE_TO_PARAMETRIC.
        """
        search_space_definition = self.search_space.get_search_space_definition()
        meta_types = []
        param_types = []
        for val in search_space_definition.values():
            dtype = val.get('dtype', 'float')
            if dtype == 'float':
                meta_types.append(MetaType.REAL)
                param_types.append(Gaussian)
            elif dtype == 'int':
                if 'allowed' in val and val['allowed'] is not None:
                    n_unique = len(val['allowed'])
                else:
                    n_unique = int(val['max']) - int(val['min']) + 1
                if n_unique == 2:
                    meta_types.append(MetaType.BINARY)
                    param_types.append(Bernoulli)
                elif n_unique >= 10:
                    meta_types.append(MetaType.REAL)
                    param_types.append(Gaussian)
                else:
                    meta_types.append(MetaType.DISCRETE)
                    param_types.append(Categorical)
            elif dtype in ('str', 'cat'):
                meta_types.append(MetaType.DISCRETE)
                param_types.append(Categorical)
            elif dtype == 'bool':
                meta_types.append(MetaType.BINARY)
                param_types.append(Bernoulli)
            else:
                meta_types.append(MetaType.REAL)
                param_types.append(Gaussian)
        meta_types.append(MetaType.REAL)   # performance column
        param_types.append(Gaussian)

        ctxt = Context(meta_types=meta_types, parametric_types=param_types)
        ctxt.add_domains(self.data)

        min_inst = min(self._mc_min_instances, max(1, self.data.shape[0] // 3))
        try:
            self.curr_mc = learn_parametric(
                self.data,
                ctxt,
                min_instances_slice=min_inst,
                threshold=self._mc_rdc_threshold,
                # leaves=leaves,
            )
        except (ValueError, Exception):
            ctxt.add_domains(self.data)
            self.curr_mc = learn_parametric(
                self.data,
                ctxt,
                min_instances_slice=max(1, self.data.shape[0] // 2),
                threshold=0.5,
                # leaves=leaves,
            )

    def _get_conditioning_value(self):
        """Return the best observed performance value."""
        return self.data[:, -1].max()

    def _uniform_random_hp_pool(self, size):
        """Sample `size` configs uniformly at random from the search space and
        transform them into the SPN's training feature space.

        Returns: np.ndarray of shape (size, n_dims - 1) — HP cols only,
        scaled to match the SPN's input space when normalize_inputs=True.
        """
        configs = self.search_space.sample(size=size)
        rows = [self.transform.transform_configuration(cfg) for cfg in configs]
        arr = np.asarray(rows, dtype=float)
        # Build a (size, n_dims) matrix with NaN perf so _scale_to_full
        # can apply the same min-max scaling used for the training data.
        # NaN propagates harmlessly through the scaling for the perf col.
        full = np.concatenate([arr, np.full((size, 1), np.nan)], axis=1)
        full = self._scale_to_full(full)
        return full[:, :-1]

    def _apply_point_intervention(self, data):
        """Apply point intervention to conditioning data (same as PCOptimizer)."""
        prot_cfg = self.search_space.sample(keep_domain=False, size=1)[0]
        keys = list(self._intervention.keys())
        if self._intervention is not None:
            for idx, key in enumerate(prot_cfg.keys()):
                if key in keys:
                    data[idx] = self._intervention[key]
        return data

    def _apply_intervention(self, size=None):
        """Apply distribution-based intervention (same as PCOptimizer)."""
        if size is None:
            size = self._samples_per_iter

        def sample_from_condition_prior(intervention_key):
            if self._intervention[intervention_key]['dist'] == 'gauss':
                mu, std = self._intervention[intervention_key]['parameters']
                samples = np.random.normal(mu, std, size)
            elif self._intervention[intervention_key]['dist'] == 'uniform':
                s, e = self._intervention[intervention_key]['parameters']
                samples = np.random.uniform(s, e, size)
            elif self._intervention[intervention_key]['dist'] == 'int_uniform':
                s, e = self._intervention[intervention_key]['parameters']
                samples = np.random.randint(s, e, size)
            elif self._intervention[intervention_key]['dist'] == 'cat':
                weights = self._intervention[intervention_key]['parameters']
                probs = np.array(weights) / np.sum(weights)
                samples = np.random.choice(np.arange(len(probs)), size, p=probs)
            return samples

        prot_cfg = self.search_space.sample(keep_domain=False, size=1)[0]
        keys = list(self._intervention.keys())
        cond_samples = np.array(
            [np.nan] * size * (len(prot_cfg) + 1)
        ).reshape(-1, len(prot_cfg) + 1)
        if self._intervention is not None:
            for idx, key in enumerate(prot_cfg.keys()):
                if key in keys:
                    if isinstance(self._intervention[key], dict):
                        cond_samples[:, idx] = sample_from_condition_prior(key)
                    else:
                        cond_samples[:, idx] = np.repeat(self._intervention[key], size)
        # Intervention values are specified in raw HP space; scale to match
        # the SPN's space when input normalization is on. NaNs propagate.
        if self._normalize_inputs:
            cond_samples = self._scale_to_full(cond_samples)
        return cond_samples

    def _sample_stratified(self, n_samples, intervene=False):
        """
        Stratified p-value sampling from a performance-conditioned SPN pool.

        Draws a large pool from the SPN conditioned on the best observed
        performance, computes p-values on the HP dimensions only, then
        selects candidates via equal-count bin stratification.

        Equal-count (quantile) bins are used rather than equal-width bins.
        This is robust to the conservative p-value problem in deep SPNs:
        regardless of whether the pool p-values cluster at [0.4, 1.0] or
        [0.3, 0.95], each bin always contains the same number of candidates
        and the selected set is approximately uniform over the *relative*
        novelty spectrum of the pool.

        Returns:
            numpy array of shape (n_samples, D) — HP configs only.
        """
        n_dims = self.data.shape[1]
        best_perf = self._get_conditioning_value()
        pool_size = self._stratified_pool_size

        # Hybrid pool: split between SPN-conditioned (with range conditioning)
        # and uniform-random over the search space. The uniform half guarantees
        # the explore-end of γ stratification has access to genuine global
        # outliers regardless of how localised the SPN has become.
        # During an active intervention we keep the full pool conditioned so
        # intervention semantics are preserved.
        if intervene:
            accept_intervention = np.random.choice(
                [1, 0], p=[self._prob_interaction_pc, 1 - self._prob_interaction_pc])
            self._prob_interaction_pc *= self._interaction_dist_sample_decay
        else:
            accept_intervention = 0
        intervention_active = bool(intervene and accept_intervention == 1)

        n_uniform = 0
        if not intervention_active and self._hybrid_pool_uniform_frac > 0.0:
            n_uniform = int(round(pool_size * self._hybrid_pool_uniform_frac))
            n_uniform = max(0, min(n_uniform, pool_size - 1))
        n_spn = pool_size - n_uniform

        # --- SPN-conditioned half (range-conditioning on perf) ---
        cond_array = np.full((n_spn, n_dims), np.nan)
        # Range conditioning: per-row uniform draw from [q_lo, max(perf)]. This
        # lets every leaf whose perf-mean lies in [q_lo, max] contribute a
        # share of pool samples, instead of collapsing onto the single leaf
        # closest to max.
        q_lo = float(np.quantile(self.data[:, -1], self._condition_perf_quantile_lo))
        if q_lo >= best_perf:
            cond_array[:, -1] = best_perf
        else:
            cond_array[:, -1] = self._rand_state.uniform(q_lo, best_perf, size=n_spn)

        if intervention_active:
            cond_array = self._apply_intervention(size=n_spn)
            # Re-apply (random) perf condition; _apply_intervention left perf as NaN.
            if q_lo >= best_perf:
                cond_array[:, -1] = best_perf
            else:
                cond_array[:, -1] = self._rand_state.uniform(q_lo, best_perf, size=n_spn)

        if not np.isnan(cond_array).any():
            spn_candidates = cond_array
        else:
            spn_candidates = sample_instances(self.curr_mc, cond_array, self._rand_state)
        spn_hp = spn_candidates[:, :-1]

        # --- Uniform-random half (only when not intervening) ---
        if n_uniform > 0:
            uniform_hp = self._uniform_random_hp_pool(n_uniform)
            hp_pool = np.concatenate([spn_hp, uniform_hp], axis=0)
        else:
            hp_pool = spn_hp

        # --- Compute p-values on HP dimensions (marginalize performance) ---
        full_evidence = np.full((pool_size, n_dims), np.nan)
        full_evidence[:, :-1] = hp_pool
        pvals = pvalue_spflow(self.curr_mc, full_evidence, side="both").squeeze()

        # --- Stratified selection: B power-law bins by p-value rank ---
        # Bin b covers pool positions [q_b, q_{b+1}] where q_b = (b/B)^gamma.
        # gamma > 1: tiny bins at low-p end → high sampling density there (exploration).
        # gamma < 1: tiny bins at high-p end → high sampling density there (exploitation).
        # gamma follows a sinusoidal cycle: starts at gamma_start (explore), reaches gamma_end
        # (exploit) after cycle_length iterations, then returns to gamma_start, etc.
        B = min(self._stratified_bins, pool_size // 2, n_samples)
        B = max(B, 1)
        sorted_idx = np.argsort(pvals)

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
        Sample candidates. Dispatches based on exploration_method:
        - 'none': pure conditioned sampling (mc_plain)
        - 'pvalue_filter': conditioned sampling + p-value binary filter
        - 'ood_uniform': split budget between conditioned sampling and OOD
        - 'ood_antidensity': split budget between conditioned and anti-density OOD
        - 'rejection': p-value rejection sampling on conditioned candidates
        - 'stratified': stratified p-value selection for uniform p-value coverage
        Applies ball sampling perturbation and clamping to all candidates.
        """
        if self._exploration_method == 'stratified':
            sampled_configs_np = self._sample_stratified(self._samples_per_iter, intervene)
        else:
            raise ValueError(f"Only support stratified sampling, but found _exploration_method: {self._exploration_method}")

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
            sampled_configs_np.shape[0]
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

    def _evaluate(self, sampled_configs):
        """Evaluate sampled configs and add results to dataset.

        `sampled_configs` arrives in scaled space when `_normalize_inputs=True`.
        We inverse-scale per-row to call `inv_transform_configuration` (which
        expects raw HP values), but keep the scaled row for `self.data` so the
        SPN keeps learning in the same scaled space it samples in.
        """
        evaled_samples = []
        for i in range(len(sampled_configs)):
            scaled_sample = list(sampled_configs[i])
            if self._normalize_inputs:
                raw_sample = list(self._scale_from_hp(np.array([scaled_sample], dtype=float))[0])
            else:
                raw_sample = scaled_sample
            config_dict = self.transform.inv_transform_configuration(raw_sample)
            evaluation = self.objective(config_dict)
            if evaluation is not None:
                self.evaluations.append((config_dict, evaluation, self._curr_iter))
                scaled_sample.append(evaluation.val_performance)
                evaled_samples.append(scaled_sample)
        evaled_samples = np.array(evaled_samples)
        if evaled_samples.shape[0] > 0:
            self.data = np.concatenate((self.data, evaled_samples))

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
        if self._log_hpo_runtime:
            evaluations = []
            optim_cost = np.sum(self.hpo_runtimes)
            for cfg, eval_res, it in self.evaluations:
                eval_res.set_optim_cost(optim_cost)
                evaluations.append((cfg, eval_res, it))
            return evaluations
        return self.evaluations

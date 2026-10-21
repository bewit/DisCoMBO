from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
	CategoricalHyperparameter,
	Hyperparameter,
	UniformFloatHyperparameter,
	UniformIntegerHyperparameter,
)
from ConfigSpace import Configuration as _Configuration
from smac import HyperparameterOptimizationFacade, Scenario, BlackBoxFacade
from smac.acquisition.function import LCB
from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.main.config_selector import ConfigSelector
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from ..benchmarks import BenchQueryResult
from ..search_spaces import SearchSpace
from ..utils.smac import SafeLocalAndSrotedRandomSearch
from ..utils.smac.dynamic_prior_acq import DynamicPriorAcquisitionFunction
from .optimizer import Optimizer


@dataclass
class _InterventionEvent:
	iteration: int
	payload: Dict[str, Any]


def _build_prior_configspace(base_configspace: ConfigurationSpace, interaction: Dict[str, Any], prior_std_denominator: float, seed: int = 0) -> ConfigurationSpace:
	hyperparameters: List[Hyperparameter] = []

	for hp in base_configspace.get_hyperparameters():
		if hp.name not in interaction:
			hyperparameters.append(hp)
			continue

		payload = interaction[hp.name]

		if isinstance(payload, Hyperparameter):
			hyperparameters.append(payload)
			continue

		if isinstance(hp, CategoricalHyperparameter):
			choices = list(hp.choices)
			if payload not in choices:
				hyperparameters.append(hp)
				continue

			weights = np.full(len(choices), 1e-3, dtype=float)
			weights[choices.index(payload)] = 1.0
			weights = weights / weights.sum()
			hyperparameters.append(CategoricalHyperparameter(hp.name, choices=choices, weights=list(weights.tolist())))
			continue

		if isinstance(hp, UniformIntegerHyperparameter):
			span = max(1, hp.upper - hp.lower)
			delta = max(1, int(round(span / max(1.0, prior_std_denominator))))
			center = int(round(payload))
			lower = max(int(hp.lower), center - delta)
			upper = min(int(hp.upper), center + delta)
			if lower >= upper:
				lower = max(int(hp.lower), center - 1)
				upper = min(int(hp.upper), center + 1)
			hyperparameters.append(UniformIntegerHyperparameter(hp.name, lower=lower, upper=upper, default_value=center))
			continue

		if isinstance(hp, UniformFloatHyperparameter):
			span = hp.upper - hp.lower
			delta = span / max(1.0, prior_std_denominator)
			center = float(payload)
			lower = max(float(hp.lower), center - delta)
			upper = min(float(hp.upper), center + delta)
			if center < lower or center > upper:
				center = np.clip(center, lower, upper)
			if lower >= upper:
				lower = max(float(hp.lower), center - 1e-6)
				upper = min(float(hp.upper), center + 1e-6)
			hyperparameters.append(UniformFloatHyperparameter(hp.name, lower=lower, upper=upper, default_value=center))
			continue

		hyperparameters.append(hp)

	prior_configspace = ConfigurationSpace(seed=seed)
	prior_configspace.add_hyperparameters(hyperparameters)
	return prior_configspace


class _SequencePriorCallback(Callback):
	def __init__(
		self,
		events: List[_InterventionEvent],
		prior_std_denominator: float,
		prior_decay_beta: float,
		remove_old_prior: bool,
		seed: int = 0,
		intervention_indices: List[int] = None,
		score_log: List[Dict[str, Any]] = None,
		abort_holder: Dict[str, bool] = None,
		n_validation_samples: int = 10,
		initial_design_size: int = 0,
	) -> None:
		super().__init__()
		self._events = sorted(events, key=lambda x: x.iteration)
		self._next_idx = 0
		self._prior_std_denominator = prior_std_denominator
		self._prior_decay_beta = prior_decay_beta
		self._remove_old_prior = remove_old_prior
		self._seed = seed
		self._intervention_indices = intervention_indices if intervention_indices is not None else [-1] * len(self._events)
		self._score_log = score_log
		self._abort_holder = abort_holder
		self._n_validation_samples = n_validation_samples
		self._initial_design_size = initial_design_size

	def _compute_lcb_score(self, smbo: SMBO, prior_configspace: ConfigurationSpace, event: _InterventionEvent) -> float:
		"""Compute lcb_prior.mean() - lcb_incumbent.mean() over n_validation_samples each.

		Mirrors the AbstractPriorCallback.accept_prior DIFFERENCE logic from
		discombo/utils/smac/dynamic_prior_cb.py:185-222, adapted to our intervention payloads.
		"""
		if smbo.runhistory.finished < self._initial_design_size:
			print(f"[warn] intervention triggered before initial design completed "
			      f"(finished={smbo.runhistory.finished}, initial_design_size={self._initial_design_size}); "
			      f"score=NaN")
			return float("nan")

		acquisition_wrapper = smbo.intensifier.config_selector._acquisition_function
		model = getattr(acquisition_wrapper, "_model", None) or getattr(acquisition_wrapper, "model", None)
		if model is None:
			print("[warn] surrogate model not yet fit at intervention time; score=NaN")
			return float("nan")

		incumbent_config = smbo.intensifier.get_incumbent()
		if incumbent_config is None:
			print("[warn] no incumbent at intervention time; score=NaN")
			return float("nan")

		incumbent_dict = dict(incumbent_config)
		# Restrict to keys present in the intervention payload so the incumbent-side configspace
		# uses the same set of locally narrowed hyperparameters as the prior side.
		incumbent_payload = {k: incumbent_dict[k] for k in event.payload.keys() if k in incumbent_dict}
		incumbent_configspace = _build_prior_configspace(
			base_configspace=smbo._configspace,
			interaction=incumbent_payload,
			prior_std_denominator=self._prior_std_denominator * (self._next_idx + 1),
			seed=self._seed,
		)

		try:
			prior_samples = prior_configspace.sample_configuration(size=self._n_validation_samples)
			incumbent_samples = incumbent_configspace.sample_configuration(size=self._n_validation_samples)
		except Exception as exc:
			print(f"[warn] failed sampling for LCB scoring ({exc}); score=NaN")
			return float("nan")

		if self._n_validation_samples == 1:
			prior_samples = [prior_samples]
			incumbent_samples = [incumbent_samples]

		# Re-project onto the original configspace so the surrogate sees configs in the
		# coordinate system it was trained on (sampled configs live in narrowed configspaces).
		origin_cs = smbo._configspace
		def _project(cfgs):
			out = []
			for c in cfgs:
				try:
					out.append(_Configuration(origin_cs, values=dict(c)))
				except Exception:
					out.append(c)
			return out
		prior_samples = _project(prior_samples)
		incumbent_samples = _project(incumbent_samples)

		lcb = LCB()
		lcb.update(model=model, num_data=smbo.runhistory.finished)
		try:
			lcb_prior = np.asarray(lcb(prior_samples)).squeeze()
			lcb_incumbent = np.asarray(lcb(incumbent_samples)).squeeze()
		except Exception as exc:
			print(f"[warn] LCB evaluation failed ({exc}); score=NaN")
			return float("nan")

		return float(np.mean(lcb_prior) - np.mean(lcb_incumbent))

	def on_ask_start(self, smbo: SMBO):
		acquisition = smbo.intensifier.config_selector._acquisition_function
		if hasattr(acquisition, "current_config_nuber"):
			acquisition.current_config_nuber = smbo.runhistory.finished

		while self._next_idx < len(self._events):
			event = self._events[self._next_idx]
			if event.iteration > smbo.runhistory.finished:
				break

			prior_configspace = _build_prior_configspace(
				base_configspace=smbo._configspace,
				interaction=event.payload,
				prior_std_denominator=self._prior_std_denominator * (self._next_idx + 1),
				seed=self._seed,
			)

			# Score before applying the prior so the LCB reflects pre-intervention surrogate state.
			if self._score_log is not None:
				score = self._compute_lcb_score(smbo, prior_configspace, event)
				self._score_log.append({
					"iter": int(event.iteration),
					"intervention_idx": int(self._intervention_indices[self._next_idx]) if self._next_idx < len(self._intervention_indices) else -1,
					"score": score,
				})

			maximizer = smbo.intensifier.config_selector._acquisition_maximizer
			if hasattr(maximizer, "dynamic_init"):
				maximizer.dynamic_init(prior_configspace, remove_old_prior=self._remove_old_prior)

			acquisition.dynamic_init(
				acquisition_function=acquisition._acquisition_function,
				prior_configspace=prior_configspace,
				decay_beta=self._prior_decay_beta,
				prior_start=smbo.runhistory.finished,
				remove_old_prior=self._remove_old_prior,
			)
			self._next_idx += 1

			if self._abort_holder is not None and self._abort_holder.get("enabled"):
				self._abort_holder["fired"] = True

		return super().on_ask_start(smbo)


class DynaBOOptimizer(Optimizer):
	def __init__(
		self,
		search_space: SearchSpace,
		objective,
		initial_design_size: int = 20,
		prior_decay: str = "linear",
		prior_std_denominator: float = 5,
		prior_decay_enumerator: int = 20,
		prior_decay_denominator: int = 1,
		remove_old_priors: bool = False,
		cfg_selector_retries: int = 16,
		abort_after_intervention: bool = False,
		n_validation_samples: int = 10,
		**scenario_params,
	) -> None:
		super().__init__(search_space, objective)
		self.config_space = search_space.to_configspace()
		self.scenario = Scenario(self.config_space, **scenario_params)
		self.objective = objective
		self._scenario_args = scenario_params
		self._num_iter = scenario_params["n_trials"]
		self.search_space = search_space
		self.hist = []

		self._initial_design_size = initial_design_size
		self._prior_decay = prior_decay
		self._prior_std_denominator = prior_std_denominator
		# Match PiBO: prior decay strength scales with budget, beta = N/10.
		self._prior_decay_beta = self._num_iter / 10.0
		self._remove_old_priors = remove_old_priors
		self._cfg_selector_retries = cfg_selector_retries
		self._n_validation_samples = n_validation_samples

		self._events: List[_InterventionEvent] = []
		self._intervention_indices: List[int] = []
		self._intervention_score_log: List[Dict[str, Any]] = []
		self._abort_holder: Dict[str, bool] = {"enabled": bool(abort_after_intervention), "fired": False}

		self._init_smac_objects()

	def _init_smac_objects(self):
		maximizer = SafeLocalAndSrotedRandomSearch(
			self.search_space,
			challengers=10000,
			local_search_iterations=20,
			n_steps_plateau_walk=20,
			seed=self._scenario_args["seed"],
		)
		intensifier = HyperparameterOptimizationFacade.get_intensifier(
			self.scenario,
			max_config_calls=1,
		)
		initial_design = HyperparameterOptimizationFacade.get_initial_design(
			scenario=self.scenario,
			n_configs=self._initial_design_size,
		)

		self.acquisition = DynamicPriorAcquisitionFunction(
			acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(self.scenario),
			initial_design_size=initial_design._n_configs,
			dynabo=True,
			prior_decay=self._prior_decay,
		)

		self._prior_callback = _SequencePriorCallback(
			events=self._events,
			prior_std_denominator=self._prior_std_denominator,
			prior_decay_beta=self._prior_decay_beta,
			remove_old_prior=self._remove_old_priors,
			seed=self._scenario_args["seed"],
			intervention_indices=self._intervention_indices,
			score_log=self._intervention_score_log,
			abort_holder=self._abort_holder,
			n_validation_samples=self._n_validation_samples,
			initial_design_size=self._initial_design_size,
		)

		self.runhistory_encoder = BlackBoxFacade.get_runhistory_encoder(scenario=self.scenario)

		self.smac = HyperparameterOptimizationFacade(
			self.scenario,
			self._objective_wrapper,
			intensifier=intensifier,
			acquisition_function=self.acquisition,
			acquisition_maximizer=maximizer,
			callbacks=[self._prior_callback],
			initial_design=initial_design,
			overwrite=True,
			config_selector=ConfigSelector(
				self.scenario,
				retries=self._cfg_selector_retries,
			),
			runhistory_encoder=self.runhistory_encoder,
		)

	def _objective_wrapper(self, config: Configuration, seed: int = 0) -> float:
		res: BenchQueryResult = self.objective(config)
		return -res.val_performance

	def optimize(self):
		evaluations, configs = [], []
		for i in range(self._num_iter):
			if i % 50 == 0:
				print(f"Iteration {i}/{self._num_iter}")

			try:
				info = self.smac.ask()
				if self.search_space.is_valid(info.config):
					res: BenchQueryResult = self.objective(info.config.get_dictionary())
					value = TrialValue(-res.val_performance, time=0.5)

					evaluations.append(res)
					configs.append(info.config.get_dictionary())
					self.hist.append((info.config.get_dictionary(), res, i))

					self.smac.tell(info, value)
			except StopIteration:
				cfg = self.search_space.to_configspace().sample_configuration()
				res = self.objective(cfg.get_dictionary())
				evaluations.append(res)
				configs.append(cfg.get_dictionary())
				self.hist.append((cfg.get_dictionary(), res, i))
				value = TrialValue(-res.val_performance, time=0.5)
				info = TrialInfo(cfg)
				self.smac.tell(info, value)

			if self._abort_holder.get("fired"):
				print(f"[Optimizer][DynaBO] abort_after_intervention set; "
				      f"stopping at iter {i} after capturing intervention score.")
				break

		if not evaluations:
			cfg = self.search_space.to_configspace().sample_configuration()
			res = self.objective(cfg.get_dictionary())
			evaluations.append(res)
			configs.append(cfg.get_dictionary())

		val_accs = [res.val_performance for res in evaluations]
		best = max(val_accs)
		best_idx = val_accs.index(best)
		return configs[best_idx], best

	def get_intervention_score_log(self) -> List[Dict[str, Any]]:
		"""Return per-intervention score records captured during optimize()."""
		return list(self._intervention_score_log)

	@property
	def history(self):
		return self.hist

	def intervene(self, interventions, iters=None, **kwargs):
		# Reset run-local state: previous score log, abort flag, event list.
		self._intervention_score_log.clear()
		self._abort_holder["fired"] = False

		if interventions is None:
			self._events = []
			self._intervention_indices = []
			self._init_smac_objects()
			return

		if isinstance(interventions, dict):
			interventions = [interventions]

		if iters is None:
			iter_value = kwargs.get("iter", 0)
			iters = [iter_value] * len(interventions)

		if len(interventions) != len(iters):
			raise ValueError("interventions and iters must have the same length")

		events = []
		original_indices = []
		for original_idx, (interaction, iteration) in enumerate(zip(interventions, iters)):
			if interaction is None:
				continue
			events.append(_InterventionEvent(iteration=int(iteration), payload=interaction))
			original_indices.append(original_idx)

		# Sort events by iteration; carry the original indices alongside so the score log
		# can record the caller's intervention index regardless of iteration order.
		paired = sorted(zip(events, original_indices), key=lambda p: p[0].iteration)
		self._events = [p[0] for p in paired]
		self._intervention_indices = [p[1] for p in paired]
		self._init_smac_objects()

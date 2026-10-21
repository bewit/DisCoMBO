import json
import os
import pickle
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .benchmark import BaseBenchmark, BenchQueryResult
from ..search_spaces import PVSearchSpace


class FlorisBenchmark(BaseBenchmark):

    def __init__(self, save_dir='./benchmark_data/floris/') -> None:
        super().__init__()
        self._file_path = save_dir
        self._search_space = None
        self.surrogate = None
        self.metadata = None
        self._feature_columns = None
        self._target_column = None
        self._min_prediction = None
        self._max_prediction = None

        self._load_artifacts_and_search_space()

    def _load_artifacts_and_search_space(self):
        model_path = os.path.join(self._file_path, 'model.pkl')
        metadata_path = os.path.join(self._file_path, 'metadata.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Floris surrogate model not found: {model_path}')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f'Floris surrogate metadata not found: {metadata_path}')

        with open(model_path, 'rb') as f:
            self.surrogate = pickle.load(f)

        if not isinstance(self.surrogate, GradientBoostingRegressor):
            raise TypeError(
                f'Expected a GradientBoostingRegressor surrogate in {model_path}, '
                f'got {type(self.surrogate).__name__}'
            )

        # Compatibility shim for sklearn version mismatches.
        # Example: models trained in 1.1.x may store `_loss`, while 1.0.x
        # runtime predict path expects `loss_`.
        if not hasattr(self.surrogate, 'loss_') and hasattr(self.surrogate, '_loss'):
            self.surrogate.loss_ = self.surrogate._loss
            warnings.warn(
                'Loaded FLORIS surrogate with sklearn attribute mismatch '
                '(missing `loss_`, found `_loss`). Applied compatibility shim. '
                'Consider retraining the surrogate with the current sklearn version.',
                RuntimeWarning,
            )

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        if 'feature_columns' not in self.metadata or len(self.metadata['feature_columns']) == 0:
            raise ValueError(f'Invalid metadata in {metadata_path}: missing feature_columns')
        if 'target_column' not in self.metadata:
            raise ValueError(f'Invalid metadata in {metadata_path}: missing target_column')

        self._feature_columns = self.metadata['feature_columns']
        self._target_column = self.metadata['target_column']

        source_file = self.metadata.get('source_file')
        if source_file is None:
            raise ValueError(f'Invalid metadata in {metadata_path}: missing source_file')

        csv_path = os.path.join(self._file_path, source_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Floris source csv not found: {csv_path}')

        df = pd.read_csv(csv_path)
        missing = [c for c in self._feature_columns if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f'Columns {missing} from metadata are missing in {csv_path}')

        feature_bounds = {
            feature_name: (df[feature_name].min(), df[feature_name].max())
            for feature_name in self._feature_columns
        }
        self._search_space = PVSearchSpace.from_feature_bounds(feature_bounds)

        target_series = self._resolve_target_series(df, csv_path)
        if target_series.empty:
            raise ValueError(f'Target column {self._target_column} in {csv_path} has no numeric values')

        self._min_prediction = float(target_series.min())
        self._max_prediction = float(target_series.max())

    def _resolve_target_series(self, df: pd.DataFrame, csv_path: str):
        if self._target_column in df.columns:
            return pd.to_numeric(df[self._target_column], errors='coerce').dropna()

        norm_cfg = self.metadata.get('target_normalization', {})
        original_target = norm_cfg.get('original_target_column')
        normalized_target = norm_cfg.get('normalized_target_column')
        target_mean = norm_cfg.get('mean')
        target_std = norm_cfg.get('std')

        if normalized_target == self._target_column and original_target in df.columns:
            raw_target = pd.to_numeric(df[original_target], errors='coerce').dropna()
            if raw_target.empty:
                return raw_target

            if target_std == 0.0:
                return pd.Series(np.zeros(raw_target.shape[0], dtype=float))

            if target_mean is None or target_std is None:
                raise ValueError(
                    f'Invalid metadata in {csv_path}: missing target normalization mean/std.'
                )

            return (raw_target - float(target_mean)) / float(target_std)

        raise ValueError(
            f'Target column {self._target_column} not found in {csv_path} and no valid '\
            'target_normalization metadata was provided.'
        )

    def _prepare_query_vector(self, cfg: Dict):
        expected_keys = set(self._feature_columns)
        cfg_keys = set(cfg.keys())
        if cfg_keys != expected_keys:
            missing = sorted(expected_keys - cfg_keys)
            unexpected = sorted(cfg_keys - expected_keys)
            raise ValueError(
                f'Configuration keys do not match Floris feature columns. Missing: {missing}, Unexpected: {unexpected}'
            )

        ordered_values = []
        for feature_name in self._feature_columns:
            value = cfg[feature_name]
            if not isinstance(value, (float, int, np.floating, np.integer)):
                raise ValueError(f'Feature {feature_name} must be numeric, got {type(value)}')
            ordered_values.append(float(value))

        x = np.array(ordered_values, dtype=float).reshape(1, -1)
        if x.shape[1] != len(self._feature_columns):
            raise ValueError(
                f'Query vector feature mismatch: expected {len(self._feature_columns)}, got {x.shape[1]}'
            )
        return x

    def query(self, cfg: Dict, budget=None) -> BenchQueryResult:
        x = self._prepare_query_vector(cfg)
        y = float(self.surrogate.predict(x).flatten()[0])
        return BenchQueryResult(
            train_performance=y,
            val_performance=y,
            test_performance=y,
        )

    def get_min_and_max(self):
        return self._min_prediction, self._max_prediction

    @property
    def search_space(self):
        return self._search_space

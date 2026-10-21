"""
Train a Gradient Boosting surrogate model for the FLORIS benchmark data.

The script expects the FLORIS directory to contain exactly one CSV file.
It trains a regressor to predict `total_power_w` from yaw-angle features and
stores the model and metadata under the given surrogate output path.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "total_power_w"
EXCLUDED_COLUMNS = {"eval_id", TARGET_COLUMN}


def _persist_normalized_target_column_if_missing(
	csv_path: Path,
	target_col: str,
	normalized_target_col: str,
	target_mean: float,
	target_std: float,
):
	original_df = pd.read_csv(csv_path)
	if normalized_target_col in original_df.columns:
		return

	target_series = pd.to_numeric(original_df[target_col], errors="coerce")
	normalized_values = pd.Series(np.nan, index=original_df.index, dtype=float)
	valid_mask = target_series.notna()

	if target_std == 0.0:
		normalized_values.loc[valid_mask] = 0.0
	else:
		normalized_values.loc[valid_mask] = (
			(target_series.loc[valid_mask] - target_mean) / target_std
		)

	original_df[normalized_target_col] = normalized_values
	original_df.to_csv(csv_path, index=False)
	print(f"Added '{normalized_target_col}' to {csv_path}.")


def _normalize_target(df: pd.DataFrame, target_col: str):
	normalized_target_col = f"normalized_{target_col}"
	target_mean = float(df[target_col].mean())
	target_std = float(df[target_col].std(ddof=0))

	if target_std == 0.0:
		print(
			f"Target column '{target_col}' has zero variance. "
			f"Using zeros for '{normalized_target_col}'."
		)
		df[normalized_target_col] = 0.0
	else:
		df[normalized_target_col] = (df[target_col] - target_mean) / target_std

	return df, normalized_target_col, target_mean, target_std


def _find_single_csv(data_path: str) -> Path:
	csv_files = sorted(Path(data_path).glob("*.csv"))
	if len(csv_files) != 1:
		raise ValueError(
			f"Expected exactly one CSV in '{data_path}', found {len(csv_files)}."
		)
	return csv_files[0]


def _load_training_data(csv_path: Path):
	df = pd.read_csv(csv_path)
	if TARGET_COLUMN not in df.columns:
		raise ValueError(f"Missing target column '{TARGET_COLUMN}' in {csv_path}.")

	numeric_df = df.apply(pd.to_numeric, errors="coerce")
	if TARGET_COLUMN not in numeric_df.columns:
		raise ValueError(f"Missing numeric target column '{TARGET_COLUMN}' in {csv_path}.")

	clean_df = numeric_df.dropna(subset=[TARGET_COLUMN]).copy()
	clean_df, normalized_target_col, target_mean, target_std = _normalize_target(clean_df, TARGET_COLUMN)

	feature_cols = [
		c
		for c in clean_df.columns
		if c not in EXCLUDED_COLUMNS and c != normalized_target_col
	]
	if len(feature_cols) == 0:
		raise ValueError("No usable feature columns found in FLORIS CSV.")

	clean_df = clean_df[feature_cols + [normalized_target_col]].dropna()
	if clean_df.shape[0] < 10:
		raise ValueError(
			f"Not enough rows after cleaning: {clean_df.shape[0]}. Need at least 10."
		)

	_persist_normalized_target_column_if_missing(
		csv_path=csv_path,
		target_col=TARGET_COLUMN,
		normalized_target_col=normalized_target_col,
		target_mean=target_mean,
		target_std=target_std,
	)

	X = clean_df[feature_cols].to_numpy(dtype=float)
	y = clean_df[normalized_target_col].to_numpy(dtype=float)
	return (
		X,
		y,
		feature_cols,
		int(clean_df.shape[0]),
		normalized_target_col,
		target_mean,
		target_std,
	)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
	return {
		"r2": float(r2_score(y_true, y_pred)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
	}


def train_floris_surrogate(
	data_path: str,
	surrogate_path: str,
	test_size: float,
	seed: int,
	n_estimators: int,
	learning_rate: float,
	max_depth: int,
	subsample: float,
):
	csv_path = _find_single_csv(data_path)
	(
		X,
		y,
		feature_cols,
		n_rows,
		normalized_target_col,
		target_mean,
		target_std,
	) = _load_training_data(csv_path)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=seed,
	)

	model = GradientBoostingRegressor(
		n_estimators=n_estimators,
		learning_rate=learning_rate,
		max_depth=max_depth,
		subsample=subsample,
		random_state=seed,
	)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	eval_metrics = _metrics(y_test, y_pred)
	print(
		f"FLORIS surrogate metrics: "
		f"R2={eval_metrics['r2']:.4f}, "
		f"MAE={eval_metrics['mae']:.4f}, "
		f"RMSE={eval_metrics['rmse']:.4f}"
	)

	output_dir = Path(surrogate_path) / "floris"
	output_dir.mkdir(parents=True, exist_ok=True)

	model_path = output_dir / "model.pkl"
	with open(model_path, "wb") as fout:
		pickle.dump(model, fout)

	metadata = {
		"source_file": csv_path.name,
		"n_rows": n_rows,
		"n_features": len(feature_cols),
		"feature_columns": feature_cols,
		"target_column": normalized_target_col,
		"target_normalization": {
			"method": "standard",
			"original_target_column": TARGET_COLUMN,
			"normalized_target_column": normalized_target_col,
			"mean": target_mean,
			"std": target_std,
		},
		"split": {
			"test_size": test_size,
			"seed": seed,
			"n_train": int(X_train.shape[0]),
			"n_test": int(X_test.shape[0]),
		},
		"metrics": eval_metrics,
		"model": {
			"library": "scikit-learn",
			"estimator": "GradientBoostingRegressor",
			"params": {
				"n_estimators": n_estimators,
				"learning_rate": learning_rate,
				"max_depth": max_depth,
				"subsample": subsample,
				"random_state": seed,
			},
		},
	}

	with open(output_dir / "metadata.json", "w", encoding="utf-8") as fout:
		json.dump(metadata, fout, indent=2, sort_keys=True)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-path", required=True, help="Path to benchmark_data/floris")
	parser.add_argument("--surrogate-path", required=True, help="Directory where surrogate artifacts are stored")
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--n-estimators", type=int, default=300)
	parser.add_argument("--learning-rate", type=float, default=0.05)
	parser.add_argument("--max-depth", type=int, default=4)
	parser.add_argument("--subsample", type=float, default=0.9)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	os.makedirs(args.surrogate_path, exist_ok=True)

	train_floris_surrogate(
		data_path=args.data_path,
		surrogate_path=args.surrogate_path,
		test_size=args.test_size,
		seed=args.seed,
		n_estimators=args.n_estimators,
		learning_rate=args.learning_rate,
		max_depth=args.max_depth,
		subsample=args.subsample,
	)

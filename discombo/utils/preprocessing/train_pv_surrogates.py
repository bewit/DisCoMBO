"""
Train scikit-learn Gradient Boosting surrogate models for the PV benchmark datasets.

The script trains one surrogate per dataset and stores:
1) A final model fitted on the full dataset.
2) Metadata with CV metrics, feature names, and configuration.
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
from sklearn.model_selection import KFold, RepeatedKFold


PV_DATASETS = [
	("AgNP", "AgNP.csv", "loss"),
	("AutoAM", "AutoAM.csv", "Score"),
	("crossed_barrel", "crossed_barrel.csv", "toughness"),
	("p3ht", "p3ht.csv", "Conductivity (measured) (S/cm)"),
	("perovskite", "perovskite.csv", "Instability index"),
]

DEFAULT_GB_PARAMS = {
	"n_estimators": 300,
	"max_depth": 6,
	"learning_rate": 0.05,
	"subsample": 0.9,
	"random_state": 42,
}


def _persist_normalized_target_column_if_missing(
	csv_path: str,
	dataset_name: str,
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
	print(f"[{dataset_name}] Added '{normalized_target_col}' to {csv_path}.")


def _ensure_numeric_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
	converted = df.apply(pd.to_numeric, errors="coerce")
	dropped = int(converted.isna().any(axis=1).sum())
	if dropped > 0:
		print(f"[{dataset_name}] Dropping {dropped} rows due to non-numeric/missing values.")
		converted = converted.dropna(axis=0).reset_index(drop=True)
	if converted.shape[0] < 3:
		raise ValueError(
			f"[{dataset_name}] Not enough clean rows after preprocessing: {converted.shape[0]}"
		)
	return converted


def _normalize_target(df: pd.DataFrame, dataset_name: str, target_col: str):
	normalized_target_col = f"normalized_{target_col}"
	target_mean = float(df[target_col].mean())
	target_std = float(df[target_col].std(ddof=0))

	if target_std == 0.0:
		print(
			f"[{dataset_name}] Target column '{target_col}' has zero variance. "
			f"Using zeros for '{normalized_target_col}'."
		)
		df[normalized_target_col] = 0.0
	else:
		df[normalized_target_col] = (df[target_col] - target_mean) / target_std

	return df, normalized_target_col, target_mean, target_std


def load_pv_dataset(data_dir: str, dataset_name: str, filename: str, target_col: str):
	csv_path = os.path.join(data_dir, filename)
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"[{dataset_name}] File not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if target_col not in df.columns:
		raise ValueError(f"[{dataset_name}] Missing target column: {target_col}")

	df = _ensure_numeric_dataframe(df, dataset_name)
	df, normalized_target_col, target_mean, target_std = _normalize_target(df, dataset_name, target_col)
	_persist_normalized_target_column_if_missing(
		csv_path=csv_path,
		dataset_name=dataset_name,
		target_col=target_col,
		normalized_target_col=normalized_target_col,
		target_mean=target_mean,
		target_std=target_std,
	)
	feature_cols = [c for c in df.columns if c not in {target_col, normalized_target_col}]
	if len(feature_cols) == 0:
		raise ValueError(f"[{dataset_name}] No feature columns found.")

	X = df[feature_cols].to_numpy(dtype=float)
	y = df[normalized_target_col].to_numpy(dtype=float)
	return (
		X,
		y,
		feature_cols,
		int(df.shape[0]),
		normalized_target_col,
		target_mean,
		target_std,
	)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	return {
		"r2": float(r2_score(y_true, y_pred)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"rmse": rmse,
	}


def evaluate_with_cv(X: np.ndarray, y: np.ndarray, folds: int, repeats: int, seed: int, gb_params: dict):
	if folds < 2:
		raise ValueError("folds must be >= 2")
	if repeats < 1:
		raise ValueError("repeats must be >= 1")
	if X.shape[0] < folds:
		raise ValueError(
			f"folds={folds} is larger than dataset size={X.shape[0]}."
		)

	splitter = (
		RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
		if repeats > 1
		else KFold(n_splits=folds, shuffle=True, random_state=seed)
	)

	fold_metrics = []
	for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
		model = GradientBoostingRegressor(**gb_params)
		model.fit(X[train_idx], y[train_idx])
		preds = model.predict(X[test_idx])
		metrics = compute_regression_metrics(y[test_idx], preds)
		metrics["fold"] = fold_idx
		fold_metrics.append(metrics)

	summary = {}
	for metric_name in ["r2", "mae", "rmse"]:
		values = np.array([m[metric_name] for m in fold_metrics], dtype=float)
		summary[f"{metric_name}_mean"] = float(values.mean())
		summary[f"{metric_name}_std"] = float(values.std())

	return fold_metrics, summary


def train_final_model(X: np.ndarray, y: np.ndarray, gb_params: dict):
	model = GradientBoostingRegressor(**gb_params)
	model.fit(X, y)
	return model


def save_artifacts(
	surrogate_path: str,
	dataset_name: str,
	model,
	metadata: dict,
):
	dataset_dir = Path(surrogate_path) / dataset_name
	dataset_dir.mkdir(parents=True, exist_ok=True)

	model_path = dataset_dir / "model.pkl"
	metadata_path = dataset_dir / "metadata.json"

	with open(model_path, "wb") as fout:
		pickle.dump(model, fout)

	with open(metadata_path, "w", encoding="utf-8") as fout:
		json.dump(metadata, fout, indent=2, sort_keys=True)


def train_surrogates_for_pv(
	data_path: str,
	surrogate_path: str,
	folds: int,
	repeats: int,
	seed: int,
	gb_params: dict,
):
	os.makedirs(surrogate_path, exist_ok=True)

	for dataset_name, filename, target_col in PV_DATASETS:
		print(f"======== DATASET: {dataset_name} ========")
		(
			X,
			y,
			feature_cols,
			n_rows,
			normalized_target_col,
			target_mean,
			target_std,
		) = load_pv_dataset(data_path, dataset_name, filename, target_col)

		fold_metrics, cv_summary = evaluate_with_cv(
			X=X,
			y=y,
			folds=folds,
			repeats=repeats,
			seed=seed,
			gb_params=gb_params,
		)

		print(
			f"[{dataset_name}] CV r2={cv_summary['r2_mean']:.4f}±{cv_summary['r2_std']:.4f}, "
			f"mae={cv_summary['mae_mean']:.4f}±{cv_summary['mae_std']:.4f}, "
			f"rmse={cv_summary['rmse_mean']:.4f}±{cv_summary['rmse_std']:.4f}"
		)

		final_model = train_final_model(X, y, gb_params=gb_params)

		metadata = {
			"dataset": dataset_name,
			"source_file": filename,
			"target_column": normalized_target_col,
			"target_normalization": {
				"method": "standard",
				"original_target_column": target_col,
				"normalized_target_column": normalized_target_col,
				"mean": target_mean,
				"std": target_std,
			},
			"feature_columns": feature_cols,
			"n_rows": n_rows,
			"n_features": len(feature_cols),
			"cv": {
				"folds": folds,
				"repeats": repeats,
				"shuffle": True,
				"seed": seed,
				"n_total_splits": len(fold_metrics),
				"fold_metrics": fold_metrics,
				"summary": cv_summary,
			},
			"model": {
				"library": "scikit-learn",
				"estimator": "GradientBoostingRegressor",
				"params": gb_params,
			},
		}

		save_artifacts(
			surrogate_path=surrogate_path,
			dataset_name=dataset_name,
			model=final_model,
			metadata=metadata,
		)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-path", required=True, help="Path to benchmark_data/pv")
	parser.add_argument("--surrogate-path", required=True, help="Path where surrogate artifacts are stored")
	parser.add_argument("--folds", type=int, default=5)
	parser.add_argument("--repeats", type=int, default=1)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--n-estimators", type=int, default=DEFAULT_GB_PARAMS["n_estimators"])
	parser.add_argument("--max-depth", type=int, default=DEFAULT_GB_PARAMS["max_depth"])
	parser.add_argument("--learning-rate", type=float, default=DEFAULT_GB_PARAMS["learning_rate"])
	parser.add_argument("--subsample", type=float, default=DEFAULT_GB_PARAMS["subsample"])
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	gb_params = {
		"n_estimators": args.n_estimators,
		"max_depth": args.max_depth,
		"learning_rate": args.learning_rate,
		"subsample": args.subsample,
		"random_state": args.seed,
	}

	train_surrogates_for_pv(
		data_path=args.data_path,
		surrogate_path=args.surrogate_path,
		folds=args.folds,
		repeats=args.repeats,
		seed=args.seed,
		gb_params=gb_params,
	)

"""Internal validation utilities for small-sample binary survey models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold

from smallsurv.calibration import evaluate_calibration


@dataclass
class ValidationReport:
    """Structured internal validation result for binary prediction models."""

    method: str
    n_obs: int
    validated_metrics: dict[str, float]
    metric_std: dict[str, float] | None
    apparent_metrics: dict[str, float] | None
    optimism: dict[str, float] | None
    n_iterations: int
    details: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return validation metrics as a plain dictionary."""
        return {
            "method": self.method,
            "n_obs": self.n_obs,
            "n_iterations": self.n_iterations,
            "validated_metrics": {k: round(v, 4) for k, v in self.validated_metrics.items()},
            "metric_std": (
                {k: round(v, 4) for k, v in self.metric_std.items()}
                if self.metric_std is not None
                else None
            ),
            "apparent_metrics": (
                {k: round(v, 4) for k, v in self.apparent_metrics.items()}
                if self.apparent_metrics is not None
                else None
            ),
            "optimism": (
                {k: round(v, 4) for k, v in self.optimism.items()}
                if self.optimism is not None
                else None
            ),
        }

    def summary(self) -> str:
        """Return a human-readable validation summary report."""
        lines = [
            "=" * 64,
            f"smallsurv Internal Validation Report ({self.method})",
            "=" * 64,
            f"Sample Size (N) : {self.n_obs}",
            f"Iterations      : {self.n_iterations}",
            "-" * 64,
        ]
        if self.method == "repeated_cv" and self.metric_std is not None:
            lines.append("Cross-Validated Performance (Mean +/- SD):")
            for metric, val in self.validated_metrics.items():
                sd = self.metric_std.get(metric, float("nan"))
                lines.append(f"  {metric:<18}: {val:.4f} +/- {sd:.4f}")
        else:
            lines.append("Optimism-Corrected Performance:")
            for metric, val in self.validated_metrics.items():
                app = self.apparent_metrics[metric] if self.apparent_metrics else float("nan")
                opt = self.optimism[metric] if self.optimism else float("nan")
                lines.append(
                    f"  {metric:<18}: {val:.4f} (Apparent={app:.4f}, Optimism={opt:+.4f})"
                )
        lines.append("=" * 64)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _subset_X(X: pd.DataFrame | np.ndarray, idx: np.ndarray) -> pd.DataFrame | np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return X[idx]


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    cal = evaluate_calibration(y_true, y_prob, n_bins=5)
    return {
        "roc_auc": cal.roc_auc,
        "brier_score": cal.brier_score,
        "calibration_slope": cal.calibration_slope,
        "citl": cal.citl,
    }


def repeated_cv_validate(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> ValidationReport:
    """Validate a binary classifier using Repeated Stratified K-Fold CV.

    Parameters
    ----------
    estimator : BaseEstimator
        Any scikit-learn compatible classifier implementing ``fit`` and ``predict_proba``.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Binary outcome vector (0/1).
    n_splits : int, default=5
        Number of stratified folds per repeat.
    n_repeats : int, default=10
        Number of CV repetitions to reduce small-sample split variance.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ValidationReport
        Mean and standard deviation of discrimination and calibration metrics.
    """
    y_arr = np.asarray(y, dtype=int).ravel()
    n_obs = len(y_arr)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    records: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y_arr), start=1):
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = clone(estimator)
        model.fit(_subset_X(X, train_idx), y_train)
        probs = model.predict_proba(_subset_X(X, test_idx))[:, 1]

        metrics = _compute_metrics(y_test, probs)
        metrics["fold"] = float(fold_idx)
        records.append(metrics)

    if not records:
        raise ValueError("No valid CV folds contained both classes.")

    df_details = pd.DataFrame(records)
    metric_cols = ["roc_auc", "brier_score", "calibration_slope", "citl"]
    mean_metrics = {col: float(np.nanmean(df_details[col])) for col in metric_cols}
    std_metrics = {col: float(np.nanstd(df_details[col], ddof=1)) for col in metric_cols}

    return ValidationReport(
        method="repeated_cv",
        n_obs=n_obs,
        validated_metrics=mean_metrics,
        metric_std=std_metrics,
        apparent_metrics=None,
        optimism=None,
        n_iterations=len(records),
        details=df_details,
    )


def bootstrap_optimism_validate(
    estimator: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_bootstraps: int = 100,
    random_state: int = 42,
) -> ValidationReport:
    """Perform Harrell's optimism-corrected bootstrap internal validation.

    Fits the model on the full dataset to obtain apparent performance, then
    estimates overfitting optimism across ``n_bootstraps`` resamples:
    ``corrected = apparent - mean(boot_performance - orig_performance)``.

    Parameters
    ----------
    estimator : BaseEstimator
        Any scikit-learn compatible classifier implementing ``fit`` and ``predict_proba``.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Binary outcome vector (0/1).
    n_bootstraps : int, default=100
        Number of bootstrap resamples (use 200-500 for final publication results).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ValidationReport
        Apparent, optimism, and optimism-corrected performance metrics.
    """
    y_arr = np.asarray(y, dtype=int).ravel()
    n_obs = len(y_arr)
    if n_bootstraps < 5:
        raise ValueError("n_bootstraps must be at least 5.")

    full_model = clone(estimator).fit(X, y_arr)
    apparent_probs = full_model.predict_proba(X)[:, 1]
    apparent = _compute_metrics(y_arr, apparent_probs)

    rng = np.random.default_rng(random_state)
    metric_cols = ["roc_auc", "brier_score", "calibration_slope", "citl"]
    records: list[dict[str, float]] = []

    for b in range(1, n_bootstraps + 1):
        boot_idx = rng.integers(0, n_obs, size=n_obs)
        y_boot = y_arr[boot_idx]
        if len(np.unique(y_boot)) < 2:
            continue

        X_boot = _subset_X(X, boot_idx)
        boot_model = clone(estimator).fit(X_boot, y_boot)

        boot_probs = boot_model.predict_proba(X_boot)[:, 1]
        orig_probs = boot_model.predict_proba(X)[:, 1]

        boot_m = _compute_metrics(y_boot, boot_probs)
        orig_m = _compute_metrics(y_arr, orig_probs)

        row = {"bootstrap": float(b)}
        for col in metric_cols:
            row[f"optimism_{col}"] = boot_m[col] - orig_m[col]
        records.append(row)

    if not records:
        raise ValueError("No valid bootstrap resamples contained both classes.")

    df_details = pd.DataFrame(records)
    mean_optimism = {
        col: float(np.nanmean(df_details[f"optimism_{col}"])) for col in metric_cols
    }
    corrected = {col: apparent[col] - mean_optimism[col] for col in metric_cols}

    return ValidationReport(
        method="bootstrap_optimism",
        n_obs=n_obs,
        validated_metrics=corrected,
        metric_std=None,
        apparent_metrics=apparent,
        optimism=mean_optimism,
        n_iterations=len(records),
        details=df_details,
    )
"""Calibration metrics and reliability diagnostics for binary prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass
class CalibrationReport:
    """Structured calibration and discrimination assessment for binary predictions."""

    n_obs: int
    event_rate: float
    mean_predicted_prob: float
    citl: float
    calibration_slope: float
    brier_score: float
    brier_skill_score: float
    ece: float
    roc_auc: float
    reliability_table: pd.DataFrame
    status: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return scalar calibration metrics as a plain dictionary."""
        return {
            "n_obs": self.n_obs,
            "event_rate": round(self.event_rate, 4),
            "mean_predicted_prob": round(self.mean_predicted_prob, 4),
            "citl": round(self.citl, 4),
            "calibration_slope": round(self.calibration_slope, 4),
            "brier_score": round(self.brier_score, 4),
            "brier_skill_score": round(self.brier_skill_score, 4),
            "ece": round(self.ece, 4),
            "roc_auc": round(self.roc_auc, 4),
            "status": self.status,
            "notes": self.notes,
        }

    def summary(self) -> str:
        """Return a human-readable calibration summary report."""
        lines = [
            "=" * 64,
            "smallsurv Calibration & Discrimination Report",
            "=" * 64,
            f"Calibration Status   : {self.status}",
            f"Sample Size (N)      : {self.n_obs}",
            f"Observed Event Rate  : {self.event_rate:.3f}",
            f"Mean Predicted Prob  : {self.mean_predicted_prob:.3f}",
            f"CITL (Ideal = 0.00)  : {self.citl:+.3f}",
            f"Calib Slope (Id=1.00): {self.calibration_slope:.3f}",
            f"Brier Score          : {self.brier_score:.4f} (Scaled / IPA = {self.brier_skill_score:.3f})",
            f"ECE                  : {self.ece:.4f}",
            f"ROC-AUC              : {self.roc_auc:.3f}",
            "-" * 64,
        ]
        if self.notes:
            lines.append("Diagnostic Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _safe_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0 - eps)
    return special.logit(clipped)


def _compute_citl(y: np.ndarray, logit_p: np.ndarray) -> float:
    """Estimate calibration-in-the-large (intercept with slope fixed at 1)."""
    target_sum = float(np.sum(y))

    def score(alpha: float) -> float:
        return float(np.sum(special.expit(alpha + logit_p)) - target_sum)

    low, high = -5.0, 5.0
    for _ in range(25):
        if score(low) <= 0.0 <= score(high):
            break
        low -= 5.0
        high += 5.0

    return float(optimize.brentq(score, low, high, xtol=1e-10, maxiter=200))


def _compute_calibration_slope(y: np.ndarray, logit_p: np.ndarray) -> float:
    """Estimate logistic calibration slope: logit(P(y=1)) = a + b * logit(p_hat)."""
    if np.std(logit_p) < 1e-10:
        return float("nan")

    def neg_loglik(params: np.ndarray) -> float:
        a, b = params
        eta = a + b * logit_p
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def grad(params: np.ndarray) -> np.ndarray:
        a, b = params
        p = special.expit(a + b * logit_p)
        diff = p - y
        return np.array([np.sum(diff), np.sum(diff * logit_p)], dtype=float)

    res = optimize.minimize(
        neg_loglik,
        x0=np.array([0.0, 1.0], dtype=float),
        jac=grad,
        method="BFGS",
    )
    return float(res.x[1])


def reliability_curve_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 5,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Compute binned reliability curve table (predicted vs observed probabilities).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class in [0, 1].
    n_bins : int, default=5
        Number of bins (5 is recommended for small-sample survey datasets).
    strategy : {'quantile', 'uniform'}, default='quantile'
        Binning strategy. 'quantile' creates roughly equal-sized bins.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()

    if strategy == "quantile":
        bins = pd.qcut(p, q=n_bins, duplicates="drop")
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = pd.cut(p, bins=edges, include_lowest=True)
    else:
        raise ValueError("strategy must be either 'quantile' or 'uniform'.")

    df_bins = pd.DataFrame({"y": y, "p": p, "bin": bins})
    grouped = (
        df_bins.groupby("bin", observed=True)
        .agg(n_obs=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
        .reset_index()
    )
    grouped["abs_diff"] = (grouped["mean_pred"] - grouped["obs_rate"]).abs()
    grouped["bin"] = grouped["bin"].astype(str)
    return grouped


def evaluate_calibration(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 5,
    strategy: str = "quantile",
) -> CalibrationReport:
    """Evaluate probability calibration and discrimination for binary predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for class 1 in [0, 1].
    n_bins : int, default=5
        Number of bins for reliability curve and Expected Calibration Error (ECE).
    strategy : {'quantile', 'uniform'}, default='quantile'
        Binning strategy for ECE and reliability table.

    Returns
    -------
    CalibrationReport
        Structured calibration diagnostics and interpretation notes.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()

    if len(y) != len(p):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y) == 0:
        raise ValueError("Input arrays must not be empty.")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only binary values 0 and 1.")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1) to evaluate calibration.")
    if np.any((p < 0.0) | (p > 1.0)) or np.isnan(p).any():
        raise ValueError("y_prob must contain finite probabilities in [0, 1].")

    n_obs = len(y)
    event_rate = float(np.mean(y))
    mean_pred = float(np.mean(p))

    logit_p = _safe_logit(p)
    citl = _compute_citl(y, logit_p)
    slope = _compute_calibration_slope(y, logit_p)

    brier = float(brier_score_loss(y, p))
    brier_null = event_rate * (1.0 - event_rate)
    brier_skill = float(1.0 - brier / brier_null) if brier_null > 0 else 0.0
    auc = float(roc_auc_score(y, p))

    rel_table = reliability_curve_data(y, p, n_bins=n_bins, strategy=strategy)
    ece = float(np.sum((rel_table["n_obs"] / n_obs) * rel_table["abs_diff"]))

    notes: list[str] = []
    if abs(citl) > 0.25:
        direction = "under-predicting" if citl > 0 else "over-predicting"
        notes.append(
            f"Systematic miscalibration (CITL={citl:+.3f}): model is {direction} average risk."
        )
    if np.isfinite(slope):
        if slope < 0.80:
            notes.append(
                f"Calibration slope ({slope:.3f} < 0.80) indicates overfitting (extreme predictions)."
            )
        elif slope > 1.25:
            notes.append(
                f"Calibration slope ({slope:.3f} > 1.25) indicates underfitting (over-shrunk predictions)."
            )
    if brier_skill < 0.0:
        notes.append(
            "Negative Scaled Brier score: predictions are worse than predicting the constant event rate."
        )

    if abs(citl) <= 0.15 and np.isfinite(slope) and 0.85 <= slope <= 1.15:
        status = "WELL-CALIBRATED"
    elif abs(citl) <= 0.30 and np.isfinite(slope) and 0.70 <= slope <= 1.30:
        status = "MODERATE"
    else:
        status = "MISCALIBRATED"

    return CalibrationReport(
        n_obs=n_obs,
        event_rate=event_rate,
        mean_predicted_prob=mean_pred,
        citl=citl,
        calibration_slope=slope,
        brier_score=brier,
        brier_skill_score=brier_skill,
        ece=ece,
        roc_auc=auc,
        reliability_table=rel_table,
        status=status,
        notes=notes,
    )

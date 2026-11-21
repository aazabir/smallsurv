"""Monte Carlo simulation utilities for comparing small-sample binary estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special

from smallsurv.firth import FirthLogisticRegression


@dataclass
class SimulationResult:
    """Structured Monte Carlo comparison across MLE, Firth, and FLIC estimators."""

    n_sim: int
    n_obs: int
    true_intercept: float
    true_coef: np.ndarray
    summary_df: pd.DataFrame
    raw_df: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return simulation metadata and estimator summary rows as dictionaries."""
        return {
            "n_sim": self.n_sim,
            "n_obs": self.n_obs,
            "true_intercept": self.true_intercept,
            "true_coef": self.true_coef.tolist(),
            "estimators": self.summary_df.to_dict(orient="records"),
        }

    def summary(self) -> str:
        """Return a formatted text table comparing estimators."""
        header = (
            f"Monte Carlo Simulation (N={self.n_obs}, n_sim={self.n_sim})\n"
            f"True Intercept={self.true_intercept:.3f}, True Coef={np.round(self.true_coef, 3).tolist()}\n"
            + "-" * 78
        )
        return header + "\n" + self.summary_df.to_string(index=False)

    def __str__(self) -> str:
        return self.summary()


def _fit_mle_unpenalized(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, bool]:
    """Fit standard unpenalized logistic regression via regularized BFGS fallback."""
    n_obs, n_feat = X.shape
    X_design = np.column_stack([np.ones(n_obs, dtype=float), X])

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X_design @ params
        nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        prob = special.expit(eta)
        grad = X_design.T @ (prob - y)
        return nll, grad

    res = optimize.minimize(
        loss_and_grad,
        x0=np.zeros(n_feat + 1, dtype=float),
        jac=True,
        method="BFGS",
    )
    return float(res.x[0]), res.x[1:].copy(), bool(res.success)


def run_monte_carlo(
    n_obs: int = 60,
    n_sim: int = 100,
    true_intercept: float = -1.0,
    true_coef: list[float] | np.ndarray | None = None,
    random_state: int = 42,
) -> SimulationResult:
    """Compare MLE, Firth, and FLIC on synthetic small-sample binary datasets.

    Parameters
    ----------
    n_obs : int, default=60
        Sample size per simulated dataset.
    n_sim : int, default=100
        Number of Monte Carlo replicates.
    true_intercept : float, default=-1.0
        True data-generating intercept on the log-odds scale.
    true_coef : array-like, optional
        True slope coefficients. Defaults to ``[0.8, -0.6]``.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    if n_obs < 10:
        raise ValueError("n_obs must be at least 10.")
    if n_sim < 5:
        raise ValueError("n_sim must be at least 5.")

    beta_true = (
        np.array([0.8, -0.6], dtype=float)
        if true_coef is None
        else np.asarray(true_coef, dtype=float).ravel()
    )
    if beta_true.size == 0:
        raise ValueError("true_coef must contain at least one coefficient.")

    n_feat = beta_true.size
    rng = np.random.default_rng(random_state)

    records: list[dict[str, Any]] = []
    for sim_idx in range(1, n_sim + 1):
        X = rng.normal(loc=0.0, scale=1.0, size=(n_obs, n_feat))
        eta_true = true_intercept + X @ beta_true
        p_true = special.expit(eta_true)
        y = rng.binomial(1, p_true, size=n_obs)

        if len(np.unique(y)) < 2:
            continue

        estimators: dict[str, tuple[float, np.ndarray, np.ndarray, bool]] = {}

        mle_b0, mle_b, mle_ok = _fit_mle_unpenalized(X, y)
        mle_p = special.expit(mle_b0 + X @ mle_b)
        estimators["mle"] = (mle_b0, mle_b, mle_p, mle_ok)

        firth_m = FirthLogisticRegression(flic=False).fit(X, y)
        estimators["firth"] = (
            float(firth_m.intercept_[0]),
            firth_m.coef_.ravel(),
            firth_m.predict_proba(X)[:, 1],
            bool(firth_m.converged_),
        )

        flic_m = FirthLogisticRegression(flic=True).fit(X, y)
        estimators["flic"] = (
            float(flic_m.intercept_[0]),
            flic_m.coef_.ravel(),
            flic_m.predict_proba(X)[:, 1],
            bool(flic_m.converged_),
        )

        for name, (b0_hat, b_hat, p_hat, ok) in estimators.items():
            coef_bias = float(np.mean(b_hat - beta_true))
            coef_rmse = float(np.sqrt(np.mean((b_hat - beta_true) ** 2)))
            intercept_bias = float(b0_hat - true_intercept)
            prob_rmse = float(np.sqrt(np.mean((p_hat - p_true) ** 2)))
            citl_error = float(np.mean(p_hat) - np.mean(y))
            records.append(
                {
                    "sim": sim_idx,
                    "estimator": name,
                    "converged": ok,
                    "coef_bias": coef_bias,
                    "coef_rmse": coef_rmse,
                    "intercept_bias": intercept_bias,
                    "prob_rmse": prob_rmse,
                    "citl_error": citl_error,
                }
            )

    raw_df = pd.DataFrame(records)
    summary_df = (
        raw_df.groupby("estimator", as_index=False)
        .agg(
            n_runs=("sim", "count"),
            conv_rate=("converged", "mean"),
            mean_coef_bias=("coef_bias", "mean"),
            coef_rmse=("coef_rmse", "mean"),
            intercept_bias=("intercept_bias", "mean"),
            prob_rmse=("prob_rmse", "mean"),
            abs_citl_err=("citl_error", lambda s: float(np.mean(np.abs(s)))),
        )
        .sort_values("estimator")
        .reset_index(drop=True)
    )

    return SimulationResult(
        n_sim=n_sim,
        n_obs=n_obs,
        true_intercept=float(true_intercept),
        true_coef=beta_true,
        summary_df=summary_df,
        raw_df=raw_df,
    )
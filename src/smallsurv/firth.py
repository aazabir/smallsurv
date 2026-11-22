"""Firth penalized logistic regression with optional FLIC intercept correction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid function."""
    return special.expit(np.clip(z, -500.0, 500.0))


def _covariance_from_information(
    X: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    """Covariance matrix as the inverse weighted Fisher information at given parameters.

    The information ``X^T W X`` is evaluated at the supplied intercept and slopes, so
    it describes the fitted model that is actually reported. For FLIC this means the
    corrected intercept, not the intercept of the underlying Firth fit.
    """
    design = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    beta = np.concatenate([[intercept], np.asarray(coef, dtype=float).ravel()])
    pi = _sigmoid(np.clip(design @ beta, -50.0, 50.0))
    wt = np.clip(np.asarray(sample_weight, dtype=float).ravel() * pi * (1.0 - pi), 1e-12, None)
    fisher = design.T @ (design * wt[:, None]) + 1e-10 * np.eye(design.shape[1])
    try:
        return np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(fisher)


@dataclass
class _FirthFitResult:
    intercept: float
    coef: np.ndarray
    cov_params: np.ndarray
    converged: bool
    n_iter: int


def _fit_firth_mle(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> _FirthFitResult:
    """Fit weighted Firth penalized logistic regression using Newton-Raphson / Fisher scoring.

    Maximizes penalized log-likelihood:
        l*(beta) = l(beta) + 0.5 * log(det(I(beta)))
    where l(beta) is the weighted Bernoulli log-likelihood and I(beta) is the
    weighted Fisher information matrix.
    """
    n_samples, n_features = X.shape
    design = np.column_stack([np.ones(n_samples, dtype=float), X])
    n_params = n_features + 1
    w = np.asarray(sample_weight, dtype=float).ravel()

    def pen_loglik(b_vec: np.ndarray) -> float:
        """Weighted Firth penalized log-likelihood at ``b_vec``."""
        e = np.clip(design @ b_vec, -50.0, 50.0)
        p = _sigmoid(e)
        log_p = np.log(np.clip(p, 1e-15, 1.0))
        log_q = np.log(np.clip(1.0 - p, 1e-15, 1.0))
        ll = float(np.sum(w * (y * log_p + (1.0 - y) * log_q)))
        w_p = np.clip(w * p * (1.0 - p), 1e-12, None)
        info = design.T @ (design * w_p[:, None]) + 1e-10 * np.eye(n_params)
        sign, logdet = np.linalg.slogdet(info)
        penalty = 0.5 * float(logdet) if sign > 0 else -1e6
        return ll + penalty

    beta = np.zeros(n_params, dtype=float)
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        eta = np.clip(design @ beta, -50.0, 50.0)
        pi = _sigmoid(eta)
        wt = np.clip(w * pi * (1.0 - pi), 1e-12, None)

        # Weighted Fisher information matrix: X^T W X
        WX = design * wt[:, None]
        fisher = design.T @ WX + 1e-10 * np.eye(n_params)

        # Hat matrix diagonal h_i = w_i * x_i^T I^{-1} x_i
        # Solve fisher @ V_part = design.T -> h = sum(design * (fisher^{-1} design^T)^T, axis=1) * wt
        inv_fisher_XT = np.linalg.solve(fisher, design.T)
        hat_diag = np.sum(design * inv_fisher_XT.T, axis=1) * wt
        hat_diag = np.clip(hat_diag, 0.0, 0.999999)

        # Firth modified score: U*(beta)_j = sum_i w_i * (y_i - pi_i + h_i * (0.5 - pi_i)) * x_{ij}
        adj_resid = w * (y - pi) + hat_diag * (0.5 - pi)
        score = design.T @ adj_resid

        step_norm = float(np.max(np.abs(score)))
        if step_norm < tol and it > 1:
            converged = True
            break

        # Damped Newton / Fisher scoring step
        try:
            delta = np.linalg.solve(fisher, score)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(fisher, score, rcond=None)

        # Step-halving line search on penalized log-likelihood
        curr_obj = pen_loglik(beta)
        step = 1.0
        accepted = False
        for _ in range(25):
            cand = beta + step * delta
            if np.max(np.abs(cand)) > 50.0:
                step *= 0.5
                continue
            if pen_loglik(cand) >= curr_obj - 1e-12:
                beta = cand
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # No step improved the penalized likelihood. Report this honestly instead of
            # taking a blind step or declaring convergence.
            break

        # Parameter change is only a convergence signal for a full, undamped step. A
        # tiny change after repeated halving says nothing about the score.
        if step == 1.0 and float(np.max(np.abs(delta))) < tol:
            converged = True
            break

    cov_params = _covariance_from_information(X, float(beta[0]), beta[1:], w)

    return _FirthFitResult(
        intercept=float(beta[0]),
        coef=beta[1:].copy(),
        cov_params=cov_params,
        converged=converged,
        n_iter=n_iter,
    )


def _apply_flic_intercept(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Firth's Logistic Regression with Intercept Correction (Puhr et al., 2017).

    Keeps slope estimates `coef` fixed and solves for intercept `alpha0` such that
    the weighted mean predicted probability equals the weighted observed event rate:
        sum_i w_i * sigmoid(alpha0 + x_i^T beta) = sum_i w_i * y_i
    """
    if X.shape[1] == 0:
        linear_pred = np.zeros(X.shape[0], dtype=float)
    else:
        linear_pred = X @ coef

    w = np.asarray(sample_weight, dtype=float).ravel()
    target_weighted_events = float(np.sum(w * y))

    def score_fn(alpha0: float) -> float:
        probs = _sigmoid(alpha0 + linear_pred)
        return float(np.sum(w * probs) - target_weighted_events)

    low, high = -10.0, 10.0
    f_low, f_high = score_fn(low), score_fn(high)
    for _ in range(30):
        if f_low <= 0.0 <= f_high or f_high <= 0.0 <= f_low:
            break
        low *= 2.0
        high *= 2.0
        f_low, f_high = score_fn(low), score_fn(high)

    try:
        root = optimize.brentq(score_fn, low, high, xtol=1e-10, maxiter=200)
        return float(root)
    except ValueError:
        res = optimize.minimize_scalar(lambda a: score_fn(float(a)) ** 2, method="bounded", bounds=(low, high))
        return float(res.x)


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """Firth penalized logistic regression with optional FLIC intercept correction.

    Parameters
    ----------
    flic : bool, default=True
        Whether to apply Firth's Logistic Regression with Intercept Correction
        (FLIC, Puhr et al., 2017) so average predicted probabilities match the
        observed event rate.
    max_iter : int, default=200
        Maximum number of Newton-Raphson / Fisher scoring iterations.
    tol : float, default=1e-7
        Convergence tolerance on score norm / parameter step.
    conf_level : float, default=0.95
        Confidence level for Wald intervals in ``summary_table()``.

    Notes
    -----
    Inference in ``summary_table()`` is Wald based. Standard errors are the square
    roots of the diagonal of the inverse weighted Fisher information evaluated at the
    reported parameters, and confidence intervals and p-values use the normal
    approximation. Other Firth implementations, such as ``logistf`` in R, report
    profile penalized likelihood intervals and penalized likelihood ratio tests by
    default. These can differ from Wald intervals, especially for rare events and
    near separation, where Wald intervals are less reliable.

    With ``flic=True`` the slope coefficients are identical to the Firth fit and only
    the intercept changes. The standard errors, including those of the slopes, are
    computed at the corrected model, so they differ slightly from ``flic=False``. This
    matches the ``var`` matrix returned by ``flic()`` in ``logistf``, the inverse of
    ``X^T W X`` with weights from the corrected probabilities. The intervals and
    p-values that ``logistf`` prints for FLIC are not identical to the Wald values
    here. It uses the standard error of an intercept-only refit for the intercept, and
    carries the slope intervals and p-values over from the Firth fit.

    ``sample_weight`` is treated as a frequency weight. Integer weights give the same
    fit as repeating each row that many times. The reported standard errors do not
    account for survey design features such as clustering or stratification.
    """

    def __init__(
        self,
        flic: bool = True,
        max_iter: int = 200,
        tol: float = 1e-7,
        conf_level: float = 0.95,
    ) -> None:
        self.flic = flic
        self.max_iter = max_iter
        self.tol = tol
        self.conf_level = conf_level

    def _validate_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
        reset: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if isinstance(X, pd.DataFrame):
            if reset:
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if reset and hasattr(self, "feature_names_in_"):
                delattr(self, "feature_names_in_")

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X_arr.ndim}D array instead.")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("X must be non-empty with at least one feature.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X contains NaN or infinite values.")

        if reset:
            self.n_features_in_ = int(X_arr.shape[1])
        elif hasattr(self, "n_features_in_") and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but model was fitted with {self.n_features_in_} features."
            )

        if y is None:
            return X_arr, None, None

        y_arr = np.asarray(y).ravel()
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if np.any(pd.isna(y_arr)):
            raise ValueError("y contains NaN values.")

        unique_vals = np.unique(y_arr)
        if len(unique_vals) != 2:
            raise ValueError(f"Target y must contain exactly 2 classes, got {len(unique_vals)}.")

        if set(unique_vals).issubset({0, 1}):
            y_bin = y_arr.astype(float)
            self.classes_ = np.array([0, 1], dtype=int)
        else:
            self.classes_ = unique_vals
            y_bin = (y_arr == self.classes_[1]).astype(float)

        w_arr: np.ndarray | None = None
        if sample_weight is not None:
            w_arr = np.asarray(sample_weight, dtype=float).ravel()
            if w_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must have the same length as y.")
            if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
                raise ValueError("sample_weight must contain non-negative finite values.")
            if float(np.sum(w_arr)) <= 0.0:
                raise ValueError("Sum of sample_weight must be positive.")

        return X_arr, y_bin, w_arr

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> FirthLogisticRegression:
        """Fit Firth penalized logistic regression model."""
        if not (0.0 < self.conf_level < 1.0):
            raise ValueError("conf_level must be strictly between 0 and 1.")

        X_arr, y_bin, w_arr = self._validate_inputs(
            X, y, sample_weight=sample_weight, reset=True
        )
        assert y_bin is not None
        weights = np.ones(X_arr.shape[0], dtype=float) if w_arr is None else w_arr

        fit_res = _fit_firth_mle(
            X_arr,
            y_bin,
            sample_weight=weights,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        if not fit_res.converged:
            warnings.warn(
                f"Firth fitting did not converge after {fit_res.n_iter} iterations. "
                "Estimates may be unreliable. Increase max_iter, rescale the predictors "
                "or check the data with audit_data().",
                ConvergenceWarning,
                stacklevel=2,
            )

        intercept = fit_res.intercept
        cov_params = fit_res.cov_params
        if self.flic:
            intercept = _apply_flic_intercept(
                X_arr, y_bin, fit_res.coef, sample_weight=weights
            )
            # The reported model uses the corrected intercept, so the standard errors
            # must come from the information at that model, not at the Firth fit.
            cov_params = _covariance_from_information(
                X_arr, intercept, fit_res.coef, weights
            )

        self.intercept_ = np.array([intercept], dtype=float)
        self.coef_ = fit_res.coef.reshape(1, -1)
        self.cov_params_ = cov_params
        self.converged_ = fit_res.converged
        self.n_iter_ = fit_res.n_iter
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise ValueError("This FirthLogisticRegression instance is not fitted yet.")

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute log-odds (linear predictor) for samples in X."""
        self._check_is_fitted()
        X_arr, _, _ = self._validate_inputs(X, y=None, reset=False)
        return (X_arr @ self.coef_.ravel() + self.intercept_[0]).ravel()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities of shape (n_samples, 2)."""
        scores = self.decision_function(X)
        prob_pos = _sigmoid(scores)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary class labels for samples in X."""
        probs = self.predict_proba(X)[:, 1]
        indices = (probs >= 0.5).astype(int)
        return self.classes_[indices]

    def summary_table(self) -> pd.DataFrame:
        """Return a summary table with coefficients, SEs, ORs, CIs, z, and p-values.

        Confidence intervals and p-values are Wald based (see the class notes), not
        profile penalized likelihood intervals.
        """
        self._check_is_fitted()

        if hasattr(self, "feature_names_in_"):
            feat_names = [str(c) for c in self.feature_names_in_]
        else:
            feat_names = [f"x{i}" for i in range(self.coef_.shape[1])]

        terms = ["(Intercept)"] + feat_names
        estimates = np.concatenate([self.intercept_, self.coef_.ravel()])
        se = np.sqrt(np.clip(np.diag(self.cov_params_), 0.0, None))

        z_stats = np.divide(
            estimates,
            se,
            out=np.full_like(estimates, np.nan),
            where=se > 0,
        )
        p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stats)))

        z_crit = float(stats.norm.ppf(0.5 + self.conf_level / 2.0))
        ci_low = estimates - z_crit * se
        ci_high = estimates + z_crit * se

        odds_ratio = np.exp(estimates)
        or_ci_low = np.exp(ci_low)
        or_ci_high = np.exp(ci_high)

        return pd.DataFrame(
            {
                "term": terms,
                "coef": estimates,
                "se": se,
                "odds_ratio": odds_ratio,
                "or_ci_low": or_ci_low,
                "or_ci_high": or_ci_high,
                "z_stat": z_stats,
                "p_value": p_values,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return fitted model metadata and summary rows."""
        self._check_is_fitted()
        return {
            "flic": self.flic,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "intercept": float(self.intercept_[0]),
            "coef": self.coef_.ravel().tolist(),
            "summary": self.summary_table().to_dict(orient="records"),
        }

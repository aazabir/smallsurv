"""Firth penalized logistic regression with FLIC intercept correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.base import BaseEstimator, ClassifierMixin


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """Firth's bias-reduced logistic regression with optional FLIC correction.

    Implements Jeffreys-invariant prior penalization (Firth, 1993) to eliminate
    first-order bias and resolve complete or quasi-complete separation in
    small-sample binary survey data. When ``flic=True`` (default), applies
    Firth's Logistic regression with Intercept Correction (Puhr et al., 2017)
    so that the average predicted probability matches the observed event rate.

    Parameters
    ----------
    flic : bool, default=True
        If True, refit the intercept via unpenalized maximum likelihood with
        the Firth linear predictor held fixed as an offset (calibration-in-the-large).
    max_iter : int, default=100
        Maximum number of Newton-Raphson iterations.
    tol : float, default=1e-6
        Convergence tolerance on the maximum absolute parameter update.
    max_step : float, default=5.0
        Maximum allowed absolute step size per parameter in a single iteration.
    """

    def __init__(
        self,
        flic: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_step: float = 5.0,
    ) -> None:
        self.flic = flic
        self.max_iter = max_iter
        self.tol = tol
        self.max_step = max_step

    @staticmethod
    def _penalized_loglik(
        X_design: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Firth penalized log-likelihood, probabilities, Fisher info, and hat diag."""
        eta = X_design @ beta
        p = special.expit(eta)
        # Numerically stable log-likelihood: sum(y * eta - log(1 + exp(eta)))
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))

        w = np.clip(p * (1.0 - p), 1e-12, None)
        Xw = X_design * np.sqrt(w)[:, None]
        fisher = Xw.T @ Xw

        # Add tiny jitter only if Fisher matrix is ill-conditioned
        reg = 0.0
        sign, logdet = np.linalg.slogdet(fisher)
        if sign <= 0 or not np.isfinite(logdet):
            reg = 1e-10
            fisher = fisher + reg * np.eye(fisher.shape[0])
            sign, logdet = np.linalg.slogdet(fisher)

        try:
            fisher_inv = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            fisher_inv = np.linalg.pinv(fisher)

        # Hat matrix diagonal: h_i = w_i * x_i^T I^{-1} x_i
        hat_diag = np.sum((Xw @ fisher_inv) * Xw, axis=1)
        penalized_ll = ll + 0.5 * float(logdet)
        return penalized_ll, p, fisher_inv, hat_diag

    @staticmethod
    def _flic_intercept(offset: np.ndarray, y: np.ndarray, init_b0: float) -> float:
        """Solve sum(sigmoid(b0 + offset)) = sum(y) for calibration-in-the-large."""
        target_sum = float(np.sum(y))

        def score(b0: float) -> float:
            return float(np.sum(special.expit(b0 + offset)) - target_sum)

        low, high = init_b0 - 2.0, init_b0 + 2.0
        for _ in range(30):
            if score(low) <= 0.0 <= score(high):
                break
            low -= 5.0
            high += 5.0

        return float(optimize.brentq(score, low, high, xtol=1e-10, maxiter=200))

    def _prepare_X(self, X: pd.DataFrame | np.ndarray, fit: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if fit:
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            arr = X.to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {arr.ndim}D.")
        if np.isnan(arr).any():
            raise ValueError("Input X contains NaN values. Impute missing values before fitting.")
        return arr

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> FirthLogisticRegression:
        """Fit Firth (or FLIC) logistic regression model."""
        X_arr = self._prepare_X(X, fit=True)
        y_arr = np.asarray(y)

        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of observations.")

        classes = np.unique(y_arr)
        if len(classes) != 2:
            raise ValueError(f"Target y must have exactly 2 classes; found {len(classes)}.")

        self.classes_ = classes
        y_bin = (y_arr == classes[1]).astype(float)
        n_obs, n_features = X_arr.shape
        self.n_features_in_ = n_features

        # Design matrix with intercept in column 0
        X_design = np.column_stack([np.ones(n_obs, dtype=float), X_arr])
        beta = np.zeros(n_features + 1, dtype=float)
        # Initialize intercept at log-odds of event rate
        p_bar = np.clip(np.mean(y_bin), 1e-6, 1.0 - 1e-6)
        beta[0] = float(np.log(p_bar / (1.0 - p_bar)))

        converged = False
        n_iter = 0
        current_ll, p, fisher_inv, hat_diag = self._penalized_loglik(X_design, y_bin, beta)

        for it in range(1, self.max_iter + 1):
            n_iter = it
            # Modified Firth score vector
            score = X_design.T @ (y_bin - p + hat_diag * (0.5 - p))
            step = fisher_inv @ score

            max_abs_step = float(np.max(np.abs(step)))
            if max_abs_step > self.max_step:
                step = step * (self.max_step / max_abs_step)

            # Step-halving line search to guarantee ascent on penalized log-likelihood
            alpha = 1.0
            accepted = False
            for _ in range(25):
                beta_cand = beta + alpha * step
                cand_ll, p_cand, fisher_inv_cand, hat_cand = self._penalized_loglik(
                    X_design, y_bin, beta_cand
                )
                if np.isfinite(cand_ll) and cand_ll >= current_ll - 1e-10:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                beta_cand = beta + alpha * step
                cand_ll, p_cand, fisher_inv_cand, hat_cand = self._penalized_loglik(
                    X_design, y_bin, beta_cand
                )

            delta = float(np.max(np.abs(beta_cand - beta)))
            beta = beta_cand
            current_ll = cand_ll
            p = p_cand
            fisher_inv = fisher_inv_cand
            hat_diag = hat_cand

            if delta < self.tol:
                converged = True
                break

        self.firth_intercept_ = float(beta[0])
        slopes = beta[1:].copy()

        if self.flic:
            offset = X_arr @ slopes
            intercept = self._flic_intercept(offset, y_bin, init_b0=self.firth_intercept_)
        else:
            intercept = self.firth_intercept_

        self.intercept_ = np.array([intercept], dtype=float)
        self.coef_ = slopes.reshape(1, -1)
        self.cov_ = fisher_inv
        self.se_ = np.sqrt(np.clip(np.diag(fisher_inv), 0.0, None))
        self.loglik_penalized_ = float(current_ll)
        self.n_iter_ = n_iter
        self.converged_ = converged
        return self

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute linear predictor (log-odds) for samples in X."""
        if not hasattr(self, "coef_"):
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        X_arr = self._prepare_X(X, fit=False)
        return self.intercept_[0] + X_arr @ self.coef_.ravel()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(y=0), P(y=1)]."""
        eta = self.decision_function(X)
        p1 = special.expit(eta)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels using the specified probability threshold."""
        p1 = self.predict_proba(X)[:, 1]
        return np.where(p1 >= threshold, self.classes_[1], self.classes_[0])

    def summary_table(
        self,
        feature_names: list[str] | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Return a coefficient summary table with Odds Ratios and Wald CIs."""
        if not hasattr(self, "coef_"):
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        if feature_names is not None:
            names = ["(Intercept)"] + list(feature_names)
        elif hasattr(self, "feature_names_in_"):
            names = ["(Intercept)"] + [str(c) for c in self.feature_names_in_]
        else:
            names = ["(Intercept)"] + [f"x{i}" for i in range(self.n_features_in_)]

        params = np.concatenate([self.intercept_, self.coef_.ravel()])
        se = self.se_
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_stat = np.divide(params, se, out=np.zeros_like(params), where=se > 0)
        p_val = 2.0 * stats.norm.sf(np.abs(z_stat))

        ci_low = params - z_crit * se
        ci_high = params + z_crit * se

        return pd.DataFrame(
            {
                "term": names,
                "coef": params,
                "se": se,
                "odds_ratio": np.exp(params),
                "or_ci_low": np.exp(ci_low),
                "or_ci_high": np.exp(ci_high),
                "z_stat": z_stat,
                "p_value": p_val,
            }
        )
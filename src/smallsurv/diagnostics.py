"""Diagnostic audit and statistical guardrails for small-sample binary survey data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DiagnosticReport:
    """Structured diagnostic audit result for a small-sample binary dataset."""

    n_obs: int
    n_events: int
    n_non_events: int
    event_rate: float
    n_predictors: int
    n_parameters: int
    epv: float
    riley_min_n: int
    riley_breakdown: dict[str, int]
    missing_summary: dict[str, float]
    zero_variance_cols: list[str]
    rare_categories: dict[str, dict[str, int]]
    separation_flags: dict[str, list[str]]
    high_collinearity: list[tuple[str, str, float]]
    risk_level: str
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the diagnostic report as a plain dictionary."""
        return {
            "n_obs": self.n_obs,
            "n_events": self.n_events,
            "n_non_events": self.n_non_events,
            "event_rate": round(self.event_rate, 4),
            "n_predictors": self.n_predictors,
            "n_parameters": self.n_parameters,
            "epv": round(self.epv, 2),
            "riley_min_n": self.riley_min_n,
            "riley_breakdown": self.riley_breakdown,
            "missing_summary": self.missing_summary,
            "zero_variance_cols": self.zero_variance_cols,
            "rare_categories": self.rare_categories,
            "separation_flags": self.separation_flags,
            "high_collinearity": self.high_collinearity,
            "risk_level": self.risk_level,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        """Return a human-readable diagnostic summary report."""
        lines = [
            "=" * 64,
            "smallsurv Diagnostic Audit Report",
            "=" * 64,
            f"Overall Risk Level : {self.risk_level}",
            f"Sample Size (N)    : {self.n_obs}",
            f"Events (y=1 / y=0) : {self.n_events} / {self.n_non_events} "
            f"(Event rate = {self.event_rate:.1%})",
            f"Predictors / Params: {self.n_predictors} columns / {self.n_parameters} parameters",
            f"Events Per Var(EPV): {self.epv:.2f}",
            f"Riley (2020) Min N : {self.riley_min_n} "
            f"(C1={self.riley_breakdown['criterion_1_shrinkage']}, "
            f"C2={self.riley_breakdown['criterion_2_r2_optimism']}, "
            f"C3={self.riley_breakdown['criterion_3_precision']})",
            "-" * 64,
        ]

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        else:
            lines.append("Warnings: None detected.")

        if self.recommendations:
            lines.append("-" * 64)
            lines.append("Recommended Guardrails:")
            for rec in self.recommendations:
                lines.append(f"  -> {rec}")

        lines.append("=" * 64)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def riley_min_sample_size(
    n_parameters: int,
    event_rate: float,
    anticipated_r2_cs: float | None = None,
    shrinkage: float = 0.90,
    margin_of_error: float = 0.05,
) -> tuple[int, dict[str, int]]:
    """Compute minimum sample size for binary prediction models (Riley et al., 2020).

    Parameters
    ----------
    n_parameters : int
        Number of candidate predictor parameters (excluding intercept).
    event_rate : float
        Outcome event proportion in (0, 1).
    anticipated_r2_cs : float, optional
        Anticipated Cox-Snell R-squared. Defaults to 15% of the maximum possible
        Cox-Snell R-squared when no prior estimate is supplied.
    shrinkage : float, default=0.90
        Target global shrinkage factor to limit overfitting.
    margin_of_error : float, default=0.05
        Acceptable margin of error for estimating the overall event proportion.

    Returns
    -------
    tuple[int, dict[str, int]]
        Minimum required sample size and per-criterion breakdown.
    """
    if n_parameters < 1:
        raise ValueError("n_parameters must be at least 1.")
    if not (0.0 < event_rate < 1.0):
        raise ValueError("event_rate must be strictly between 0 and 1.")

    max_r2_cs = 1.0 - ((event_rate**event_rate) * ((1.0 - event_rate) ** (1.0 - event_rate))) ** 2
    r2_cs = 0.15 * max_r2_cs if anticipated_r2_cs is None else anticipated_r2_cs
    r2_cs = min(max(r2_cs, 1e-4), max_r2_cs * 0.99)

    # Criterion 1: Global shrinkage >= target shrinkage
    n1 = math.ceil(n_parameters / ((shrinkage - 1.0) * math.log(1.0 - r2_cs / shrinkage)))

    # Criterion 2: Small absolute optimism (<= 0.05) in Nagelkerke R-squared
    s_new = r2_cs / (r2_cs + 0.05 * max_r2_cs)
    n2 = math.ceil(n_parameters / ((s_new - 1.0) * math.log(1.0 - r2_cs / s_new)))

    # Criterion 3: Precise estimation of overall outcome proportion
    n3 = math.ceil(((1.96 / margin_of_error) ** 2) * event_rate * (1.0 - event_rate))

    breakdown = {
        "criterion_1_shrinkage": int(n1),
        "criterion_2_r2_optimism": int(n2),
        "criterion_3_precision": int(n3),
    }
    return max(n1, n2, n3), breakdown


def audit_data(
    df: pd.DataFrame,
    target: str,
    predictors: list[str] | None = None,
    min_category_count: int = 10,
    collinearity_threshold: float = 0.85,
    anticipated_r2_cs: float | None = None,
) -> DiagnosticReport:
    """Run a comprehensive small-sample diagnostic audit on binary survey data.

    Parameters
    ----------
    df : pd.DataFrame
        Input survey dataset.
    target : str
        Name of the binary outcome column (coded 0/1 or two unique values).
    predictors : list[str], optional
        List of predictor column names. If None, uses all columns except `target`.
    min_category_count : int, default=10
        Minimum count per categorical level before flagging as a rare category.
    collinearity_threshold : float, default=0.85
        Absolute Pearson correlation threshold for flagging numeric collinearity.
    anticipated_r2_cs : float, optional
        Anticipated Cox-Snell R-squared for Riley et al. (2020) sample size calculation.

    Returns
    -------
    DiagnosticReport
        Structured audit results with warnings and guardrail recommendations.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    if predictors is None:
        predictors = [c for c in df.columns if c != target]
    else:
        missing_cols = [c for c in predictors if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Predictor columns not found in DataFrame: {missing_cols}")

    if not predictors:
        raise ValueError("At least one predictor column is required.")

    valid_mask = df[target].notna()
    data = df.loc[valid_mask].copy()
    y_raw = data[target]
    unique_vals = sorted(y_raw.unique().tolist())
    if len(unique_vals) != 2:
        raise ValueError(
            f"Target column '{target}' must be binary (2 unique non-null values); "
            f"found {len(unique_vals)}: {unique_vals}"
        )

    if set(unique_vals) == {0, 1}:
        y = y_raw.astype(int)
    else:
        y = (y_raw == unique_vals[1]).astype(int)

    n_obs = int(len(data))
    n_events = int(y.sum())
    n_non_events = n_obs - n_events
    minority_events = min(n_events, n_non_events)
    event_rate = n_events / n_obs

    missing_summary: dict[str, float] = {}
    zero_variance_cols: list[str] = []
    rare_categories: dict[str, dict[str, int]] = {}
    separation_flags: dict[str, list[str]] = {}
    numeric_cols: list[str] = []
    n_parameters = 0

    for col in predictors:
        series = data[col]
        miss_pct = float(series.isna().mean() * 100.0)
        if miss_pct > 0.0:
            missing_summary[col] = round(miss_pct, 2)

        n_unique = int(series.nunique(dropna=True))
        if n_unique <= 1:
            zero_variance_cols.append(col)
            continue

        is_categorical = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        )

        if is_categorical:
            n_parameters += max(1, n_unique - 1)
            counts = series.value_counts(dropna=True)
            rares = {str(k): int(v) for k, v in counts.items() if v < min_category_count}
            if rares:
                rare_categories[col] = rares

            ct = pd.crosstab(series, y)
            sep_levels = [
                str(lvl) for lvl, row in ct.iterrows() if (row == 0).any() and row.sum() > 0
            ]
            if sep_levels:
                separation_flags[col] = sep_levels
        else:
            numeric_cols.append(col)
            n_parameters += 1
            valid_num = series.notna()
            x0 = series[valid_num & (y == 0)]
            x1 = series[valid_num & (y == 1)]
            if len(x0) > 0 and len(x1) > 0:
                if x0.max() <= x1.min() or x1.max() <= x0.min():
                    separation_flags[col] = ["univariate_numeric_separation"]

    n_parameters = max(1, n_parameters)
    epv = minority_events / n_parameters
    riley_min_n, riley_breakdown = riley_min_sample_size(
        n_parameters=n_parameters,
        event_rate=event_rate,
        anticipated_r2_cs=anticipated_r2_cs,
    )

    high_collinearity: list[tuple[str, str, float]] = []
    if len(numeric_cols) >= 2:
        corr_matrix = data[numeric_cols].corr().abs()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r_val = float(corr_matrix.iloc[i, j])
                if not np.isnan(r_val) and r_val >= collinearity_threshold:
                    high_collinearity.append((numeric_cols[i], numeric_cols[j], round(r_val, 3)))

    warnings: list[str] = []
    recommendations: list[str] = []

    if epv < 10.0:
        warnings.append(
            f"Low Events Per Variable (EPV = {epv:.2f} < 10). "
            "Standard MLE logistic regression risks severe overfitting and biased odds ratios."
        )
        recommendations.append(
            "Use Firth penalized logistic regression with intercept correction (FLIC) or ElasticNet."
        )
    elif epv < 15.0:
        warnings.append(
            f"Moderate EPV ({epv:.2f} < 15). Penalization or shrinkage is advisable."
        )

    if n_obs < riley_min_n:
        warnings.append(
            f"Sample size (N={n_obs}) is below Riley et al. (2020) minimum ({riley_min_n})."
        )
        recommendations.append(
            "Avoid complex tree/boosting models (XGBoost/RF) without strict calibration checks; "
            "consider reducing candidate parameters."
        )

    if separation_flags:
        sep_cols = ", ".join(separation_flags.keys())
        warnings.append(
            f"Quasi-complete or complete separation detected in: {sep_cols}. "
            "Unpenalized MLE may fail to converge or produce infinite coefficients."
        )
        recommendations.append(
            "Use FLIC (Firth logistic regression) to guarantee finite, well-calibrated estimates."
        )

    if rare_categories:
        rare_cols = ", ".join(rare_categories.keys())
        warnings.append(f"Rare category levels (< {min_category_count} obs) in: {rare_cols}.")
        recommendations.append("Collapse rare categories into 'Other' before fitting models.")

    if zero_variance_cols:
        warnings.append(f"Zero-variance columns detected: {', '.join(zero_variance_cols)}.")
        recommendations.append("Drop zero-variance columns before modeling.")

    if high_collinearity:
        pairs = ", ".join(f"{a}~{b} (r={r})" for a, b, r in high_collinearity)
        warnings.append(f"High numeric collinearity detected: {pairs}.")
        recommendations.append("Consider ElasticNet or dropping redundant collinear predictors.")

    if missing_summary:
        high_miss = [f"{k} ({v}%)" for k, v in missing_summary.items() if v >= 5.0]
        if high_miss:
            warnings.append(f"Notable missingness (>=5%) in: {', '.join(high_miss)}.")
            recommendations.append(
                "Perform imputation inside cross-validation folds to prevent data leakage."
            )

    if epv < 10.0 or bool(separation_flags) or n_obs < 0.75 * riley_min_n:
        risk_level = "HIGH"
    elif epv < 15.0 or n_obs < riley_min_n or bool(rare_categories) or bool(high_collinearity):
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Deduplicate recommendations while preserving order
    recommendations = list(dict.fromkeys(recommendations))

    return DiagnosticReport(
        n_obs=n_obs,
        n_events=n_events,
        n_non_events=n_non_events,
        event_rate=event_rate,
        n_predictors=len(predictors),
        n_parameters=n_parameters,
        epv=epv,
        riley_min_n=riley_min_n,
        riley_breakdown=riley_breakdown,
        missing_summary=missing_summary,
        zero_variance_cols=zero_variance_cols,
        rare_categories=rare_categories,
        separation_flags=separation_flags,
        high_collinearity=high_collinearity,
        risk_level=risk_level,
        warnings=warnings,
        recommendations=recommendations,
    )
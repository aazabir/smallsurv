import numpy as np
import pandas as pd

from smallsurv import (
    FirthLogisticRegression,
    ValidationReport,
    bootstrap_optimism_validate,
    repeated_cv_validate,
)


def test_repeated_cv_validate_runs():
    rng = np.random.default_rng(42)
    n = 90
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    logits = -0.4 + 0.9 * X["x1"] - 0.6 * X["x2"]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    estimator = FirthLogisticRegression(flic=True)
    report = repeated_cv_validate(
        estimator, X, y, n_splits=5, n_repeats=2, random_state=42
    )

    assert isinstance(report, ValidationReport)
    assert report.method == "repeated_cv"
    assert report.n_obs == n
    assert report.n_iterations == 10
    assert report.metric_std is not None
    assert 0.5 <= report.validated_metrics["roc_auc"] <= 1.0
    assert "smallsurv Internal Validation Report (repeated_cv)" in report.summary()
    assert report.to_dict()["method"] == "repeated_cv"


def test_bootstrap_optimism_validate_runs():
    rng = np.random.default_rng(123)
    n = 70
    X = rng.normal(size=(n, 3))
    logits = -0.2 + 0.8 * X[:, 0]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    estimator = FirthLogisticRegression(flic=True)
    report = bootstrap_optimism_validate(
        estimator, X, y, n_bootstraps=15, random_state=123
    )

    assert isinstance(report, ValidationReport)
    assert report.method == "bootstrap_optimism"
    assert report.n_obs == n
    assert report.n_iterations == 15
    assert report.apparent_metrics is not None
    assert report.optimism is not None
    assert "roc_auc" in report.validated_metrics
    assert "Optimism-Corrected Performance:" in report.summary()
    assert report.to_dict()["method"] == "bootstrap_optimism"

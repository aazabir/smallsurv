import numpy as np
import pytest

from smallsurv import (
    CalibrationReport,
    FirthLogisticRegression,
    evaluate_calibration,
    reliability_curve_data,
)


def test_evaluate_calibration_well_calibrated():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, 2))
    logits = -0.5 + 1.0 * X[:, 0] - 0.8 * X[:, 1]
    true_p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, true_p)

    model = FirthLogisticRegression(flic=True).fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    report = evaluate_calibration(y, probs, n_bins=5)

    assert isinstance(report, CalibrationReport)
    assert report.n_obs == n
    assert abs(report.citl) < 0.05
    assert 0.85 <= report.calibration_slope <= 1.15
    assert report.brier_skill_score > 0.0
    assert report.roc_auc > 0.70
    assert report.status == "WELL-CALIBRATED"
    assert "smallsurv Calibration & Discrimination Report" in report.summary()
    assert report.to_dict()["status"] == "WELL-CALIBRATED"


def test_evaluate_calibration_detects_miscalibration():
    rng = np.random.default_rng(10)
    n = 300
    x = rng.normal(size=n)
    true_logits = -0.3 + 0.8 * x
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-true_logits)))

    # Artificially overconfident and shifted predictions
    bad_logits = 1.2 + 3.0 * true_logits
    bad_probs = 1.0 / (1.0 + np.exp(-bad_logits))

    report = evaluate_calibration(y, bad_probs, n_bins=5)

    assert report.status == "MISCALIBRATED"
    assert report.calibration_slope < 0.80
    assert abs(report.citl) > 0.25
    assert len(report.notes) >= 2


def test_reliability_curve_data_strategies():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.4, size=100)
    p = rng.uniform(0.05, 0.95, size=100)

    q_df = reliability_curve_data(y, p, n_bins=5, strategy="quantile")
    u_df = reliability_curve_data(y, p, n_bins=5, strategy="uniform")

    assert list(q_df.columns) == ["bin", "n_obs", "mean_pred", "obs_rate", "abs_diff"]
    assert q_df["n_obs"].sum() == 100
    assert u_df["n_obs"].sum() == 100

    with pytest.raises(ValueError, match="strategy"):
        reliability_curve_data(y, p, strategy="invalid")


def test_evaluate_calibration_validation_errors():
    with pytest.raises(ValueError, match="same length"):
        evaluate_calibration([0, 1], [0.2])

    with pytest.raises(ValueError, match="empty"):
        evaluate_calibration([], [])

    with pytest.raises(ValueError, match="binary"):
        evaluate_calibration([0, 1, 2], [0.1, 0.5, 0.9])

    with pytest.raises(ValueError, match="both classes"):
        evaluate_calibration([1, 1, 1], [0.2, 0.5, 0.8])

    with pytest.raises(ValueError, match="finite probabilities"):
        evaluate_calibration([0, 1], [0.2, 1.5])
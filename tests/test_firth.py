import numpy as np
import pandas as pd
import pytest

from smallsurv import FirthLogisticRegression


def test_firth_handles_complete_separation():
    # Complete separation: all x > 0 have y = 1, all x < 0 have y = 0
    X = np.array([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = FirthLogisticRegression(flic=True).fit(X, y)

    assert model.converged_ is True
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.intercept_))
    assert model.coef_[0, 0] > 0

    probs = model.predict_proba(X)[:, 1]
    assert np.all((probs > 0.0) & (probs < 1.0))


def test_flic_calibration_in_the_large():
    rng = np.random.default_rng(7)
    n = 80
    X = rng.normal(size=(n, 3))
    # Imbalanced outcome (~20% events)
    logits = -1.5 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    flic_model = FirthLogisticRegression(flic=True).fit(X, y)
    mean_pred = flic_model.predict_proba(X)[:, 1].mean()

    assert np.isclose(mean_pred, y.mean(), atol=1e-6)

    firth_only = FirthLogisticRegression(flic=False).fit(X, y)
    assert firth_only.converged_ is True
    assert np.allclose(flic_model.coef_, firth_only.coef_, atol=1e-6)


def test_firth_summary_table_and_dataframe_input():
    rng = np.random.default_rng(99)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, size=n),
            "income": rng.normal(50, 15, size=n),
        }
    )
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.05 * (df["age"] - 40)))))

    model = FirthLogisticRegression(flic=True).fit(df, y)
    summary = model.summary_table()

    expected_cols = {
        "term",
        "coef",
        "se",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
        "z_stat",
        "p_value",
    }
    assert expected_cols.issubset(summary.columns)
    assert list(summary["term"]) == ["(Intercept)", "age", "income"]
    assert len(model.predict(df)) == n


def test_firth_sample_weight_support():
    rng = np.random.default_rng(21)
    n = 75
    X = rng.normal(size=(n, 2))
    logits = -0.7 + 0.8 * X[:, 0] - 0.4 * X[:, 1]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
    weights = rng.uniform(0.5, 2.5, size=n)

    model = FirthLogisticRegression(flic=True).fit(X, y, sample_weight=weights)
    probs = model.predict_proba(X)[:, 1]

    weighted_mean_pred = np.average(probs, weights=weights)
    weighted_obs_rate = np.average(y, weights=weights)

    assert model.converged_ is True
    assert np.isclose(weighted_mean_pred, weighted_obs_rate, atol=1e-6)

    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(X, y, sample_weight=np.ones(n - 1))

    with pytest.raises(ValueError, match="non-negative"):
        w_neg = weights.copy()
        w_neg[0] = -0.1
        model.fit(X, y, sample_weight=w_neg)


def test_firth_validation_errors():
    model = FirthLogisticRegression()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.array([[1.0]]))

    X_nan = np.array([[1.0], [np.nan], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X_nan, y)

    X_ok = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="2 classes"):
        model.fit(X_ok, np.array([0, 1, 2]))
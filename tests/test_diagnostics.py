import numpy as np
import pandas as pd
import pytest

from smallsurv import DiagnosticReport, audit_data, riley_min_sample_size


def test_riley_min_sample_size_basic():
    min_n, breakdown = riley_min_sample_size(n_parameters=10, event_rate=0.30)
    assert min_n > 0
    assert min_n == max(breakdown.values())
    assert set(breakdown.keys()) == {
        "criterion_1_shrinkage",
        "criterion_2_r2_optimism",
        "criterion_3_precision",
    }


def test_riley_min_sample_size_invalid_inputs():
    with pytest.raises(ValueError):
        riley_min_sample_size(n_parameters=0, event_rate=0.30)
    with pytest.raises(ValueError):
        riley_min_sample_size(n_parameters=5, event_rate=1.0)


def test_audit_data_flags_small_sample_risks():
    rng = np.random.default_rng(42)
    n = 40
    y = np.array([1] * 10 + [0] * 30)
    x1 = rng.normal(size=n)
    x2 = x1 * 0.99 + rng.normal(scale=0.01, size=n)  # high collinearity
    const_col = np.ones(n)  # zero variance

    # Rare category + quasi-complete separation (all 'Rare_Sep' have y=1)
    cat = ["Common_A"] * 17 + ["Common_B"] * 18 + ["Rare_Sep"] * 5
    # Place 'Rare_Sep' on indices 0..4 where y==1
    cat = ["Rare_Sep"] * 5 + ["Common_A"] * 17 + ["Common_B"] * 18

    df = pd.DataFrame(
        {
            "outcome": y,
            "x1": x1,
            "x2": x2,
            "const_col": const_col,
            "group": cat,
        }
    )
    df.loc[0:4, "x1"] = np.nan  # >5% missingness

    report = audit_data(df, target="outcome", min_category_count=10)

    assert isinstance(report, DiagnosticReport)
    assert report.n_obs == 40
    assert report.n_events == 10
    assert report.risk_level == "HIGH"
    assert "const_col" in report.zero_variance_cols
    assert "group" in report.rare_categories
    assert "Rare_Sep" in report.rare_categories["group"]
    assert "group" in report.separation_flags
    assert "Rare_Sep" in report.separation_flags["group"]
    assert len(report.high_collinearity) == 1
    assert "x1" in report.missing_summary
    assert "smallsurv Diagnostic Audit Report" in report.summary()
    assert report.to_dict()["risk_level"] == "HIGH"


def test_audit_data_low_risk_clean_dataset():
    rng = np.random.default_rng(123)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    prob = 1 / (1 + np.exp(-(0.5 * x1 - 0.5 * x2)))
    y = rng.binomial(1, prob)
    group = rng.choice(["A", "B", "C"], size=n)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})
    report = audit_data(df, target="y")

    assert report.risk_level == "LOW"
    assert report.epv >= 15.0
    assert not report.separation_flags
    assert not report.rare_categories


def test_audit_data_validation_errors():
    df = pd.DataFrame({"y": [0, 1, 2], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="binary"):
        audit_data(df, target="y")
    with pytest.raises(ValueError, match="not found"):
        audit_data(df, target="missing_col")
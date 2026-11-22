"""Agreement of FirthLogisticRegression with independent reference implementations.

Reference values in ``tests/data/reference_firthlogist.json`` were produced with the
``firthlogist`` package (Python 3.10 environment, penalized likelihood fit) on four
datasets: the public ``sex2`` data, a simulated small sample, a rare-event sample and a
completely separated sample. They were generated outside the test run so the test suite
does not depend on ``firthlogist``.

Further checks use only SciPy and statsmodels, which are independent of the code under
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, special

from smallsurv import FirthLogisticRegression

DATA = Path(__file__).parent / "data" / "reference_firthlogist.json"
REFERENCE = json.loads(DATA.read_text())


def _penalized_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Firth penalized log-likelihood, written independently of the package."""
    design = np.column_stack([np.ones(len(y)), X])
    eta = design @ beta
    ll = np.sum(y * eta - np.logaddexp(0.0, eta))
    p = special.expit(eta)
    info = design.T @ (design * (p * (1.0 - p))[:, None])
    return float(ll + 0.5 * np.linalg.slogdet(info)[1])


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_firth_matches_firthlogist(name: str) -> None:
    d = REFERENCE[name]
    X, y = np.array(d["X"]), np.array(d["y"])

    model = FirthLogisticRegression(flic=False).fit(X, y)
    summary = model.summary_table()

    assert model.converged_
    np.testing.assert_allclose(model.coef_.ravel(), d["coef"], atol=1e-4)
    np.testing.assert_allclose(model.intercept_[0], d["intercept"], atol=1e-4)
    np.testing.assert_allclose(
        summary["se"].to_numpy()[1:], np.array(d["bse"])[: len(d["coef"])], rtol=1e-3
    )


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_firth_solution_is_penalized_likelihood_maximum(name: str) -> None:
    d = REFERENCE[name]
    X, y = np.array(d["X"]), np.array(d["y"])

    model = FirthLogisticRegression(flic=False).fit(X, y)
    beta_hat = np.concatenate([model.intercept_, model.coef_.ravel()])
    value_hat = _penalized_loglik(beta_hat, X, y)

    # A generic optimizer started from several perturbed points must not beat the fit.
    rng = np.random.default_rng(0)
    for _ in range(5):
        start = beta_hat + rng.normal(scale=0.5, size=beta_hat.shape)
        res = optimize.minimize(
            lambda b: -_penalized_loglik(b, X, y), start, method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000, "maxfev": 20000},
        )
        assert -res.fun <= value_hat + 1e-7


def test_flic_intercept_matches_offset_glm() -> None:
    sm = pytest.importorskip("statsmodels.api")
    d = REFERENCE["sim_n80"]
    X, y = np.array(d["X"]), np.array(d["y"])

    flic = FirthLogisticRegression(flic=True).fit(X, y)
    firth = FirthLogisticRegression(flic=False).fit(X, y)

    np.testing.assert_allclose(flic.coef_, firth.coef_, atol=1e-8)

    # FLIC = re-estimate the intercept by ML with the Firth linear predictor as offset.
    lp = X @ firth.coef_.ravel()
    glm = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    np.testing.assert_allclose(flic.intercept_[0], glm.params[0], atol=1e-6)


def test_integer_weights_equal_row_replication() -> None:
    d = REFERENCE["sim_n80"]
    X, y = np.array(d["X"])[:40], np.array(d["y"])[:40]
    w = np.random.default_rng(3).integers(1, 4, size=len(y))

    weighted = FirthLogisticRegression(flic=False).fit(X, y, sample_weight=w)
    replicated = FirthLogisticRegression(flic=False).fit(np.repeat(X, w, axis=0), np.repeat(y, w))

    np.testing.assert_allclose(weighted.coef_, replicated.coef_, atol=1e-5)
    np.testing.assert_allclose(weighted.intercept_, replicated.intercept_, atol=1e-5)


def test_flic_standard_errors_use_corrected_intercept() -> None:
    d = REFERENCE["sim_n80"]
    X, y = np.array(d["X"]), np.array(d["y"])

    firth = FirthLogisticRegression(flic=False).fit(X, y)
    flic = FirthLogisticRegression(flic=True).fit(X, y)

    design = np.column_stack([np.ones(len(y)), X])
    beta = np.concatenate([flic.intercept_, flic.coef_.ravel()])
    p = special.expit(design @ beta)
    info = design.T @ (design * (p * (1.0 - p))[:, None])
    expected_se = np.sqrt(np.diag(np.linalg.inv(info)))

    np.testing.assert_allclose(flic.summary_table()["se"].to_numpy(), expected_se, rtol=1e-6)
    # The intercept moved, so its standard error must no longer equal the Firth one.
    assert not np.isclose(
        flic.summary_table()["se"].iloc[0], firth.summary_table()["se"].iloc[0], rtol=1e-4
    )


def test_single_observation_category_converges_without_warning() -> None:
    """A dummy with one observation has leverage 1 and made Fisher scoring cycle."""
    import warnings

    d = REFERENCE["single_obs_category"]
    X, y = np.array(d["X"]), np.array(d["y"])
    assert (X != 0).sum(axis=0).min() == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = FirthLogisticRegression(flic=True).fit(X, y)

    assert model.converged_
    assert model.n_iter_ < 200

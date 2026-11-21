import numpy as np
import pytest

from smallsurv import SimulationResult, run_monte_carlo


def test_run_monte_carlo_compares_estimators():
    result = run_monte_carlo(
        n_obs=50,
        n_sim=20,
        true_intercept=-1.2,
        true_coef=[0.9, -0.7],
        random_state=42,
    )

    assert isinstance(result, SimulationResult)
    assert result.n_obs == 50
    assert result.n_sim == 20
    assert set(result.summary_df["estimator"]) == {"mle", "firth", "flic"}
    assert "Monte Carlo Simulation" in result.summary()
    assert "estimators" in result.to_dict()


def test_run_monte_carlo_validation_errors():
    with pytest.raises(ValueError, match="n_obs"):
        run_monte_carlo(n_obs=5)

    with pytest.raises(ValueError, match="n_sim"):
        run_monte_carlo(n_sim=2)

    with pytest.raises(ValueError, match="true_coef"):
        run_monte_carlo(true_coef=[])
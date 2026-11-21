# smallsurv

Statistical guardrails, Firth penalized logistic regression with FLIC intercept correction, calibration diagnostics, internal validation, and Monte Carlo simulation utilities for small-sample binary survey data.

## Core Modules

- **`diagnostics`**: Pre-modeling sample audit (`audit_data`), Events Per Variable (EPV), separation and rare-category detection, and minimum sample size calculation (`riley_min_sample_size`) following Riley et al. (2020).
- **`firth`**: `FirthLogisticRegression` implementing bias-reduced logistic regression with FLIC intercept correction (Puhr et al., 2017), survey/observation weight (`sample_weight`) support, and Odds Ratio summary tables.
- **`calibration`**: Calibration-in-the-Large (CITL), calibration slope, Brier score / Scaled Brier (IPA), Expected Calibration Error (ECE), and reliability curve data (`evaluate_calibration`, `reliability_curve_data`).
- **`validation`**: Repeated Stratified K-Fold cross-validation (`repeated_cv_validate`) and Harrell's optimism-corrected bootstrap validation (`bootstrap_optimism_validate`).
- **`simulation`**: Monte Carlo simulation (`run_monte_carlo`, `SimulationResult`) comparing MLE, Firth, and FLIC estimators on small-sample binary datasets.

## Installation

Install `v0.2.0` directly from GitHub:

```bash
pip install git+https://github.com/aazabir/smallsurv.git@v0.2.0
```

For local development:

```bash
git clone https://github.com/aazabir/smallsurv.git
cd smallsurv
pip install -e ".[dev]"
```

## Quick Usage Example

```python
import numpy as np
import pandas as pd

from smallsurv import (
    FirthLogisticRegression,
    audit_data,
    bootstrap_optimism_validate,
    evaluate_calibration,
)

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "age": rng.normal(38, 10, size=60),
    "income_k": rng.normal(45, 15, size=60),
    "adopted": rng.binomial(1, 0.25, size=60),
})

# 1. Pre-modeling diagnostic audit
report = audit_data(df, target="adopted")
print(report.summary())

# 2. Fit Firth penalized logistic regression with FLIC
X = df[["age", "income_k"]]
y = df["adopted"]
model = FirthLogisticRegression(flic=True).fit(X, y)
print(model.summary_table())

# 3. Calibration & optimism-corrected validation
probs = model.predict_proba(X)[:, 1]
print(evaluate_calibration(y, probs).summary())

val = bootstrap_optimism_validate(model, X, y, n_bootstraps=50, random_state=42)
print(val.summary())
```

## Running Tests & Demo

```bash
pytest
python examples/demo.py
```

## License

MIT License. See `LICENSE` for details.

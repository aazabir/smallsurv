# smallsurv

Statistical guardrails, Firth penalized logistic regression with FLIC intercept correction, calibration diagnostics, and internal validation for small-sample binary survey data.

## Core Modules

- **`diagnostics`**: Pre-modeling sample audit (`audit_data`), Events Per Variable (EPV), separation & rare-category detection, and sample size calculation (`riley_min_sample_size`) following Riley et al. (2020).
- **`firth`**: `FirthLogisticRegression` implementing bias-reduced logistic regression with FLIC intercept correction (Puhr et al., 2017) and Odds Ratio summary tables.
- **`calibration`**: Calibration-in-the-Large (CITL), calibration slope, Brier score / Scaled Brier (IPA), Expected Calibration Error (ECE), and reliability tables.
- **`validation`**: Repeated Stratified K-Fold cross-validation (`repeated_cv_validate`) and Harrell's optimism-corrected bootstrap validation (`bootstrap_optimism_validate`).

## Quick Example

```bash
python examples/demo.py
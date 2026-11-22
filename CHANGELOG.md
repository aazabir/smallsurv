# Changelog

All notable changes to this project are recorded here. The format follows
Keep a Changelog and the project uses semantic versioning.

## [Unreleased]

### Fixed
- False `ConvergenceWarning` for fits that had in fact converged. With a category that
  has a single observation, the leverage is exactly 1 and Fisher scoring cycles between
  two nearby points forever, so the score tolerance was never met. The fit now stops
  when the estimates stop moving, and if it stalls close to a solution it is polished
  by solving the modified score equations directly. The estimates were already correct
  and match `firthlogist`.
- Fitting is done on standardized predictors and mapped back to the original scale.
  The penalized maximum is identical, and convergence no longer depends on whether a
  predictor is measured in years, thousands or units.

### Added
- Regression test with a one-observation category taken from the demo bootstrap.

## [0.2.1] - 2026-11-22

### Fixed
- FLIC standard errors. With `flic=True` the intercept is re-estimated, but the
  covariance matrix came from the uncorrected Firth fit, so the intercept standard
  error never changed. It is now the inverse weighted Fisher information at the
  reported parameters, which matches the `var` matrix of `flic()` in `logistf`.
- Convergence reporting in `FirthLogisticRegression`. A failed line search could be
  reported as convergence. Fitting now stops when no step improves the penalized
  likelihood and `converged_` is `False` in that case.
- Package version strings now match the release. The `v0.2.0` tag still carried
  `0.1.0` in `pyproject.toml` and `smallsurv.__version__`.

### Added
- `ConvergenceWarning` when Firth fitting does not converge.
- Reference agreement tests against `firthlogist` fits on four datasets, an
  independent penalized likelihood maximiser, an offset GLM for the FLIC intercept,
  and row replication for integer `sample_weight`.
- Inference notes in the README and class docstring. Intervals and p-values are Wald
  based, `sample_weight` is a frequency weight, and standard errors ignore clustering
  and stratification.
- Continuous integration on Python 3.10 to 3.13 with a ruff lint step.

### Changed
- Runtime dependencies are now NumPy, SciPy, pandas and scikit-learn. `statsmodels`
  moved to the `dev` extra and `matplotlib` was removed because neither is imported
  by the package.

Earlier releases (`v0.1.0` and `v0.2.0`) were not documented in this file.

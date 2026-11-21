"""smallsurv: Statistical guardrails for small-sample binary survey data."""

from smallsurv.calibration import (
    CalibrationReport,
    evaluate_calibration,
    reliability_curve_data,
)
from smallsurv.diagnostics import DiagnosticReport, audit_data, riley_min_sample_size
from smallsurv.firth import FirthLogisticRegression
from smallsurv.simulation import SimulationResult, run_monte_carlo
from smallsurv.validation import (
    ValidationReport,
    bootstrap_optimism_validate,
    repeated_cv_validate,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "CalibrationReport",
    "DiagnosticReport",
    "FirthLogisticRegression",
    "SimulationResult",
    "ValidationReport",
    "audit_data",
    "bootstrap_optimism_validate",
    "evaluate_calibration",
    "reliability_curve_data",
    "repeated_cv_validate",
    "riley_min_sample_size",
    "run_monte_carlo",
]
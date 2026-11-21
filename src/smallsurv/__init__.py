"""smallsurv: Statistical guardrails for small-sample binary survey data."""

from smallsurv.calibration import (
    CalibrationReport,
    evaluate_calibration,
    reliability_curve_data,
)
from smallsurv.diagnostics import DiagnosticReport, audit_data, riley_min_sample_size
from smallsurv.firth import FirthLogisticRegression

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "CalibrationReport",
    "DiagnosticReport",
    "FirthLogisticRegression",
    "audit_data",
    "evaluate_calibration",
    "reliability_curve_data",
    "riley_min_sample_size",
]
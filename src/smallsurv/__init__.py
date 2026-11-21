"""smallsurv: Statistical guardrails for small-sample binary survey data."""

from smallsurv.diagnostics import DiagnosticReport, audit_data, riley_min_sample_size

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "DiagnosticReport",
    "audit_data",
    "riley_min_sample_size",
]
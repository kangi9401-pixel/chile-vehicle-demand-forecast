"""Pre-audit 3-month workflow, kept only so earlier results can be audited.

Nothing here is used by the default leakage-safe pipeline (`chile_forecast.pipeline.run`).
Its feature engineering z-scores over the full history and its backtests read the
evaluation window's actual macro values, so its metrics are not comparable to the
current results; see docs/AUDIT_KO.md.
"""

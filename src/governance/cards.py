"""
Automated Model Card and Data Card generation (transparency / auditability).

render_* are pure (dict -> markdown) for testability; build_data_card derives a
card from the live aggregates. Training logs the model card as an MLflow
artifact on every run, so every registered version ships its own card.
"""
from datetime import datetime, timezone


def render_model_card(meta: dict) -> str:
    feats = "\n".join(f"  - {c}" for c in meta.get("features", []))
    return f"""# Model Card — {meta.get('name', 'NYC Taxi Demand Forecaster')} ({meta.get('granularity', '?')})

_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_

## Model details
- Version: {meta.get('version', 'n/a')}
- Type: leakage-free next-period LightGBM forecaster
- Git commit: {meta.get('git_commit', 'unknown')}
- Artifact SHA-256: {meta.get('sha256', 'n/a')}

## Intended use
Forecast next-period NYC taxi trip volume to inform fleet capacity planning. Daily =
capacity planning; hourly = intraday steering. NOT for individual-level or
customer-facing decisions.

## Features (leakage-free; use only data before the target period)
{feats or '  - n/a'}

## Metrics (out-of-sample holdout)
- Holdout MAE: {meta.get('holdout_mae', 'n/a')}
- 99% VaR sigma (GARCH): {meta.get('garch_sigma', 'n/a')}

## Caveats & limitations
- Cannot anticipate unprecedented events (outages, viral spikes). The 99% VaR
  buffer sizes for tail risk; it does not predict the spike itself.
- Trained on historical volume only — no exogenous event features yet.

## Ethical considerations
- Operates on anonymous aggregate counts only; PII is removed at ingest.
- Outputs inform capacity planning, not individual decisions.
"""


def render_data_card(meta: dict) -> str:
    return f"""# Data Card — {meta.get('name', 'NYC Taxi Demand Aggregates')}

_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_

## Source
{meta.get('source', 'NYC taxi trip records (TLC open data)')}

## Granularity & coverage
- Granularity: {meta.get('granularity', '?')}
- Range: {meta.get('start', '?')} .. {meta.get('end', '?')}
- Rows: {meta.get('rows', '?')}

## Schema
- TimePeriod — datetime (UTC)
- Volume — float, 0 <= x <= 1e6 (enforced by the validation gate)

## Preprocessing
Timestamps coerced to UTC, invalid rows dropped, deduplicated, resampled to
per-period counts.

## PII handling
Raw inputs scanned (email / SSN / credit-card / phone); detected columns dropped
at ingest. Published aggregates are anonymous counts, asserted PII-free.

## Versioning
DVC-tracked (content md5); remote: {meta.get('dvc_remote', 'n/a')}
"""


def build_data_card(granularity: str, df, dvc_remote: str = None) -> str:
    df = df.sort_values("TimePeriod")
    return render_data_card({
        "name": f"taxi_demand_{'daily' if granularity.upper() == 'D' else 'hourly'}",
        "granularity": "Daily" if granularity.upper() == "D" else "Hourly",
        "start": str(df["TimePeriod"].min()),
        "end": str(df["TimePeriod"].max()),
        "rows": int(len(df)),
        "dvc_remote": dvc_remote,
    })

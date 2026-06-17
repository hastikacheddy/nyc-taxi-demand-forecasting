import os
from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field, ValueType
from feast.types import Float64

# Resolve feature parquet paths relative to the project root so the repo works
# in any environment (local, CI, Docker, K8s) — not just one machine.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_FEATURES_DIR = os.path.join(_PROJECT_ROOT, 'data', 'features')

# --- Entity ---
time_period = Entity(
    name="time_period",
    value_type=ValueType.STRING,
    description="Unique time-bucket identifier for a taxi trip volume record (e.g., '2025-01-01' or '2025-01-01T09')",
)

# --- Data Sources ---
daily_features_source = FileSource(
    path=os.path.join(_FEATURES_DIR, 'daily_features.parquet'),
    timestamp_field="TimePeriod",
)

hourly_features_source = FileSource(
    path=os.path.join(_FEATURES_DIR, 'hourly_features.parquet'),
    timestamp_field="TimePeriod",
)

# --- Daily Feature View ---
daily_demand_features = FeatureView(
    name="daily_demand_features",
    entities=[time_period],
    ttl=timedelta(days=14),
    schema=[
        Field(name="Volume_Lag1",      dtype=Float64),
        Field(name="Volume_LagDelta1", dtype=Float64),
        Field(name="Volume_Lag7",      dtype=Float64),
        Field(name="Volume_LagDelta7", dtype=Float64),
        Field(name="Sin_DayOfWeek",       dtype=Float64),
        Field(name="Cos_DayOfWeek",       dtype=Float64),
        Field(name="Sin_WeekOfYear",      dtype=Float64),
        Field(name="Cos_WeekOfYear",      dtype=Float64),
        Field(name="Regime_0",            dtype=Float64),
        Field(name="Regime_1",            dtype=Float64),
        Field(name="Regime_2",            dtype=Float64),
    ],
    source=daily_features_source,
)

# --- Hourly Feature View ---
hourly_demand_features = FeatureView(
    name="hourly_demand_features",
    entities=[time_period],
    ttl=timedelta(hours=48),
    schema=[
        Field(name="Volume_Lag1",       dtype=Float64),
        Field(name="Volume_LagDelta1",  dtype=Float64),
        Field(name="Volume_Lag6",       dtype=Float64),
        Field(name="Volume_LagDelta6",  dtype=Float64),
        Field(name="Volume_Lag12",      dtype=Float64),
        Field(name="Volume_LagDelta12", dtype=Float64),
        Field(name="Volume_Lag24",      dtype=Float64),
        Field(name="Volume_LagDelta24", dtype=Float64),
        Field(name="Sin_Hour",             dtype=Float64),
        Field(name="Cos_Hour",             dtype=Float64),
        Field(name="Regime_0",             dtype=Float64),
        Field(name="Regime_1",             dtype=Float64),
        Field(name="Regime_2",             dtype=Float64),
    ],
    source=hourly_features_source,
)

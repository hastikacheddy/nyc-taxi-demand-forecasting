"""
Adversarial / red-team tests — evasion and data-poisoning payloads must be
rejected by the validation gate and the API's input validators, not silently
processed.
"""
import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pydantic import ValidationError

from src.inference.validation import validate_input_data
from src.serving.api import ForecastRequest


# ── Data poisoning: the circuit breaker must reject malformed counts ──
def _frame(counts):
    return pd.DataFrame({
        "TimePeriod": pd.date_range("2026-01-01", periods=len(counts), freq="D"),
        "Volume": counts,
    })


@pytest.mark.parametrize("counts", [
    [10.0, -5.0, 20.0],          # negative volume
    [10.0, np.nan, 20.0],        # NaN injection
    [10.0, np.inf, 20.0],        # inf injection
    [10.0, 5_000_000.0, 20.0],   # absurd spike (above the 1e6 bound)
])
def test_poisoned_counts_are_rejected(counts):
    with pytest.raises(pa.errors.SchemaError):
        validate_input_data(_frame(counts))


def test_clean_data_passes():
    validate_input_data(_frame([10.0, 20.0, 30.0]))   # must not raise


# ── Evasion: API input validators must block injection / out-of-range ──
@pytest.mark.parametrize("bad_period", [
    "'; DROP TABLE trips;--",
    "<script>alert(1)</script>",
    "../../etc/passwd",
    "A" * 50,                     # over-length
])
def test_time_period_injection_blocked(bad_period):
    with pytest.raises(ValidationError):
        ForecastRequest(time_period=bad_period, granularity="D")


@pytest.mark.parametrize("bad_conf", [1.5, -0.1, 2.0])
def test_confidence_out_of_range_blocked(bad_conf):
    with pytest.raises(ValidationError):
        ForecastRequest(time_period="2026-03-31", granularity="D", confidence_level=bad_conf)


def test_unknown_granularity_blocked():
    with pytest.raises(ValidationError):
        ForecastRequest(time_period="2026-03-31", granularity="DROP")

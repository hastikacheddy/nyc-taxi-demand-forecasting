"""
Shared fixtures for all test modules.
"""
import os

# Set before any test module imports src.serving.api (its API key is read at
# import time). Must live in conftest so it runs before collection.
os.environ.setdefault("API_KEY", "testsecret")

import pytest          # noqa: E402
import pandas as pd    # noqa: E402
import numpy as np     # noqa: E402


@pytest.fixture
def daily_raw_df():
    """Minimal daily trip DataFrame spanning 60 days — enough for all lag windows."""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    np.random.seed(42)
    counts = np.random.randint(50, 300, size=60).astype(float)
    return pd.DataFrame({'TimePeriod': dates, 'Volume': counts})


@pytest.fixture
def hourly_raw_df():
    """Minimal hourly trip DataFrame spanning 10 days — enough for 24-period lags."""
    dates = pd.date_range(start='2024-01-01', periods=240, freq='h')
    np.random.seed(42)
    counts = np.random.randint(0, 50, size=240).astype(float)
    return pd.DataFrame({'TimePeriod': dates, 'Volume': counts})


@pytest.fixture
def raw_trip_log_df():
    """Simulated raw NYC taxi trip record with CreatedStamp column."""
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    # Introduce duplicates and a few NaT values
    ts_list = list(timestamps) + list(timestamps[:10]) + [pd.NaT] * 5
    np.random.shuffle(ts_list)
    return pd.DataFrame({'CreatedStamp': ts_list})

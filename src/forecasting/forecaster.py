"""
Leakage-free next-period forecaster.

The notebook-faithful QuantFeatureEngineer uses LagDelta = Volume - Lag_n,
which contains the *current* period's value — fine for the notebook's in-sample
analysis, but it means the production model can only "predict" a period whose
actual is already known (a nowcast). For a real forward forecast we must build
every feature from data strictly BEFORE the target period.

This module is additive: it does not touch QuantFeatureEngineer. It produces a
genuine 1-step-ahead forecaster whose features for predicting period t use only
Volume up to t-1.
"""
import numpy as np
import pandas as pd

# Past-only feature contract. For target period t (predicted before it happens):
#   Lag1        = Volume[t-1]                  last known value
#   LagS        = Volume[t-season]             same period one cycle ago
#   PrevDelta1  = Volume[t-1] - Volume[t-2] recent momentum (past)
#   PrevDeltaS  = Volume[t-1] - Volume[t-1-season]  cycle-over-cycle momentum
#   RollMean    = mean(Volume[t-season .. t-1])
#   RollStd     = std (Volume[t-season .. t-1])
#   calendar for t (known in advance): sin/cos day-of-week, week-of-year, +hour
_PARAMS = {
    'D': {'season': 7, 'freq': 'D'},
    'H': {'season': 24, 'freq': 'h'},
}


def feature_columns(granularity: str) -> list:
    cols = ['Lag1', 'LagS', 'PrevDelta1', 'PrevDeltaS', 'RollMean', 'RollStd',
            'Sin_DayOfWeek', 'Cos_DayOfWeek', 'Sin_WeekOfYear', 'Cos_WeekOfYear']
    if granularity.upper() == 'H':
        cols += ['Sin_Hour', 'Cos_Hour']
    return cols


def _calendar(index, granularity):
    cal = pd.DataFrame(index=index)
    dow = index.dayofweek
    cal['Sin_DayOfWeek'] = np.sin(2 * np.pi * dow / 7)
    cal['Cos_DayOfWeek'] = np.cos(2 * np.pi * dow / 7)
    woy = index.isocalendar().week.astype(float).values
    cal['Sin_WeekOfYear'] = np.sin(2 * np.pi * woy / 52)
    cal['Cos_WeekOfYear'] = np.cos(2 * np.pi * woy / 52)
    if granularity.upper() == 'H':
        hour = index.hour
        cal['Sin_Hour'] = np.sin(2 * np.pi * hour / 24)
        cal['Cos_Hour'] = np.cos(2 * np.pi * hour / 24)
    return cal


def _past_only_features(s: pd.Series, granularity: str) -> pd.DataFrame:
    """Build the past-only lag/rolling block aligned to each target period t."""
    season = _PARAMS[granularity.upper()]['season']
    feat = pd.DataFrame(index=s.index)
    feat['Lag1'] = s.shift(1)
    feat['LagS'] = s.shift(season)
    feat['PrevDelta1'] = s.shift(1) - s.shift(2)
    feat['PrevDeltaS'] = s.shift(1) - s.shift(1 + season)
    feat['RollMean'] = s.shift(1).rolling(season).mean()
    feat['RollStd'] = s.shift(1).rolling(season).std()
    return feat


def build_training_frame(df: pd.DataFrame, granularity: str):
    """Return (X, y) for supervised training. Row t's features use only data < t."""
    gran = granularity.upper()
    s = (df.set_index('TimePeriod')['Volume'].astype(float).sort_index()
         if 'TimePeriod' in df.columns else df['Volume'].astype(float).sort_index())
    feat = _past_only_features(s, gran).join(_calendar(s.index, gran))
    feat['y'] = s.values
    feat = feat.dropna()
    return feat[feature_columns(gran)], feat['y']


def build_next_step_features(df: pd.DataFrame, granularity: str):
    """Build the single feature row for the period AFTER the last timestamp.
    Returns (X_row, next_timestamp). Uses only observed history — no leakage."""
    gran = granularity.upper()
    freq = _PARAMS[gran]['freq']
    s = df.set_index('TimePeriod')['Volume'].astype(float).sort_index()

    next_ts = s.index[-1] + pd.tseries.frequencies.to_offset(freq)
    # Append an empty target slot so shifts line up for the future period.
    s_ext = pd.concat([s, pd.Series([np.nan], index=[next_ts])])
    feat = _past_only_features(s_ext, gran).join(_calendar(s_ext.index, gran))
    X_row = feat[feature_columns(gran)].tail(1)
    return X_row, next_ts

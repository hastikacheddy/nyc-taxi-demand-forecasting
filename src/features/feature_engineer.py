import pandas as pd
import numpy as np

class QuantFeatureEngineer:
    def __init__(self, dataframe: pd.DataFrame, granularity: str = 'D'):
        self.df = dataframe.copy()
        self.granularity = granularity.upper()

        if 'TimePeriod' not in self.df.columns or 'Volume' not in self.df.columns:
            raise ValueError("DataFrame must contain 'TimePeriod' and 'Volume' columns.")

        self.df['TimePeriod'] = pd.to_datetime(self.df['TimePeriod'])
        self.df = self.df.set_index('TimePeriod').sort_index()

        self._set_parameters()

    def _set_parameters(self):
        if self.granularity == 'H':
            self.lag_periods = [1, 6, 12, 24]
            self.rolling_periods = [6, 12, 24]
            self.ema_spans = [12, 24, 48]
        elif self.granularity == 'D':
            self.lag_periods = [1, 7, 14, 28]
            self.rolling_periods = [7, 14, 28]
            self.ema_spans = [7, 14, 28]
        elif self.granularity == 'W':
            self.lag_periods = [1, 4, 12, 52]
            self.rolling_periods = [4, 12, 26]
            self.ema_spans = [4, 12, 26]
        else:
            raise ValueError("Granularity must be 'H', 'D', or 'W'.")

    def add_lag_features(self):
        for lag in self.lag_periods:
            self.df[f'Volume_Lag{lag}'] = self.df['Volume'].shift(lag)
            # LagDelta = current - lagged (momentum), matching notebook Cell 155
            self.df[f'Volume_LagDelta{lag}'] = self.df['Volume'] - self.df[f'Volume_Lag{lag}']

    def add_rolling_features(self):
        for window in self.rolling_periods:
            self.df[f'Volume_RollingMean{window}'] = self.df['Volume'].rolling(window=window).mean()
            self.df[f'Volume_RollingStd{window}'] = self.df['Volume'].rolling(window=window).std()

    def add_ema_features(self):
        for span in self.ema_spans:
            self.df[f'Volume_EMA{span}'] = self.df['Volume'].ewm(span=span).mean()

    def add_cyclical_features(self):
        if self.granularity == 'H':
            self.df['Hour'] = self.df.index.hour
            self.df['Sin_Hour'] = np.sin(2 * np.pi * self.df['Hour'] / 24)
            self.df['Cos_Hour'] = np.cos(2 * np.pi * self.df['Hour'] / 24)

        self.df['DayOfWeek'] = self.df.index.dayofweek
        self.df['Sin_DayOfWeek'] = np.sin(2 * np.pi * self.df['DayOfWeek'] / 7)
        self.df['Cos_DayOfWeek'] = np.cos(2 * np.pi * self.df['DayOfWeek'] / 7)

        self.df['WeekOfYear'] = self.df.index.isocalendar().week.astype(float)
        self.df['Sin_WeekOfYear'] = np.sin(2 * np.pi * self.df['WeekOfYear'] / 52)
        self.df['Cos_WeekOfYear'] = np.cos(2 * np.pi * self.df['WeekOfYear'] / 52)

    def add_regime_features(self):
        if 'Regime_0' not in self.df.columns:
            self.df['Regime_0'] = 1.0
            self.df['Regime_1'] = 0.0
            self.df['Regime_2'] = 0.0

    def engineer_features(self) -> pd.DataFrame:
        self.add_lag_features()
        self.add_rolling_features()
        self.add_ema_features()
        self.add_cyclical_features()
        self.add_regime_features()
        self.df.dropna(inplace=True)
        return self.df

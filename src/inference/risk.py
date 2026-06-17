"""
Risk-buffer utilities — faithful production port of the notebook's
Value-at-Risk analysis (cells 96-112):

  1. residuals  = actual - forecast               (on a chronological holdout)
  2. GARCH(1,1) fitted to the residuals           -> conditional volatility
  3. Monte Carlo: Normal(point_forecast, sigma)   -> 10,000 sims, clip negatives
  4. VaR_99     = 99th percentile of the sims

GARCH is fitted once per weekly training run (expensive, needs a residual
series). The resulting forward volatility (sigma) is stored on the model
version and consumed at inference, where only the cheap Monte Carlo runs.
"""
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

N_SIMULATIONS = 10_000
# Static fallbacks = the notebook's reported LightGBM MAE (cells 174/183),
# used only when GARCH cannot be fitted (e.g. too few residuals).
FALLBACK_SIGMA = {'D': 48.65, 'H': 0.21}


def monte_carlo_var(point_forecast: float, sigma: float,
                    confidence_level: float = 0.99,
                    n_simulations: int = N_SIMULATIONS,
                    seed: int = 42) -> float:
    """
    Monte Carlo Value-at-Risk, matching notebook cells 110-112.
    Draws Normal(point_forecast, sigma), clips negatives to 0 (taxi trip volume
    can't be negative), and returns the confidence_level quantile.
    """
    sigma = max(float(sigma), 1e-9)
    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=max(0.0, point_forecast), scale=sigma, size=n_simulations)
    sims[sims < 0] = 0
    return float(np.quantile(sims, confidence_level))


def compute_garch_sigma(residuals, fallback: float) -> float:
    """
    Fit GARCH(1,1) to a residual series and return the 1-step-ahead
    conditional volatility forecast. Falls back to the residual std (or the
    supplied fallback) if GARCH is unavailable or fails to converge.
    """
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 30 or not np.isfinite(residuals).all():
        logger.info("GARCH skipped (only %d residuals); using fallback sigma.", len(residuals))
        return fallback
    try:
        from arch import arch_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(residuals, vol='Garch', p=1, q=1, rescale=False).fit(disp='off')
            forecast = res.forecast(horizon=1, reindex=False)
        sigma = float(np.sqrt(forecast.variance.values[-1, 0]))
        if not np.isfinite(sigma) or sigma <= 0:
            return fallback
        return sigma
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("GARCH fit failed (%s); using fallback sigma.", e)
        return fallback


def estimate_conditional_sigmas(residuals, keys, fallback: float,
                                min_count: int = 20, floor: float = 0.5) -> dict:
    """
    Per-bucket residual sigma (e.g. one per hour-of-day). Buckets with too few
    samples fall back to the global sigma; all are floored to avoid a zero-width
    band. Captures heteroscedasticity the single GARCH sigma misses — wide at
    peak hours, near-zero overnight.
    """
    residuals = np.asarray(residuals, dtype=float)
    keys = np.asarray(keys)
    out = {}
    for k in np.unique(keys):
        bucket = residuals[keys == k]
        # Enough samples -> use the empirical std (floored). The fallback is only
        # for sparse buckets; a well-sampled, perfectly-predictable hour should
        # get the tiny floor, not the wide global sigma.
        if len(bucket) >= min_count and np.isfinite(bucket).all():
            out[int(k)] = max(float(np.std(bucket)), floor)
        else:
            out[int(k)] = fallback
    return out


def estimate_risk_sigma(df_fe, features, granularity: str,
                        target: str = 'Volume') -> float:
    """
    Reproduce the notebook's VaR sigma: chronological 80/20 split, fit a
    LightGBM on the train portion, take residuals on the holdout, then fit
    GARCH(1,1) and return its 1-step volatility forecast.
    """
    import lightgbm as lgb

    fallback = FALLBACK_SIGMA.get(granularity.upper(), 50.0)
    n = len(df_fe)
    if n < 40:
        return fallback

    split = int(n * 0.8)
    X_tr, X_te = df_fe[features].iloc[:split], df_fe[features].iloc[split:]
    y_tr, y_te = df_fe[target].iloc[:split], df_fe[target].iloc[split:]

    tmp = lgb.LGBMRegressor(random_state=42, n_estimators=100,
                            learning_rate=0.05, verbosity=-1)
    tmp.fit(X_tr, y_tr)
    residuals = y_te.values - tmp.predict(X_te)

    resid_std = float(np.std(residuals)) if len(residuals) else fallback
    return compute_garch_sigma(residuals, fallback=resid_std)

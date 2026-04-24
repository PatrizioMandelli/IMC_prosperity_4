import numpy as np
import pandas as pd


def ema_fair_value(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 30,
    col: str = "mid_price",
) -> pd.DataFrame:
    """
    Adds ema_fast, ema_slow, trend_signal (fast-slow), deviation (col - ema5),
    and deviation_z (rolling z-score of deviation).
    Generalizes pepper_behaviour_analyzer.py EMA logic to any column and window.
    """
    df = df.copy()
    mid = df[col]
    df["ema_fast"] = mid.ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = mid.ewm(span=slow, adjust=False).mean()
    df["trend_signal"] = df["ema_fast"] - df["ema_slow"]

    ema5 = mid.ewm(span= fast // 2 if fast > 4 else 2, adjust=False).mean()
    df["deviation"] = mid - ema5

    roll_mean = df["deviation"].rolling(50, min_periods=5).mean()
    roll_std = df["deviation"].rolling(50, min_periods=5).std().replace(0, np.nan)
    df["deviation_z"] = (df["deviation"] - roll_mean) / roll_std
    return df


def macd_series(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Computes MACD, Signal line, and Histogram.
    Returns DataFrame with columns: macd, macd_signal, macd_hist.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist
    }, index=series.index)


def rsi_series(series: pd.Series, window: int = 14) -> pd.Series:
    """Standard Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def ewma_series(series: pd.Series, span: int = 20) -> pd.Series:
    """Simple wrapper for exponential weighted moving average."""
    return series.ewm(span=span, adjust=False).mean()


def zscore_signal(
    series: pd.Series,
    window: int = 50,
    clip: float = 3.0,
) -> pd.Series:
    """Rolling z-score of a series, clipped to [-clip, +clip]."""
    mean = series.rolling(window, min_periods=5).mean()
    std = series.rolling(window, min_periods=5).std().replace(0, np.nan)
    z = (series - mean) / std
    return z.clip(-clip, clip)


def inventory_pressure(
    position: float | int,
    limit: int,
    mode: str = "quadratic",
) -> float:
    """
    Scalar in [-1, 1] representing how much inventory skews quotes.
    mode='linear': position/limit
    mode='quadratic': sign(position) * (position/limit)^2
    """
    ratio = position / limit if limit != 0 else 0.0
    ratio = max(-1.0, min(1.0, ratio))
    if mode == "quadratic":
        return float(np.sign(ratio) * ratio ** 2)
    return float(ratio)

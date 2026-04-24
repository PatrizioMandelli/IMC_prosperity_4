import numpy as np
import pandas as pd


def micro_price_series(df: pd.DataFrame) -> pd.Series:
    """WallMid = (bid_vol_1*ask_price_1 + ask_vol_1*bid_price_1) / (bid_vol_1+ask_vol_1)."""
    bv = df["bid_volume_1"].fillna(0)
    av = df["ask_volume_1"].fillna(0)
    total = bv + av
    result = np.where(total > 0, (bv * df["ask_price_1"] + av * df["bid_price_1"]) / total, np.nan)
    return pd.Series(result, index=df.index).ffill(limit=3)


def ofi_series(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Rolling OFI = sum(Δbid_vol - Δask_vol) / sum(total_depth) over `window` ticks."""
    bv = df["bid_volume_1"].fillna(0)
    av = df["ask_volume_1"].fillna(0)
    bid_depth = df.get("bid_depth", bv)
    ask_depth = df.get("ask_depth", av)
    total_depth = (bid_depth + ask_depth).clip(lower=1)

    delta_bid = bv.diff().fillna(0)
    delta_ask = av.diff().fillna(0)
    raw_ofi = (delta_bid - delta_ask) / total_depth

    if window <= 1:
        return raw_ofi.fillna(0)
    return raw_ofi.rolling(window, min_periods=1).sum().fillna(0)


def spread_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Adds spread_pct_rank (0-1) and spread_regime ('tight'/'normal'/'wide') columns."""
    df = df.copy()
    spread = df.get("spread", df["ask_price_1"] - df["bid_price_1"])
    df["spread_pct_rank"] = spread.rank(pct=True)
    q33 = spread.quantile(0.33)
    q67 = spread.quantile(0.67)
    df["spread_regime"] = pd.cut(
        spread,
        bins=[-np.inf, q33, q67, np.inf],
        labels=["tight", "normal", "wide"],
    )
    return df


def tick_returns(df: pd.DataFrame) -> pd.Series:
    """Per-product tick-by-tick return: mid_price.diff() / mid_price.shift(1)."""
    if "product" not in df.columns:
        mid = df["mid_price"]
        return mid.diff() / mid.shift(1)
    return df.groupby("product")["mid_price"].transform(
        lambda s: s.diff() / s.shift(1)
    )

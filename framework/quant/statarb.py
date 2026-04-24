import numpy as np
import pandas as pd
from scipy import stats


def pair_spread(
    df: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    hedge_ratio: float | None = None,
) -> pd.Series:
    """
    Computes spread = mid_a - hedge_ratio * mid_b.
    If hedge_ratio is None, estimates via OLS over the full series.
    df must contain both products (multi-product LOB DataFrame).
    """
    pivot = df.pivot_table(index="global_time", columns="product", values="mid_price")
    a = pivot[asset_a].dropna()
    b = pivot[asset_b].dropna()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    if hedge_ratio is None:
        slope, _, _, _, _ = stats.linregress(b, a)
        hedge_ratio = slope

    spread = a - hedge_ratio * b
    spread.name = f"{asset_a}_vs_{asset_b}_spread"
    return spread


def spread_zscore(spread: pd.Series, window: int = 100) -> pd.Series:
    """Rolling z-score of the spread. Entry signal at ±2, exit at 0."""
    mean = spread.rolling(window, min_periods=10).mean()
    std = spread.rolling(window, min_periods=10).std().replace(0, np.nan)
    return (spread - mean) / std


def lagged_cross_correlation(
    df: pd.DataFrame,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Computes lagged return correlations for all product pairs.
    Returns DataFrame with columns: asset_a, asset_b, lag, correlation.
    Generalizes data_explorer.analyze_lagged_correlations().
    """
    pivot = df.pivot_table(index="global_time", columns="product", values="mid_price")
    returns = pivot.pct_change().dropna()
    assets = list(returns.columns)

    records = []
    for i, a in enumerate(assets):
        for b in assets[i + 1:]:
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = returns[a].corr(returns[b])
                elif lag > 0:
                    corr = returns[a].corr(returns[b].shift(lag))
                else:
                    corr = returns[a].shift(-lag).corr(returns[b])
                records.append({"asset_a": a, "asset_b": b, "lag": lag, "correlation": corr})

    return pd.DataFrame(records)


def calculate_ou_half_life(spread: pd.Series) -> float:
    """
    Calcola l'half-life di mean-reversion di uno spread usando un processo di Ornstein-Uhlenbeck.
    Stima l'equazione: d(spread) = theta * (mu - spread) * dt + dW
    L'half-life è -ln(2) / theta.
    """
    if spread.empty:
        return -1.0
        
    # Regressione lineare per stimare il parametro di mean-reversion (theta)
    # d(spread) ~ theta * spread_{t-1}
    spread_diff = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()
    
    # Assicura che gli indici siano allineati
    common_idx = spread_diff.index.intersection(spread_lag.index)
    if len(common_idx) < 20:
        return -1.0

    spread_diff = spread_diff.loc[common_idx]
    spread_lag = spread_lag.loc[common_idx]
    
    # Aggiungi una costante per il drift (mu)
    spread_lag_const = pd.concat([spread_lag, pd.Series(1, index=spread_lag.index)], axis=1)

    try:
        # Esegui la regressione di spread_diff su spread_lag
        res = stats.linregress(spread_lag, spread_diff)
        theta = res.slope
    except ValueError:
        return -1.0

    if theta >= 0:
        return -1.0 # Non è un processo di mean-reversion
    
    # Calcola l'half-life
    half_life = -np.log(2) / theta
    return float(half_life)

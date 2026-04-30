import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / 'data' / 'round5'

BASKET_WEIGHTS = {
    'UV_VISOR_MAGENTA': 1.0,
    'UV_VISOR_ORANGE': 0.0653,
    'UV_VISOR_YELLOW': -0.0312,
    'UV_VISOR_AMBER': 0.6566,
    'UV_VISOR_RED': 0.2874,
}


def load_data(data_dir):
    """Loads and pivots price data; drops rows missing any basket product."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('prices_*.csv'))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in '{data_dir.resolve()}'")
    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    pivot = df.pivot(index=['day', 'timestamp'], columns='product', values='mid_price')
    return pivot.dropna(subset=list(BASKET_WEIGHTS))


def simulate_alpha(spread_series, alpha):
    """EMA-based z-score strategy backtest on spread.

    Position is proportional to z-score, capped at ±3σ and scaled to [-1, 1].
    Returns (final_pnl, cumulative_pnl_list).
    """
    ema = spread_series.iloc[0]
    var = 0.0
    position = 0.0
    pnl = 0.0
    pnls = []

    for i, current_spread in enumerate(spread_series):
        if i > 0:
            pnl += position * (current_spread - spread_series.iloc[i - 1])
        pnls.append(pnl)

        dev = current_spread - ema
        ema = alpha * current_spread + (1 - alpha) * ema
        var = alpha * dev ** 2 + (1 - alpha) * var
        std_dev = math.sqrt(var) if var > 0 else 1.0
        z = max(min(dev / std_dev, 3.0), -3.0)
        position = -z / 3.0

    return pnl, pnls


if __name__ == '__main__':
    df = load_data(DATA_DIR)
    df['SPREAD'] = sum(df[p] * w for p, w in BASKET_WEIGHTS.items())
    spread_series = df['SPREAD']

    alphas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
    best_alpha, best_pnl, best_curve = None, -float('inf'), []

    print(f"{'ALPHA':<10} | {'PNL':>12}")
    print('-' * 25)
    for alpha in alphas:
        pnl, curve = simulate_alpha(spread_series, alpha)
        print(f"{alpha:<10} | {pnl:>12.2f}")
        if pnl > best_pnl:
            best_pnl, best_alpha, best_curve = pnl, alpha, curve
    print(f"\nBest alpha: {best_alpha}  PnL: {best_pnl:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(best_curve, color='green', label=f'alpha={best_alpha}')
    plt.title(f'Cumulative PnL – Best Alpha: {best_alpha}')
    plt.xlabel('Tick')
    plt.ylabel('PnL')
    plt.grid(True)
    plt.legend()
    plt.show()

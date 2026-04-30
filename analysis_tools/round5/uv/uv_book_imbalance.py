import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = REPO_ROOT / 'data' / 'round5' / 'prices_round_5_day_2.csv'

BASKET_WEIGHTS = {
    'UV_VISOR_MAGENTA': 1.0,
    'UV_VISOR_ORANGE': 0.065,
    'UV_VISOR_YELLOW': -0.03,
    'UV_VISOR_AMBER': 0.65,
    'UV_VISOR_RED': 0.3,
}


def analyze_imbalance(file_path):
    """Plots basket spread vs aggregated L1 order book imbalance."""
    df = pd.read_csv(file_path, sep=';')
    df = df[df['product'].isin(BASKET_WEIGHTS)].copy()

    # Imbalance ∈ [-1, 1]: positive when bid-side dominates.
    df['imbalance'] = (
        (df['bid_volume_1'] - df['ask_volume_1'])
        / (df['bid_volume_1'] + df['ask_volume_1'])
    )

    pivot_price = df.pivot(index='timestamp', columns='product', values='mid_price').dropna()
    pivot_imb = df.pivot(index='timestamp', columns='product', values='imbalance').dropna()

    spread = sum(pivot_price[p] * w for p, w in BASKET_WEIGHTS.items())
    # Sign-weighted imbalance: short legs contribute negatively.
    basket_imbalance = sum(pivot_imb[p] * np.sign(w) for p, w in BASKET_WEIGHTS.items())

    ts = pivot_price.index[:2000]
    spread_slice = spread.iloc[:2000]
    imb_slice = basket_imbalance.iloc[:2000]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Spread', color='tab:purple')
    ax1.plot(ts, spread_slice, color='tab:purple', label='Spread')
    ax1.plot(ts, spread_slice.rolling(20).mean(), color='black', ls='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='tab:purple')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Imbalance', color='tab:cyan')
    ax2.plot(ts, imb_slice, color='tab:cyan', alpha=0.3)
    ax2.plot(ts, imb_slice.rolling(10).mean(), color='blue', lw=2, label='Imbalance (MA)')
    ax2.tick_params(axis='y', labelcolor='tab:cyan')
    ax2.axhline(0, color='gray', ls=':')

    plt.title('UV_VISOR: Spread vs Order Book Imbalance', fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    analyze_imbalance(FILE_PATH)

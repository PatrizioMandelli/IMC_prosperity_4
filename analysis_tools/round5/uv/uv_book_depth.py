import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / 'data' / 'round5'
TARGET_PRODUCT = 'UV_VISOR_MAGENTA'


def analyze_microstructure(data_dir, product):
    """Plots mid-price, bid-ask spread, and L1 liquidity for a single product."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('prices_*.csv'))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in '{data_dir.resolve()}'")

    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    df = df[df['product'] == product].copy()

    df['bid_ask_spread'] = df['ask_price_1'] - df['bid_price_1']
    df['top_level_liquidity'] = df['bid_volume_1'] + df['ask_volume_1']

    sample = df.iloc[5000:6000].copy()
    ts = sample['timestamp']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.plot(ts, sample['mid_price'], color='purple', label='Mid Price')
    ax1.set_title(f'Microstructure – {product}', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, sample['bid_ask_spread'], color='red', alpha=0.7, label='Bid-Ask Spread')
    ax2.axhline(sample['bid_ask_spread'].mean(), color='black', ls='--', label='Mean')
    ax2.set_ylabel('Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.fill_between(ts, 0, sample['top_level_liquidity'], color='blue', alpha=0.3, label='L1 Volume')
    ax3.set_ylabel('Volume')
    ax3.set_xlabel('Timestamp')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    analyze_microstructure(DATA_DIR, TARGET_PRODUCT)

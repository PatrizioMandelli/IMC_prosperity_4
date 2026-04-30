import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data' / 'round5'
OUT_DIR = REPO_ROOT / 'analysis_tools' / 'galaxy'

PRODUCTS = [
    'GALAXY_SOUNDS_DARK_MATTER',
    'GALAXY_SOUNDS_BLACK_HOLES',
    'GALAXY_SOUNDS_SOLAR_WINDS',
    'GALAXY_SOUNDS_SOLAR_FLAMES',
    'GALAXY_SOUNDS_PLANETARY_RINGS',
]

ROLLING_WINDOW = 20
STD_MULT = 2.0
# Prosperity data is sampled every ~100ms; downsampling at 40x ≈ 4s intervals.
SAMPLE_RATE = 40

files = sorted(DATA_DIR.glob('prices_*.csv'))
if not files:
    print(f"No prices_*.csv found in {DATA_DIR}")
    df = pd.DataFrame()
else:
    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    if 'day' in df.columns:
        df['continuous_timestamp'] = df['day'] * 1_000_000 + df['timestamp']
        df = df.sort_values('continuous_timestamp')
    else:
        df['continuous_timestamp'] = df['timestamp']
        df = df.sort_values('timestamp')


def calculate_hurst(ts):
    """Estimates Hurst exponent via log-log regression of variance over lags.

    H < 0.5 → mean-reverting; H ≈ 0.5 → random walk; H > 0.5 → trending.
    """
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def calculate_hurst_robust(ts, min_lag=10, max_lag=200):
    """Hurst on a coarser lag grid to suppress microstructure noise."""
    lags = np.arange(min_lag, max_lag, step=5)
    lags = lags[lags < len(ts)]
    if len(lags) < 5:
        return None
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def _regime_label(h):
    if h < 0.45:
        return 'MEAN REVERTING'
    if h > 0.55:
        return 'TRENDING'
    return 'RANDOM WALK'


if not df.empty:
    hurst_results = {}
    for product in PRODUCTS:
        df_prod = df[df['product'] == product].copy()
        if df_prod.empty:
            continue
        df_prod = df_prod.sort_values('continuous_timestamp')
        if 'mid_price' not in df_prod.columns:
            df_prod['mid_price'] = (df_prod['bid_price_1'] + df_prod['ask_price_1']) / 2

        df_prod['SMA'] = df_prod['mid_price'].rolling(ROLLING_WINDOW).mean()
        df_prod['Std_Dev'] = df_prod['mid_price'].rolling(ROLLING_WINDOW).std()
        df_prod['Upper'] = df_prod['SMA'] + df_prod['Std_Dev'] * STD_MULT
        df_prod['Lower'] = df_prod['SMA'] - df_prod['Std_Dev'] * STD_MULT

        spread_mean = (df_prod['ask_price_1'] - df_prod['bid_price_1']).mean()
        print(f"{product}: mean bid-ask spread = {spread_mean:.2f}")

        plot_df = df_prod.head(2000)
        plt.figure(figsize=(14, 6))
        plt.plot(plot_df['continuous_timestamp'], plot_df['mid_price'], label='Mid', color='blue', alpha=0.8)
        plt.plot(plot_df['continuous_timestamp'], plot_df['SMA'],
                 label=f'SMA({ROLLING_WINDOW})', color='orange', lw=2)
        plt.plot(plot_df['continuous_timestamp'], plot_df['Upper'],
                 color='red', ls='--', alpha=0.6, label=f'+{STD_MULT}σ')
        plt.plot(plot_df['continuous_timestamp'], plot_df['Lower'],
                 color='green', ls='--', alpha=0.6, label=f'-{STD_MULT}σ')
        plt.title(f'Mean Reversion – {product}')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUT_DIR / f'reverting_{product}.png')
        plt.close()

        log_prices = np.log(df_prod['mid_price'].dropna().values)
        hurst_results[product] = calculate_hurst(log_prices) if len(log_prices) > 100 else None

    print("\n--- Hurst Exponent ---")
    for product, h in hurst_results.items():
        if h is None:
            print(f"{product}: insufficient data")
        else:
            print(f"{product}: H = {h:.4f} → {_regime_label(h)}")

    print("\n--- Hurst Exponent (noise-filtered, downsampled) ---")
    for product in PRODUCTS:
        df_prod = df[df['product'] == product].copy()
        if df_prod.empty:
            continue
        df_prod = df_prod.sort_values('continuous_timestamp')
        mid = (df_prod['bid_price_1'] + df_prod['ask_price_1']) / 2.0
        downsampled = mid.iloc[::SAMPLE_RATE].values
        if len(downsampled) > 0 and np.all(downsampled > 0):
            h = calculate_hurst_robust(np.log(downsampled), min_lag=5, max_lag=100)
            if h is None:
                print(f"{product}: insufficient data after downsampling")
            else:
                print(f"{product}: H = {h:.4f} → {_regime_label(h)}")
        else:
            print(f"{product}: invalid price data")

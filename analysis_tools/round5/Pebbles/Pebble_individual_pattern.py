import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / 'data' / 'round5'
PRODUCTS = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']

all_files = sorted(LOG_DIR.glob('prices_*.csv'))
if not all_files:
    print(f"No prices_*.csv found in '{LOG_DIR.resolve()}'.")
    exit()

df_raw = pd.concat([pd.read_csv(f, sep=';') for f in all_files], ignore_index=True)

if 'day' in df_raw.columns:
    df_raw['continuous_timestamp'] = df_raw['day'] * 1_000_000 + df_raw['timestamp']
    df_raw = df_raw.sort_values('continuous_timestamp')
else:
    df_raw['continuous_timestamp'] = df_raw['timestamp']
    df_raw = df_raw.sort_values('timestamp')

df_pebbles = df_raw[df_raw['product'].isin(PRODUCTS)].copy()
if 'mid_price' not in df_pebbles.columns:
    df_pebbles['mid_price'] = (df_pebbles['bid_price_1'] + df_pebbles['ask_price_1']) / 2

df = df_pebbles.pivot(
    index='continuous_timestamp', columns='product', values='mid_price'
).dropna()

for p in PRODUCTS:
    s = df[p].values
    fft = np.abs(np.fft.rfft(s - s.mean()))
    freqs = np.fft.rfftfreq(len(s))
    top3 = np.argsort(fft)[-3:][::-1]
    top_periods = [round(1 / freqs[i]) for i in top3 if freqs[i] > 0]
    print(f"\n=== {p} ===")
    print(f"  mean={s.mean():.1f}  std={s.std():.1f}  min={s.min():.1f}  max={s.max():.1f}")
    print(f"  ADF p={adfuller(s)[1]:.4f}")
    print(f"  Top FFT periods (ticks): {top_periods}")

fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
for ax, p in zip(axes, PRODUCTS):
    ax.plot(df[p].values, lw=0.8)
    ax.set_ylabel(p.replace('PEBBLES_', ''), fontsize=8)
plt.tight_layout()
out_path = Path(__file__).parent / 'pebbles_patterns.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")

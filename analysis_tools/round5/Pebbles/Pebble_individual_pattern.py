import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

# Caricamento dati (path risolto rispetto alla root del repo)
repo_root = Path(__file__).resolve().parents[2]
log_dir = repo_root / 'data' / 'round5'
all_files = sorted(log_dir.glob('prices_*.csv'))

if not all_files:
    print(f"Errore: Nessun file prices_*.csv trovato in '{log_dir.resolve()}'.")
    exit()

df_list = [pd.read_csv(file, sep=';') for file in all_files]
df_raw = pd.concat(df_list, ignore_index=True)

if 'day' in df_raw.columns:
    df_raw['continuous_timestamp'] = df_raw['day'] * 1000000 + df_raw['timestamp']
    df_raw = df_raw.sort_values(by='continuous_timestamp')
else:
    df_raw['continuous_timestamp'] = df_raw['timestamp']
    df_raw = df_raw.sort_values(by='timestamp')

products = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']
df_pebbles = df_raw[df_raw['product'].isin(products)].copy()
if 'mid_price' not in df_pebbles.columns:
    df_pebbles['mid_price'] = (df_pebbles['bid_price_1'] + df_pebbles['ask_price_1']) / 2

df = df_pebbles.pivot(index='continuous_timestamp', columns='product', values='mid_price').dropna()

for p in products:
    s = df[p].values
    print(f"\n=== {p} ===")
    print(f"  mean={s.mean():.1f}  std={s.std():.1f}  min={s.min():.1f}  max={s.max():.1f}")
    print(f"  ADF p={adfuller(s)[1]:.4f}")

    # Cerca periodicità con FFT
    fft = np.abs(np.fft.rfft(s - s.mean()))
    freqs = np.fft.rfftfreq(len(s))
    top3 = np.argsort(fft)[-3:][::-1]
    print(f"  Top FFT periods (tick): {[round(1 / freqs[i]) for i in top3 if freqs[i] > 0]}")

# Plot tutti e 5
fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
for ax, p in zip(axes, products):
    ax.plot(df[p].values, lw=0.8)
    ax.set_ylabel(p.replace('PEBBLES_', ''), fontsize=8)
plt.tight_layout()
out_path = Path(__file__).parent / "pebbles_patterns.png"
plt.savefig(out_path, dpi=150)
print(f"\nSalvato {out_path}")
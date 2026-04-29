import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

# Caricamento dati (path risolto rispetto alla root del repo)
repo_root = Path(__file__).resolve().parents[3]
log_dir = repo_root / 'data' / 'round5'
all_files = sorted(log_dir.glob('prices_*.csv'))

if not all_files:
    print(f"Errore: Nessun file prices_*.csv trovato in '{log_dir.resolve()}'.")
    exit()

df_list = [pd.read_csv(file, sep=';') for file in all_files]
df_raw = pd.concat(df_list, ignore_index=True)
df_raw['continuous_timestamp'] = df_raw['day'] * 1000000 + df_raw['timestamp']
df_raw = df_raw.sort_values(by='continuous_timestamp')

products = ['PEBBLES_XS','PEBBLES_S','PEBBLES_M','PEBBLES_L','PEBBLES_XL']
df_pebbles = df_raw[df_raw['product'].isin(products)].copy()
if 'mid_price' not in df_pebbles.columns:
    df_pebbles['mid_price'] = (df_pebbles['bid_price_1'] + df_pebbles['ask_price_1']) / 2

df = df_pebbles.pivot(index='continuous_timestamp', columns='product', values='mid_price').dropna()

# Split per giorno usando l'indice continuous_timestamp = day * 1_000_000 + timestamp
day_col = (df.index // 1_000_000).astype(int)
days = {int(d): df[day_col == d] for d in sorted(day_col.unique())}

for day_num, ddf in days.items():
    print(f"\n=== DAY {day_num} ===")
    for p in products:
        s = ddf[p].values
        delta = s[-1] - s[0]
        pval  = adfuller(s)[1]
        print(f"  {p}: start={s[0]:.0f} end={s[-1]:.0f} delta={delta:+.0f}  ADF={pval:.3f}")

# Plot per giorno
n_days = len(days)
fig, axes = plt.subplots(5, n_days, figsize=(6 * n_days, 12), squeeze=False)
for row, p in enumerate(products):
    for col, (day_num, ddf) in enumerate(days.items()):
        axes[row, col].plot(ddf[p].values, lw=0.8)
        axes[row, col].set_title(f"{p} - Day {day_num}", fontsize=8)
plt.tight_layout()
out_path = Path(__file__).parent / "pebbles_by_day.png"
plt.savefig(out_path, dpi=150)
print(f"\nSalvato {out_path}")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / 'data' / 'round5'
PRODUCTS = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']

all_files = sorted(LOG_DIR.glob('prices_*.csv'))
if not all_files:
    print(f"No prices_*.csv found in '{LOG_DIR.resolve()}'.")
    exit()

df_raw = pd.concat([pd.read_csv(f, sep=';') for f in all_files], ignore_index=True)
# continuous_timestamp = day * 1_000_000 + intraday_tick
df_raw['continuous_timestamp'] = df_raw['day'] * 1_000_000 + df_raw['timestamp']
df_raw = df_raw.sort_values('continuous_timestamp')

df_pebbles = df_raw[df_raw['product'].isin(PRODUCTS)].copy()
if 'mid_price' not in df_pebbles.columns:
    df_pebbles['mid_price'] = (df_pebbles['bid_price_1'] + df_pebbles['ask_price_1']) / 2

df = df_pebbles.pivot(
    index='continuous_timestamp', columns='product', values='mid_price'
).dropna()

day_col = (df.index // 1_000_000).astype(int)
days = {int(d): df[day_col == d] for d in sorted(day_col.unique())}

for day_num, ddf in days.items():
    print(f"\n=== DAY {day_num} ===")
    for p in PRODUCTS:
        s = ddf[p].values
        print(f"  {p}: start={s[0]:.0f}  end={s[-1]:.0f}  delta={s[-1]-s[0]:+.0f}  ADF={adfuller(s)[1]:.3f}")

n_days = len(days)
fig, axes = plt.subplots(5, n_days, figsize=(6 * n_days, 12), squeeze=False)
for row, p in enumerate(PRODUCTS):
    for col, (day_num, ddf) in enumerate(days.items()):
        axes[row, col].plot(ddf[p].values, lw=0.8)
        axes[row, col].set_title(f"{p} - Day {day_num}", fontsize=8)
plt.tight_layout()
out_path = Path(__file__).parent / 'pebbles_by_day.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
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
P = df[PRODUCTS].values

print("=== JOHANSEN TEST (all 5) ===")
res = coint_johansen(P, det_order=0, k_ar_diff=1)
for i in range(5):
    trace = res.lr1[i]
    crit = res.cvt[i, 1]  # 95% critical value
    print(f"r<={i}: trace={trace:.2f}  crit95={crit:.2f}  {'✅' if trace > crit else '❌'}")

print("\n=== COINTEGRATING VECTORS (first 3) ===")
print(res.evec[:, :3])

PAIRS = [
    ('PEBBLES_XS', 'PEBBLES_S'),
    ('PEBBLES_S', 'PEBBLES_M'),
    ('PEBBLES_M', 'PEBBLES_L'),
    ('PEBBLES_L', 'PEBBLES_XL'),
]

print("\n=== ADJACENT RATIOS ===")
for A, B in PAIRS:
    ratio = df[A] / df[B]
    pval = adfuller(ratio)[1]
    print(f"{A}/{B}: mean={ratio.mean():.4f}  std={ratio.std():.4f}  ADF p={pval:.4f}  {'✅' if pval < 0.05 else '❌'}")

print("\n=== ADJACENT DIFFERENCES ===")
for A, B in PAIRS:
    diff = df[A] - df[B]
    pval = adfuller(diff)[1]
    print(f"{A}-{B}: mean={diff.mean():.2f}  std={diff.std():.2f}  ADF p={pval:.4f}  {'✅' if pval < 0.05 else '❌'}")

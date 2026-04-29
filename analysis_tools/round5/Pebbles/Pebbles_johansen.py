import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
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
P = df[products].values

# --- 1. JOHANSEN su tutti e 5 ---
print("=== JOHANSEN TEST ===")
res = coint_johansen(P, det_order=0, k_ar_diff=1)
for i in range(5):
    trace = res.lr1[i]
    crit  = res.cvt[i, 1]  # 95%
    print(f"r<={i}: trace={trace:.2f}  crit95={crit:.2f}  {'✅' if trace > crit else '❌'}")

print("\n=== VETTORI COINTEGRANTI (se esistono) ===")
print(res.evec[:, :3])  # prime 3 colonne = combinazioni lineari più forti

# --- 2. RAPPORTI TRA SIZE ADIACENTI ---
print("\n=== RAPPORTI ADIACENTI (literal meaning) ===")
pairs = [('PEBBLES_XS','PEBBLES_S'),
         ('PEBBLES_S', 'PEBBLES_M'),
         ('PEBBLES_M', 'PEBBLES_L'),
         ('PEBBLES_L', 'PEBBLES_XL')]
for A, B in pairs:
    ratio = df[A] / df[B]
    pval  = adfuller(ratio)[1]
    print(f"{A}/{B}: mean={ratio.mean():.4f}  std={ratio.std():.4f}  ADF p={pval:.4f}  {'✅' if pval<0.05 else '❌'}")

# --- 3. DIFFERENZE TRA SIZE ADIACENTI ---
print("\n=== DIFFERENZE ADIACENTI ===")
for A, B in pairs:
    diff = df[A] - df[B]
    pval = adfuller(diff)[1]
    print(f"{A}-{B}: mean={diff.mean():.2f}  std={diff.std():.2f}  ADF p={pval:.4f}  {'✅' if pval<0.05 else '❌'}")
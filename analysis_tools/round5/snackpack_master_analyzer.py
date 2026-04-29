import pandas as pd
import numpy as np
import glob
import math
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# --- CONFIGURATION ---
SNACKPACKS = [
    'SNACKPACK_CHOCOLATE',
    'SNACKPACK_PISTACHIO',
    'SNACKPACK_RASPBERRY',
    'SNACKPACK_STRAWBERRY',
    'SNACKPACK_VANILLA'
]
DATA_DIR = "data/round5"

# --- HELPER FUNCTIONS ---
def hurst(ts: np.ndarray, max_lag: int = 100) -> float:
    ts = ts[~np.isnan(ts)]
    if len(ts) < max_lag * 2: return np.nan
    lags = range(2, max_lag)
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if len(tau) < 2 or any(t <= 0 for t in tau): return np.nan
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])

def load_data(csv_dir: str):
    files = sorted(glob.glob(str(Path(csv_dir) / "prices_round_5_day_*.csv")))
    dfs = []
    for f in files:
        d = pd.read_csv(f, sep=";")
        d.columns = [c.strip().lower() for c in d.columns]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

def main():
    print(f"--- SNACKPACK MASTER ANALYZER ---\n")
    df_all = load_data(DATA_DIR)
    pivot_all = df_all.pivot(index=["day", "timestamp"], columns="product", values="mid_price").ffill().dropna()
    pivot_snack = pivot_all[SNACKPACKS]

    print("="*80)
    print(" 1. INTERNAL CLUSTER METRICS & STATIONARITY")
    print("="*80)
    print("Correlation Matrix:")
    print(pivot_snack.corr().to_string(float_format="%.4f"))
    
    print("\nHurst Exponents:")
    for p in SNACKPACKS:
        h = hurst(pivot_snack[p].values)
        print(f"{p:30}: {h:.4f} ({'MEAN-REV' if h < 0.5 else 'TRENDING'})")

    cluster_sum = pivot_snack.sum(axis=1)
    print(f"\nCluster Sum (Basket v1):")
    print(f"  Mean: {cluster_sum.mean():.2f} | Std: {cluster_sum.std():.2f} | ADF p-val: {adfuller(cluster_sum.values)[1]:.4e}")

    print("\n" + "="*80)
    print(" 2. PAIRWISE STATIONARITY (REVERTERS)")
    print("="*80)
    pairs = [('SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA'), ('SNACKPACK_PISTACHIO', 'SNACKPACK_RASPBERRY')]
    for p1, p2 in pairs:
        pair_sum = pivot_snack[p1] + pivot_snack[p2]
        print(f"Pair {p1} + {p2}:")
        print(f"  Mean: {pair_sum.mean():.2f} | Std: {pair_sum.std():.2f} | ADF p-val: {adfuller(pair_sum.values)[1]:.4e}")

    print("\n" + "="*80)
    print(" 3. JOHANSEN COINTEGRATION (OPTIMAL WEIGHTS)")
    print("="*80)
    try:
        j_res = coint_johansen(pivot_snack, det_order=0, k_ar_diff=1)
        evec = j_res.evec[:, 0]
        evec = evec / evec[0]
        print("Weights:")
        for p, w in zip(SNACKPACKS, evec):
            print(f"  {p:30}: {w:.4f}")
        
        spread = pivot_snack.dot(evec)
        print(f"\nJohansen Spread:")
        print(f"  Mean: {spread.mean():.2f} | Std: {spread.std():.2f} | ADF p-val: {adfuller(spread.values)[1]:.4e}")
    except Exception as e:
        print(f"Johansen failed: {e}")

    print("\n" + "="*80)
    print(" 4. CROSS-CLUSTER ORACLES")
    print("="*80)
    # Most significant oracle found: Strawberry vs Visor Amber
    target = 'SNACKPACK_STRAWBERRY'
    oracle = 'UV_VISOR_AMBER'
    beta, alpha = np.polyfit(pivot_all[oracle], pivot_all[target], 1)
    spread = pivot_all[target] - (alpha + beta * pivot_all[oracle])
    print(f"Target: {target} | Oracle: {oracle}")
    print(f"  Model: {target} = {alpha:.2f} + {beta:.4f} * {oracle}")
    print(f"  Spread Std: {spread.std():.2f} | ADF p-val: {adfuller(spread.values)[1]:.4e}")

if __name__ == "__main__":
    main()

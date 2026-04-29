import pandas as pd
import numpy as np
import glob
import math
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def load_data(csv_dir: str):
    files = sorted(glob.glob(str(Path(csv_dir) / "prices_round_5_day_*.csv")))
    dfs = []
    for f in files:
        d = pd.read_csv(f, sep=";")
        d.columns = [c.strip().lower() for c in d.columns]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

def hurst(ts: np.ndarray, max_lag: int = 100) -> float:
    lags = range(2, min(max_lag, len(ts) // 2))
    tau  = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    if len(tau) < 2 or any(t <= 0 for t in tau):
        return float("nan")
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])

def main():
    df = load_data("data/round5")
    vr_products = ['UV_VISOR_MAGENTA', 'UV_VISOR_ORANGE', 'UV_VISOR_YELLOW', 'UV_VISOR_AMBER', 'UV_VISOR_RED']
    
    pivot_df = df[df["product"].isin(vr_products)].pivot(index=["day", "timestamp"], columns="product", values="mid_price").ffill()
    
    print("--- Correlation Matrix ---")
    print(pivot_df.corr().to_string(float_format="%.4f"))
    
    print("\n--- Hurst Exponent (< 0.5 is Mean Reverting) ---")
    for p in vr_products:
        h = hurst(pivot_df[p].values)
        print(f"{p}: {h:.4f}")

    print("\n--- ADF Test on Pairs (Stationarity) ---")
    # YELLOW + RED showed good results in analysis
    spread_yr = pivot_df["UV_VISOR_YELLOW"] + pivot_df["UV_VISOR_RED"]
    adf_yr = adfuller(spread_yr.values)
    print(f"YELLOW + RED p-value: {adf_yr[1]:.4f}")

    print("\n--- Johansen Cointegration Weights (Optimal Basket) ---")
    j_res = coint_johansen(pivot_df, det_order=0, k_ar_diff=1)
    evec = j_res.evec[:, 0]
    evec = evec / evec[0]
    for p, w in zip(vr_products, evec):
        print(f"{p}: {w:.4f}")

    print("\n--- Pairwise Z-Score Matrix Analysis (Strategy Base) ---")
    alpha = 0.05
    # Simulating the matrix signal used in v13
    signals = {p: 0.0 for p in vr_products}
    for i in range(len(vr_products)):
        for j in range(i + 1, len(vr_products)):
            p1, p2 = vr_products[i], vr_products[j]
            diff = pivot_df[p1] - pivot_df[p2]
            ema_diff = diff.ewm(alpha=alpha).mean()
            std_diff = diff.ewm(alpha=alpha).std()
            z = (diff - ema_diff) / std_diff
            final_z = z.iloc[-1]
            print(f"Pair {p1}-{p2} Z-score: {final_z:.4f}")

if __name__ == "__main__":
    main()

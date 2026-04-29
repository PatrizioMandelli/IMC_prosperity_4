import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
import glob
import os
from pathlib import Path

# Configurazione Prodotti
ROBOTS = ['ROBOT_VACUUMING', 'ROBOT_MOPPING', 'ROBOT_DISHES', 'ROBOT_LAUNDRY', 'ROBOT_IRONING']
CHIPS = ['MICROCHIP_OVAL', 'MICROCHIP_SQUARE']

# Modelli usati in ema_robo_trader.py
MODELS = {
    'ROBOT_VACUUMING': (8883.57, 0.2096, -0.1053, 227.85),
    'ROBOT_MOPPING':   (10467.18, -0.2101, 0.1729, 480.70),
    'ROBOT_DISHES':    (12052.99, -0.2806, 0.0192, 310.83),
    'ROBOT_LAUNDRY':   (6119.02, 0.3793, 0.0442, 306.61),
    'ROBOT_IRONING':   (6538.64, 0.3789, -0.0689, 352.55)
}

def load_data(csv_dir: str):
    files = sorted(glob.glob(str(Path(csv_dir) / "prices_round_5_day_*.csv")))
    dfs = []
    for f in files:
        d = pd.read_csv(f, sep=";")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    return df

def analyze():
    print("=== ROBOT MASTER ANALYSIS (ROBOT vs MICROCHIP Stat-Arb) ===")
    df = load_data("data/round5")
    
    # Pivot data for robots and chips
    pivot_df = df[df["product"].isin(ROBOTS + CHIPS)].pivot(
        index=["day", "timestamp"], columns="product", values="mid_price"
    ).ffill().dropna()
    
    print("\n1. Summary Statistics (Mid Prices)")
    print(pivot_df[ROBOTS].describe().loc[['mean', 'std', 'min', 'max']])

    print("\n2. Correlation Matrix (ROBOTS vs CHIPS)")
    corr = pivot_df.corr().loc[CHIPS, ROBOTS]
    print(corr)

    print("\n3. PCA Analysis (ROBOT Cluster)")
    pca = PCA(n_components=3)
    pca.fit(pivot_df[ROBOTS])
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    print("\n4. Model Verification (EMA_ROBO_TRADER Residuals)")
    for r in ROBOTS:
        alpha, b_oval, b_square, std_err = MODELS[r]
        fair_value = alpha + b_oval * pivot_df['MICROCHIP_OVAL'] + b_square * pivot_df['MICROCHIP_SQUARE']
        residual = pivot_df[r] - fair_value
        
        res_mean = residual.mean()
        res_std = residual.std()
        print(f"{r:20} | Res Mean: {res_mean:8.2f} | Res Std: {res_std:8.2f} | Target Std: {std_err:8.2f}")

    print("\n5. New OLS Calibration (Check for Drift)")
    for r in ROBOTS:
        y = pivot_df[r]
        X = sm.add_constant(pivot_df[CHIPS])
        model = sm.OLS(y, X).fit()
        print(f"{r:20} | R-squared: {model.rsquared:.4f}")

if __name__ == "__main__":
    analyze()

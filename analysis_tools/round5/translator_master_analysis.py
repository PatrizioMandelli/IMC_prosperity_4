import pandas as pd
import numpy as np
from pathlib import Path
import glob
from statsmodels.tsa.stattools import adfuller

# ==============================================================================
# TRANSLATOR MASTER ANALYSIS
# ==============================================================================
# This file consolidates the research for the EMA Translator Trader.
# It focuses on the two primary stationary pairs found in the cluster.

TRANSLATORS = [
    'TRANSLATOR_ASTRO_BLACK', 
    'TRANSLATOR_ECLIPSE_CHARCOAL', 
    'TRANSLATOR_VOID_BLUE'
]

def load_data(data_dir: Path):
    files = sorted(glob.glob(str(data_dir / "prices_round_5_day_*.csv")))
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f, sep=";")
        temp_df.columns = [c.strip().lower() for c in temp_df.columns]
        df_list.append(temp_df)
    return pd.concat(df_list, ignore_index=True).sort_values(by=['day', 'timestamp'])

def analyze_pair(df, p1, p2):
    print(f"\n--- Analyzing Pair: {p1} vs {p2} ---")
    y, x = df[p1], df[p2]
    beta, alpha = np.polyfit(x, y, 1)
    spread = y - (alpha + beta * x)
    adf_p = adfuller(spread)[1]
    
    print(f"Model: {p1} = {alpha:.2f} + ({beta:.4f}) * {p2}")
    print(f"ADF p-value (Stationarity): {adf_p:.4f}")
    print(f"Standard Deviation (Sigma): {spread.std():.4f}")
    return alpha, beta, spread.std()

def main():
    DATA_PATH = Path("data/round5")
    if not DATA_PATH.exists():
        print(f"Error: Data path {DATA_PATH} not found.")
        return

    df_all = load_data(DATA_PATH)
    pivot_df = df_all[df_all['product'].isin(TRANSLATORS)].pivot_table(
        index=['day', 'timestamp'], columns='product', values='mid_price'
    ).ffill().dropna()

    # 1. ECLIPSE vs VOID (Primary Pair)
    analyze_pair(pivot_df, 'TRANSLATOR_ECLIPSE_CHARCOAL', 'TRANSLATOR_VOID_BLUE')

    # 2. ASTRO vs VOID (Secondary Pair)
    analyze_pair(pivot_df, 'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_VOID_BLUE')

if __name__ == "__main__":
    main()

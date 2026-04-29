import pandas as pd
import numpy as np
import glob
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

# --- CONFIGURATION ---
GALAXY = [
    'GALAXY_SOUNDS_BLACK_HOLES', 
    'GALAXY_SOUNDS_DARK_MATTER', 
    'GALAXY_SOUNDS_PLANETARY_RINGS', 
    'GALAXY_SOUNDS_SOLAR_FLAMES', 
    'GALAXY_SOUNDS_SOLAR_WINDS'
]
ORACLES = ['PEBBLES_S', 'UV_VISOR_YELLOW', 'PANEL_1X4', 'OXYGEN_SHAKE_GARLIC', 'OXYGEN_SHAKE_EVENING_BREATH']

def load_data():
    files = sorted(glob.glob("data/round5/prices_round_5_day_*.csv"))
    dfs = []
    for f in files:
        d = pd.read_csv(f, sep=";")
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

def main():
    print("--- GALAXY MASTER ANALYZER ---")
    df = load_data()
    pivot = df.pivot(index=["day", "timestamp"], columns="product", values="mid_price").ffill().dropna()
    
    # 1. High Conviction Spreads (Cross-Cluster)
    relationships = [
        ('GALAXY_SOUNDS_BLACK_HOLES', 'PEBBLES_S'),
        ('GALAXY_SOUNDS_DARK_MATTER', 'UV_VISOR_YELLOW'),
        ('GALAXY_SOUNDS_SOLAR_WINDS', 'PANEL_1X4'),
    ]
    
    print("\n1. Cross-Cluster Spreads:")
    for target, oracle in relationships:
        y, x = pivot[target].values, pivot[oracle].values
        beta, alpha = np.polyfit(x, y, 1)
        spread = y - (alpha + beta * x)
        adf_p = adfuller(spread)[1]
        print(f"Target: {target:30} | Oracle: {oracle:15} | Alpha: {alpha:10.2f} | Beta: {beta:10.4f} | ADF: {adf_p:.4e} | Std: {np.std(spread):.2f}")

    # 2. Internal Cluster Spreads
    internal_rels = [
        ('GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_DARK_MATTER'),
        ('GALAXY_SOUNDS_SOLAR_FLAMES', 'GALAXY_SOUNDS_SOLAR_WINDS')
    ]
    
    print("\n2. Internal Cluster Spreads:")
    for target, oracle in internal_rels:
        y, x = pivot[target].values, pivot[oracle].values
        beta, alpha = np.polyfit(x, y, 1)
        spread = y - (alpha + beta * x)
        adf_p = adfuller(spread)[1]
        print(f"Target: {target:30} | Oracle: {oracle:30} | Alpha: {alpha:10.2f} | Beta: {beta:10.4f} | ADF: {adf_p:.4e} | Std: {np.std(spread):.2f}")

    # 3. Correlation Matrix
    print("\n3. Cluster Correlation Matrix:")
    print(pivot[GALAXY].corr().to_string())

if __name__ == "__main__":
    main()

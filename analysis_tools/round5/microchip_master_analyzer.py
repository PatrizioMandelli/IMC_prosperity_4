import pandas as pd
import numpy as np
import glob
import math
from pathlib import Path

# --- CONFIGURAZIONE ---
CHIPS = ['MICROCHIP_SQUARE', 'MICROCHIP_RECTANGLE', 'MICROCHIP_TRIANGLE', 'MICROCHIP_OVAL', 'MICROCHIP_CIRCLE']
DATA_DIR = "data/round5"

# --- HELPER FUNCTIONS ---
def hurst(ts: np.ndarray, max_lag: int = 100) -> float:
    """Esponente di Hurst: < 0.5 = Mean Reverting, > 0.5 = Trending."""
    lags = range(2, min(max_lag, len(ts) // 2))
    tau  = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    if len(tau) < 2 or any(t <= 0 for t in tau): return float("nan")
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])

def ou_halflife(ts: np.ndarray) -> float:
    """Half-life del processo Ornstein-Uhlenbeck in tick."""
    diff = ts[1:] - ts[:-1]
    beta = np.polyfit(ts[:-1], diff, 1)[0]
    return float(-math.log(2) / beta) if beta < 0 else float("inf")

def load_data(csv_dir: str):
    files = sorted(glob.glob(str(Path(csv_dir) / "prices_round_5_day_*.csv")))
    dfs = []
    for f in files:
        d = pd.read_csv(f, sep=";")
        d.columns = [c.strip().lower() for c in d.columns]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)

# --- ANALISI CORE ---
def main():
    print(f"Lancio Analisi Master sui Microchips da: {DATA_DIR}\n")
    df = load_data(DATA_DIR)
    pivot_df = df[df["product"].isin(CHIPS)].pivot(index=["day", "timestamp"], columns="product", values="mid_price").ffill()

    print("="*80)
    print(" 1. MATRICE DI CORRELAZIONE")
    print("="*80)
    corr = pivot_df.corr()
    print(corr.to_string(float_format="%.4f"))

    # --- ANALISI COPPIA 1: SQUARE + RECTANGLE (Somma Mean Reverting) ---
    print("\n" + "="*80)
    print(" 2. ANALISI COPPIA A: SQUARE + RECTANGLE (CORR NEGATIVA)")
    print("="*80)
    s_price = pivot_df["MICROCHIP_SQUARE"].values
    r_price = pivot_df["MICROCHIP_RECTANGLE"].values
    
    # Calcolo Beta ottimale per minimizzare varianza spread
    beta_sr = np.std(s_price) / np.std(r_price)
    spread_sr = s_price + beta_sr * r_price # Nota il '+' per corr negativa
    
    h_sr = hurst(spread_sr)
    hl_sr = ou_halflife(spread_sr)
    
    print(f"BETA Ottimale (Std_Sq/Std_Rect): {beta_sr:.4f}")
    print(f"HURST Spread (S + b*R): {h_sr:.4f} ({'MEAN-REV' if h_sr < 0.5 else 'TRENDING'})")
    print(f"HALF-LIFE: {hl_sr:.1f} ticks")
    print(f"MEAN Spread: {np.mean(spread_sr):.2f}")
    print(f"STD Spread: {np.std(spread_sr):.2f}")

    # --- ANALISI COPPIA 2: TRIANGLE - OVAL (Differenza Mean Reverting) ---
    print("\n" + "="*80)
    print(" 3. ANALISI COPPIA B: TRIANGLE - OVAL (CORR POSITIVA)")
    print("="*80)
    t_price = pivot_df["MICROCHIP_TRIANGLE"].values
    o_price = pivot_df["MICROCHIP_OVAL"].values
    
    # Regressione lineare per Beta Triangle = Beta * Oval
    beta_to = np.polyfit(o_price, t_price, 1)[0]
    spread_to = t_price - beta_to * o_price
    
    h_to = hurst(spread_to)
    hl_to = ou_halflife(spread_to)
    
    print(f"BETA Ottimale (Triangle = b * Oval): {beta_to:.4f}")
    print(f"HURST Spread (T - b*O): {h_to:.4f} ({'MEAN-REV' if h_to < 0.5 else 'TRENDING'})")
    print(f"HALF-LIFE: {hl_to:.1f} ticks")
    print(f"MEAN Spread: {np.mean(spread_to):.2f}")
    print(f"STD Spread: {np.std(spread_to):.2f}")

    # --- SINTESI PER IL BOT ---
    print("\n" + "="*80)
    print(" 4. PARAMETRI PER IL TRADER (COPIA E INCOLLA)")
    print("="*80)
    print(f"self.beta_sr = {beta_sr:.4f}")
    print(f"self.mean_sr = {np.mean(spread_sr):.2f}")
    print(f"self.std_sr  = {np.std(spread_sr):.2f}")
    print("-" * 30)
    print(f"self.beta_to = {beta_to:.4f}")
    print(f"self.mean_to = {np.mean(spread_to):.2f}")
    print(f"self.std_to  = {np.std(spread_to):.2f}")
    print("-" * 30)
    print(f"CIRCLE_MEAN = {pivot_df['MICROCHIP_CIRCLE'].mean():.2f}")

if __name__ == "__main__":
    main()

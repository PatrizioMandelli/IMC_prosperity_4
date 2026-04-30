import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import math

# ─── CONFIGURAZIONE ──────────────────────────────────────────────────
# Path risolto rispetto alla root del repo
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "round5"
POS_LIMIT = 7  # Limite per i ROBOTS trovato in ema_robo_trader.py

def load_prices(data_dir, max_days=None):
    """Carica e formatta i dati dei prezzi dai CSV."""
    print(f"Caricamento dati da {data_dir}...")
    files = sorted(Path(data_dir).glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError("Nessun file prices_*.csv trovato. Controlla il DATA_DIR.")

    if max_days is not None:
        files = files[:max_days]
        print(f"Filtrato: Caricamento solo dei primi {max_days} giorni: {[f.name for f in files]}")

    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    
    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2

    wide = df.pivot_table(index=['day', 'timestamp'], columns='product',
                          values='mid_price', aggfunc='last')
    # Fill forward per gestire tick mancanti (comune in Round 5)
    return wide.groupby(level='day').ffill().dropna(how='all')

def analyze_robot_vs_chips(robot_name: str, chips: list, prices: pd.DataFrame):
    """
    Esegue OLS: Robot ~ const + Chip1 + Chip2 + ...
    Calcola residui, ADF test e genera la configurazione per il bot.
    """
    print(f"\n{'-' * 70}")
    print(f"ANALISI: {robot_name} vs components {chips}")
    print(f"{'-' * 70}")

    cols = [robot_name] + chips
    # Verifichiamo che tutti i prodotti esistano nel dataframe
    missing = [c for c in cols if c not in prices.columns]
    if missing:
        print(f"ERRORE: Prodotti mancanti nel dataset: {missing}")
        return

    data = prices[cols].dropna()
    if len(data) < 100:
        print(f"AVVISO: Troppi pochi dati ({len(data)} tick) per {robot_name}")
        return

    # Regressione OLS
    Y = data[robot_name]
    X = sm.add_constant(data[chips])
    model = sm.OLS(Y, X).fit()

    spread = model.resid
    adf_p = adfuller(spread, autolag='AIC')[1]

    coeffs = model.params
    intercept = coeffs['const']
    betas = {chip: coeffs[chip] for chip in chips}
    
    print(f"Statistiche di regressione:")
    print(f"  R² = {model.rsquared:.6f}")
    print(f"  ADF p-value = {adf_p:.6f} ({'STAZIONARIO' if adf_p < 0.05 else 'NON STAZIONARIO'})")
    print(f"  Spread Std Dev: {spread.std():.4f}")
    
    print(f"\nCoefficienti:")
    print(f"  Intercept: {intercept:.4f}")
    for chip, beta in betas.items():
        print(f"  Beta {chip}: {beta:.4f}")

    # Lot sizing (opzionale, se si volesse tradare come basket puro)
    # robots_ols_solver.py style: calcolo lotti basato su pesi
    max_beta = max(abs(v) for v in betas.values())
    lots = {robot_name: POS_LIMIT}
    for chip, beta in betas.items():
        lots[chip] = max(1, round(abs(beta) * POS_LIMIT))

    # Genera il blocco di codice per il trader
    print("\n[ COPIA E INCOLLA QUESTO NEL TRADER (robo_models dict) ]\n")
    # Formato standard usato in ema_robo_trader.py: (intercept, beta_oval, beta_square, spread_std)
    # Assumiamo l'ordine OVAL, SQUARE come default
    intercept_val = float(intercept)
    beta_oval = float(betas.get('MICROCHIP_OVAL', 0))
    beta_square = float(betas.get('MICROCHIP_SQUARE', 0))
    std_val = float(spread.std())

    output_str = f"({intercept_val:.2f}, {beta_oval:.4f}, {beta_square:.4f}, {std_val:.2f})"
    print(f"    '{robot_name}': {output_str},")

if __name__ == "__main__":
    try:
        # Per runnare solo sui primi due giorni, passiamo max_days=2
        prices_df = load_prices(DATA_DIR, max_days=2)

        robots = [
            'ROBOT_VACUUMING',
            'ROBOT_MOPPING',
            'ROBOT_DISHES',
            'ROBOT_LAUNDRY',
            'ROBOT_IRONING'
        ]
        
        # Componenti identificati in ema_robo_trader.py
        chips_to_test = ['MICROCHIP_OVAL', 'MICROCHIP_SQUARE']

        for robot in robots:
            analyze_robot_vs_chips(robot, chips_to_test, prices_df)
            
    except Exception as e:
        print(f"ERRORE DURANTE L'ESECUZIONE: {e}")

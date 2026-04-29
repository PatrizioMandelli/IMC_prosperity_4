import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Configurazione (path risolto rispetto alla root del repo)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / 'data' / 'round5'

# I pesi estratti dalla tua OLS
BASKET_WEIGHTS = {
    'UV_VISOR_MAGENTA': 1.0,
    'UV_VISOR_ORANGE': 0.0653,
    'UV_VISOR_YELLOW': -0.0312,
    'UV_VISOR_AMBER': 0.6566,
    'UV_VISOR_RED': 0.2874
}


def load_data(data_dir):
    data_dir = Path(data_dir)
    print(f"Caricamento dati da {data_dir}...")
    all_files = sorted(data_dir.glob("prices_*.csv"))

    if not all_files:
        raise FileNotFoundError(f"Nessun file prices_*.csv trovato in '{data_dir.resolve()}'")

    df_list = [pd.read_csv(file, sep=';') for file in all_files]
    full_df = pd.concat(df_list, ignore_index=True)

    # Ordiniamo per timestamp nel caso i file fossero in disordine
    full_df = full_df.sort_values(by=['day', 'timestamp'])

    # Creiamo il pivot per avere i prodotti in colonna
    pivot_df = full_df.pivot(index=['day', 'timestamp'], columns='product', values='mid_price')
    pivot_df = pivot_df.dropna(subset=list(BASKET_WEIGHTS.keys()))

    return pivot_df


def simulate_alpha(spread_series, alpha):
    """
    Simula la logica del bot su una serie storica dello spread
    per un dato alpha e calcola un PnL teorico.
    """
    ema = spread_series.iloc[0]
    var = 0.0

    pnl = 0.0
    position = 0.0  # Posizione scalata (da -1 a 1)

    pnls = []

    for i in range(len(spread_series)):
        current_spread = spread_series.iloc[i]

        # 1. Aggiorna il PnL in base al movimento dello spread dalla candela precedente
        if i > 0:
            price_change = current_spread - spread_series.iloc[i - 1]
            pnl += position * price_change
        pnls.append(pnl)

        # 2. Matematica del Bot (Aggiornamento EMA e Var)
        dev = current_spread - ema
        ema = alpha * current_spread + (1 - alpha) * ema
        var = alpha * (dev ** 2) + (1 - alpha) * var

        std_dev = math.sqrt(var) if var > 0 else 1.0

        # 3. Calcolo Z-Score
        z_score = dev / std_dev
        z_capped = max(min(z_score, 3.0), -3.0)

        # 4. Assegnazione Posizione
        # Se z > 0 (spread alto) -> vendiamo (posizione negativa)
        # Se z < 0 (spread basso) -> compriamo (posizione positiva)
        # Dividiamo per 3 per scalare la posizione tra -1 e 1
        position = -z_capped / 3.0

    return pnl, pnls


if __name__ == "__main__":
    df = load_data(DATA_DIR)

    print("\nCalcolo dello Spread Sintetico...")
    df['SPREAD'] = 0
    for product, weight in BASKET_WEIGHTS.items():
        df['SPREAD'] += df[product] * weight

    spread_series = df['SPREAD']

    # Griglia degli alpha da testare
    alphas_to_test = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]

    results = {}
    best_alpha = None
    best_pnl = -float('inf')
    best_pnl_curve = []

    print("\nInizio simulazione backtest per diversi Alpha...\n")
    print(f"{'ALPHA':<10} | {'PNL TEORICO':<15}")
    print("-" * 30)

    for alpha in alphas_to_test:
        final_pnl, pnl_curve = simulate_alpha(spread_series, alpha)
        results[alpha] = final_pnl

        print(f"{alpha:<10} | {final_pnl:>10.2f}")

        if final_pnl > best_pnl:
            best_pnl = final_pnl
            best_alpha = alpha
            best_pnl_curve = pnl_curve

    print("-" * 30)
    print(f"\n🏆 L'Alpha ottimale è: {best_alpha} (PnL: {best_pnl:.2f})")

    # Plottiamo la curva dei profitti del miglior alpha
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(best_pnl_curve)), best_pnl_curve, color='green', label=f'PnL con Alpha = {best_alpha}')
    plt.title(f'Curva dei Profitti Teorica - Miglior Alpha: {best_alpha}')
    plt.xlabel('Timestamp (Tick)')
    plt.ylabel('Cumulative PnL (Unità di Spread)')
    plt.grid(True)
    plt.legend()
    plt.show()
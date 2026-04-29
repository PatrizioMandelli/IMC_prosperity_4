import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path risolto rispetto alla root del repo
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / 'data' / 'round5'

# Concentriamoci sul prodotto guida
TARGET_PRODUCT = 'UV_VISOR_MAGENTA'


def analyze_microstructure(data_dir, product):
    data_dir = Path(data_dir)
    print(f"Analisi della microstruttura per {product}...")
    all_files = sorted(data_dir.glob("prices_*.csv"))
    if not all_files:
        raise FileNotFoundError(f"Nessun file prices_*.csv trovato in '{data_dir.resolve()}'")
    df_list = [pd.read_csv(f, sep=';') for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=['day', 'timestamp'])

    # Filtriamo solo il prodotto bersaglio
    df = df[df['product'] == product].copy()

    # 1. Calcolo dello Spread Bid-Ask Reale (Il "Respiro" del Book)
    df['bid_ask_spread'] = df['ask_price_1'] - df['bid_price_1']

    # 2. Calcolo della Profondità del Book (Quanta liquidità c'è al primo livello)
    df['top_level_liquidity'] = df['bid_volume_1'] + df['ask_volume_1']

    # Selezioniamo un campione per la visualizzazione (es. 1000 tick)
    limit = 1000
    sample_df = df.iloc[5000:5000 + limit].copy()  # Prendiamo un pezzo centrale a caso
    timestamps = sample_df['timestamp']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # GRAFICO 1: L'azione del Prezzo (Mid Price)
    ax1.plot(timestamps, sample_df['mid_price'], color='purple', label='Mid Price')
    ax1.set_title(f'Microstruttura di {product}', fontsize=14)
    ax1.set_ylabel('Prezzo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # GRAFICO 2: Il "Costo" del Mercato (Spread Bid-Ask)
    ax2.plot(timestamps, sample_df['bid_ask_spread'], color='red', alpha=0.7, label='Bid-Ask Spread (Tick)')
    ax2.axhline(sample_df['bid_ask_spread'].mean(), color='black', linestyle='--', label='Spread Medio')
    ax2.set_ylabel('Spread (Costo)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # GRAFICO 3: La Liquidità (Volumi al primo livello)
    # Riempiamo l'area per far capire quanto è "spesso" il muro di ordini
    ax3.fill_between(timestamps, 0, sample_df['top_level_liquidity'], color='blue', alpha=0.3,
                     label='Liquidità L1 (Bid_Vol + Ask_Vol)')
    ax3.set_ylabel('Volume (Pezzi)')
    ax3.set_xlabel('Timestamp')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_microstructure(DATA_DIR, TARGET_PRODUCT)
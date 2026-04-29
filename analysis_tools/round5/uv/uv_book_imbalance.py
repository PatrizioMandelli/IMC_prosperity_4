import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path risolto rispetto alla root del repo (round 5 ha day 2/3/4, non day 1)
REPO_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = REPO_ROOT / 'data' / 'round5' / 'prices_round_5_day_2.csv'

BASKET_WEIGHTS = {
    'UV_VISOR_MAGENTA': 1.0,
    'UV_VISOR_ORANGE': 0.065,
    'UV_VISOR_YELLOW': -0.03,
    'UV_VISOR_AMBER': 0.65,
    'UV_VISOR_RED': 0.3
}


def analyze_imbalance(file_path):
    print("Caricamento dati...")
    df = pd.read_csv(file_path, sep=';')

    # Filtriamo solo i visori
    df = df[df['product'].isin(BASKET_WEIGHTS.keys())].copy()

    # Calcoliamo l'Imbalance per ogni riga: (Bid_Vol - Ask_Vol) / (Bid_Vol + Ask_Vol)
    # Valore da -1 (Tutti vogliono vendere) a +1 (Tutti vogliono comprare)
    df['imbalance'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])

    # Facciamo un pivot per avere timestamp sulle righe
    pivot_price = df.pivot(index='timestamp', columns='product', values='mid_price')
    pivot_imb = df.pivot(index='timestamp', columns='product', values='imbalance')

    pivot_price = pivot_price.dropna()
    pivot_imb = pivot_imb.dropna()

    print("Calcolo Spread e Imbalance aggregato...")
    # Calcolo Spread Matematico
    spread = sum(pivot_price[p] * w for p, w in BASKET_WEIGHTS.items())

    # Calcolo Imbalance Aggregato del Basket
    # Se il peso è positivo, aggiungiamo il suo imbalance. Se negativo, lo sottraiamo (perché shortiamo)
    basket_imbalance = sum(pivot_imb[p] * np.sign(w) for p, w in BASKET_WEIGHTS.items())

    # Plottiamo una porzione di dati (es. primi 2000 tick) per vederci chiaro
    limit = 2000
    timestamps = pivot_price.index[:limit]
    spread_slice = spread.iloc[:limit]
    imb_slice = basket_imbalance.iloc[:limit]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:purple'
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Spread (Prezzo)', color=color)
    ax1.plot(timestamps, spread_slice, color=color, label='Spread (Colori)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Aggiungiamo la media mobile dello spread per capire quando siamo "fuori media"
    ax1.plot(timestamps, spread_slice.rolling(20).mean(), color='black', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:cyan'
    ax2.set_ylabel('Volume Imbalance', color=color)
    ax2.plot(timestamps, imb_slice, color=color, alpha=0.3, label='Imbalance Basket')
    # Lisciamo l'imbalance per vedere il trend dei volumi
    ax2.plot(timestamps, imb_slice.rolling(10).mean(), color='blue', linewidth=2, label='Imbalance (Media)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0, color='gray', linestyle=':')

    fig.tight_layout()
    plt.title('Analisi UV_VISOR: Spread vs Order Book Imbalance', fontsize=14)
    plt.show()


if __name__ == "__main__":
    analyze_imbalance(FILE_PATH)
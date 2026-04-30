import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Caricamento dei dati
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "round5"
OUT_DIR = REPO_ROOT / "analysis_tools" / "galaxy"

print(f"Caricamento dati da {DATA_DIR}...")
files = sorted(DATA_DIR.glob("prices_*.csv"))

if not files:
    print(f"Errore: Nessun file prices_*.csv trovato in {DATA_DIR}")
    df = pd.DataFrame()
else:
    df_list = [pd.read_csv(f, sep=';') for f in files]
    df = pd.concat(df_list, ignore_index=True)
    # Crea un timestamp continuo se ci sono più giorni
    if 'day' in df.columns:
        df['continuous_timestamp'] = df['day'] * 1000000 + df['timestamp']
        df = df.sort_values(by='continuous_timestamp')
    else:
        df['continuous_timestamp'] = df['timestamp']
        df = df.sort_values(by='timestamp')


def calculate_hurst(ts):
    """
    Calcola l'esponente di Hurst di una serie storica.
    Ritorna un valore tra 0 e 1.
    H < 0.5 -> mean reverting; H ~ 0.5 -> random walk; H > 0.5 -> trending.
    """
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


if not df.empty:
    # 2. Selezione dei prodotti target
    prodotti_target = [
        'GALAXY_SOUNDS_DARK_MATTER',
        'GALAXY_SOUNDS_BLACK_HOLES',
        'GALAXY_SOUNDS_SOLAR_WINDS',
        'GALAXY_SOUNDS_SOLAR_FLAMES',
        'GALAXY_SOUNDS_PLANETARY_RINGS'
    ]

    # Parametri della strategia
    finestra_rolling = 20  # Numero di osservazioni per la media mobile (da ottimizzare)
    moltiplicatore_std = 2.0  # Quante deviazioni standard usare per la soglia

    hurst_results = {}

    for prodotto in prodotti_target:
        print(f"\n--- Analisi in corso per: {prodotto} ---")

        df_prod = df[df['product'] == prodotto].copy()

        if df_prod.empty:
            print(f"Nessun dato trovato per {prodotto}.")
            continue

        df_prod = df_prod.sort_values(by='continuous_timestamp')

        # 3. Calcolo del Mid-Price
        if 'mid_price' not in df_prod.columns:
            df_prod['mid_price'] = (df_prod['bid_price_1'] + df_prod['ask_price_1']) / 2

        # 4. Bollinger Bands
        df_prod['SMA'] = df_prod['mid_price'].rolling(window=finestra_rolling).mean()
        df_prod['Std_Dev'] = df_prod['mid_price'].rolling(window=finestra_rolling).std()
        df_prod['Upper_Threshold'] = df_prod['SMA'] + (df_prod['Std_Dev'] * moltiplicatore_std)
        df_prod['Lower_Threshold'] = df_prod['SMA'] - (df_prod['Std_Dev'] * moltiplicatore_std)

        df_prod['spread'] = df_prod['ask_price_1'] - df_prod['bid_price_1']
        spread_medio = df_prod['spread'].mean()
        print(f"Spread medio calcolato: {spread_medio:.2f}")

        # 5. Visualizzazione (primi 2000 tick)
        plot_df = df_prod.head(2000)

        plt.figure(figsize=(14, 6))
        plt.plot(plot_df['continuous_timestamp'], plot_df['mid_price'], label='Mid Price', color='blue', alpha=0.8)
        plt.plot(plot_df['continuous_timestamp'], plot_df['SMA'], label=f'SMA ({finestra_rolling})', color='orange', linewidth=2)
        plt.plot(plot_df['continuous_timestamp'], plot_df['Upper_Threshold'], label=f'+{moltiplicatore_std} Std Dev',
                 color='red', linestyle='--', alpha=0.6)
        plt.plot(plot_df['continuous_timestamp'], plot_df['Lower_Threshold'], label=f'-{moltiplicatore_std} Std Dev',
                 color='green', linestyle='--', alpha=0.6)

        plt.title(f'Analisi Inefficienze Mean Reversion: {prodotto}')
        plt.xlabel('Timestamp (Continuo)')
        plt.ylabel('Prezzo')
        plt.legend()
        plt.grid(True, alpha=0.3)

        out_name = f"reverting_{prodotto}.png"
        plt.savefig(OUT_DIR / out_name)
        print(f"Grafico salvato in: analysis_tools/galaxy/{out_name}")
        plt.close()

        # 6. Esponente di Hurst sui log-prezzi
        log_prices = np.log(df_prod['mid_price'].dropna().values)
        if len(log_prices) > 100:
            h_value = calculate_hurst(log_prices)
            hurst_results[prodotto] = h_value
        else:
            hurst_results[prodotto] = None

    # 7. Report Hurst
    print("\n--- Analisi Esponente di Hurst ---\n")
    for prodotto, h_value in hurst_results.items():
        if h_value is None:
            print(f"{prodotto}: dati insufficienti")
            continue

        if h_value < 0.45:
            regime = "MEAN REVERTING Forte"
        elif h_value > 0.55:
            regime = "TRENDING Forte"
        else:
            regime = "RANDOM WALK (Rumore)"

        print(f"{prodotto}: H = {h_value:.4f} -> {regime}")


def calculate_hurst_robust(ts, min_lag=10, max_lag=200):
    """
    Calcola Hurst su lag distanziati per ignorare il rumore di microstruttura.
    """
    lags = np.arange(min_lag, max_lag, step=5)
    lags = lags[lags < len(ts)]

    if len(lags) < 5:
        return None

    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


if not df.empty:
    print("\n--- Analisi Esponente di Hurst (Noise-Filtered) ---\n")

    # In Prosperity i dati sono campionati ogni 100ms; sample_rate=20 ~ 2s.
    sample_rate = 40

    for prodotto in prodotti_target:
        df_prod = df[df['product'] == prodotto].copy()

        if df_prod.empty:
            continue

        df_prod = df_prod.sort_values(by='continuous_timestamp')
        mid_prices = (df_prod['bid_price_1'] + df_prod['ask_price_1']) / 2.0

        downsampled_prices = mid_prices.iloc[::sample_rate].values

        if len(downsampled_prices) > 0 and np.all(downsampled_prices > 0):
            log_prices = np.log(downsampled_prices)
            h_value = calculate_hurst_robust(log_prices, min_lag=5, max_lag=100)

            if h_value is None:
                print(f"{prodotto}: dati insufficienti dopo downsampling")
                continue

            if h_value < 0.45:
                regime = "MEAN REVERTING"
            elif h_value > 0.55:
                regime = "TRENDING"
            else:
                regime = "RANDOM WALK"

            print(f"{prodotto}: H = {h_value:.4f} -> {regime}")
        else:
            print(f"{prodotto}: Errore nei dati (prezzi non validi o insufficienti).")
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Configurazione Iniziale (path risolto rispetto alla root del repo)
REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / 'data' / 'round5'
PRODUCTS = [
    "UV_VISOR_MAGENTA",
    "UV_VISOR_ORANGE",
    "UV_VISOR_YELLOW",
    "UV_VISOR_AMBER",
    "UV_VISOR_RED"
]


def load_and_prepare_data(log_dir):
    all_files = sorted(Path(log_dir).glob('prices_*.csv'))
    if not all_files:
        raise FileNotFoundError(f"Nessun file prices_*.csv trovato in '{Path(log_dir).resolve()}'")

    print(f"Caricamento dati da {len(all_files)} file in {log_dir}...")
    df_list = [pd.read_csv(f, sep=';') for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    if 'day' in df.columns:
        df['continuous_timestamp'] = df['day'] * 1_000_000 + df['timestamp']
        df = df.sort_values(by='continuous_timestamp')
        index_col = 'continuous_timestamp'
    else:
        index_col = 'timestamp'

    # Filtriamo solo i prodotti che ci interessano
    df = df[df['product'].isin(PRODUCTS)].copy()

    # Creiamo un Pivot in modo che ogni colonna sia un prodotto e ogni riga un timestamp
    pivot_df = df.pivot(index=index_col, columns='product', values='mid_price')

    # Pulizia dati: rimuoviamo eventuali righe con dati mancanti
    pivot_df = pivot_df.dropna()
    return pivot_df


def run_ols_regression(df):
    print("\nEsecuzione Regressione OLS...")

    # Scegliamo MAGENTA come variabile dipendente (Y) e le altre come indipendenti (X)
    y = df['UV_VISOR_MAGENTA']
    X = df[['UV_VISOR_ORANGE', 'UV_VISOR_YELLOW', 'UV_VISOR_AMBER', 'UV_VISOR_RED']]

    # Aggiungiamo una costante (l'intercetta)
    X = sm.add_constant(X)

    # Eseguiamo il modello
    model = sm.OLS(y, X).fit()

    # Stampiamo il riepilogo statistico
    print(model.summary())
    return model


def plot_spread(df, model):
    # Costruiamo lo spread sintetico (La nostra combinazione lineare)
    # Spread = MAGENTA - (beta1*ORANGE + beta2*YELLOW + beta3*AMBER + beta4*RED)

    coef = model.params

    # Calcoliamo il valore predetto del MAGENTA
    predicted_magenta = (
            coef['const'] +
            coef['UV_VISOR_ORANGE'] * df['UV_VISOR_ORANGE'] +
            coef['UV_VISOR_YELLOW'] * df['UV_VISOR_YELLOW'] +
            coef['UV_VISOR_AMBER'] * df['UV_VISOR_AMBER'] +
            coef['UV_VISOR_RED'] * df['UV_VISOR_RED']
    )

    # Lo spread è la differenza tra il prezzo reale e il prezzo predetto (i residui)
    spread = df['UV_VISOR_MAGENTA'] - predicted_magenta

    # Tracciamo il grafico dello spread
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, spread, label='Spread Sintetico (Residui OLS)', color='purple', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Aggiungiamo linee a +1 e -1 deviazioni standard per visualizzare i potenziali trigger
    std_dev = spread.std()
    plt.axhline(std_dev * 2, color='red', linestyle='--', label='+2 Std Dev (Vendi Spread)')
    plt.axhline(-std_dev * 2, color='green', linestyle='--', label='-2 Std Dev (Compra Spread)')

    plt.title('Combinazione Lineare Intra-Basket (UV_VISOR)')
    plt.xlabel('Timestamp')
    plt.ylabel('Spread Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        # Esegui l'analisi
        data = load_and_prepare_data(LOG_DIR)
        fitted_model = run_ols_regression(data)

        print("\n--- PESI DA INSERIRE NEL BOT ---")
        print("self.basket_weights = {")
        print(f"    'UV_VISOR_MAGENTA': 1.0,")
        print(f"    'UV_VISOR_ORANGE': {-fitted_model.params['UV_VISOR_ORANGE']:.4f},")
        print(f"    'UV_VISOR_YELLOW': {-fitted_model.params['UV_VISOR_YELLOW']:.4f},")
        print(f"    'UV_VISOR_AMBER': {-fitted_model.params['UV_VISOR_AMBER']:.4f},")
        print(f"    'UV_VISOR_RED': {-fitted_model.params['UV_VISOR_RED']:.4f}")
        print("}")

        plot_spread(data, fitted_model)

    except FileNotFoundError as e:
        print(f"ERRORE: {e}")
    except KeyError as e:
        print(f"ERRORE nelle colonne: Verifica che i nomi dei prodotti nel CSV combacino. Dettaglio: {e}")
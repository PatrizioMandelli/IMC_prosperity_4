import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ─── CONFIGURAZIONE ──────────────────────────────────────────────────
# Sostituisci con il path corretto ai tuoi dati
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "round5"
POS_LIMIT = 10  # Sostituisci con il limite reale per i SNACKPACK se diverso


def load_prices(data_dir):
    """Carica e formatta i dati dei prezzi dai CSV."""
    print(f"Caricamento dati da {data_dir}...")
    files = sorted(Path(data_dir).glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError("Nessun file prices_*.csv trovato. Controlla il DATA_DIR.")

    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2

    wide = df.pivot_table(index=['day', 'timestamp'], columns='product',
                          values='mid_price', aggfunc='last')
    return wide.groupby(level='day').ffill().dropna(how='all')


def analyze_and_generate_registry(name: str, group: list, prices: pd.DataFrame):
    print(f"\n{'-' * 60}")
    print(f"Analisi {name}: {group}")
    print(f"{'-' * 60}")

    data = prices[group].dropna()
    best_model = {'r2': -1}

    # Prova ogni asset come "Target" (Y) e gli altri come "Others" (X)
    for target in group:
        others = [p for p in group if p != target]

        # OLS: Target ~ Others
        Y = data[target]
        X = sm.add_constant(data[others])
        model = sm.OLS(Y, X).fit()

        spread = model.resid
        adf_p = adfuller(spread, autolag='AIC')[1]

        if model.rsquared > best_model['r2']:
            coeffs = dict(zip(others, model.params[1:]))
            best_model = {
                'r2': model.rsquared,
                'target': target,
                'others': others,
                'weights': coeffs,
                'adf_p': adf_p,
                'spread_std': spread.std()
            }

    # Calcolo lotti interi (Lot Sizing) basato sul peso massimo
    max_w = max(abs(v) for v in best_model['weights'].values())
    # Moltiplicatore per scalare i lotti senza superare il POS_LIMIT sul target
    k = math_floor = int(POS_LIMIT / max(1.0, max_w)) if max_w > 1 else POS_LIMIT

    lots = {best_model['target']: POS_LIMIT}
    for p, w in best_model['weights'].items():
        # Arrotonda il lotto proporzionale, garantendo minimo 1 se c'è un peso
        lots[p] = max(1, round(abs(w) * POS_LIMIT))

    # Formattazione pesi per l'output
    formatted_weights = {k: round(v, 4) for k, v in best_model['weights'].items()}

    print(
        f"Miglior Target : {best_model['target']} (R² = {best_model['r2']:.4f}, ADF p-value = {best_model['adf_p']:.4f})")

    # Genera il blocco di codice per il trader
    print("\n[ COPIA E INCOLLA QUESTO NEL TRADER (BASKETS dict) ]\n")
    print(f"    '{name}': {{")
    print(f"        'target': '{best_model['target']}',")
    print(f"        'others': {best_model['others']},")
    print(f"        'weights': {formatted_weights},")
    print(f"        'lots': {lots},")
    print(f"        'z_entry': 2.0,       # Modificabile: soglia di entrata")
    print(f"        'z_exit': 0.5,        # Modificabile: soglia di uscita")
    print(f"        'window_size': 100    # Modificabile: tick per media mobile")
    print(f"    }},")


if __name__ == "__main__":
    prices_df = load_prices(DATA_DIR)

    snack_duo = ['SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA']
    snack_trio = ['SNACKPACK_PISTACHIO', 'SNACKPACK_RASPBERRY', 'SNACKPACK_STRAWBERRY']

    analyze_and_generate_registry("SNACK_DUO", snack_duo, prices_df)
    analyze_and_generate_registry("SNACK_TRIO", snack_trio, prices_df)
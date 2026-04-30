import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path(__file__).resolve().parents[3] / 'data' / 'round5'
POS_LIMIT = 10


def load_prices(data_dir):
    """Loads price CSVs and returns a wide pivot indexed by (day, timestamp)."""
    files = sorted(Path(data_dir).glob('prices_*.csv'))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {data_dir}")
    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    wide = df.pivot_table(
        index=['day', 'timestamp'], columns='product',
        values='mid_price', aggfunc='last',
    )
    return wide.groupby(level='day').ffill().dropna(how='all')


def analyze_and_generate_registry(name, group, prices):
    """Finds the best OLS target within a basket and prints trader config."""
    data = prices[group].dropna()
    best = {'r2': -1}

    for target in group:
        others = [p for p in group if p != target]
        model = sm.OLS(data[target], sm.add_constant(data[others])).fit()
        spread = model.resid
        adf_p = adfuller(spread, autolag='AIC')[1]
        if model.rsquared > best['r2']:
            best = {
                'r2': model.rsquared,
                'target': target,
                'others': others,
                'weights': dict(zip(others, model.params[1:])),
                'adf_p': adf_p,
                'spread_std': float(spread.std()),
            }

    lots = {best['target']: POS_LIMIT}
    for p, w in best['weights'].items():
        lots[p] = max(1, round(abs(w) * POS_LIMIT))
    formatted_weights = {k: round(v, 4) for k, v in best['weights'].items()}

    print(f"\n{name}: target={best['target']}  R²={best['r2']:.4f}  ADF p={best['adf_p']:.4f}")
    print(f"  '{name}': {{")
    print(f"      'target': '{best['target']}',")
    print(f"      'others': {best['others']},")
    print(f"      'weights': {formatted_weights},")
    print(f"      'lots': {lots},")
    print(f"      'z_entry': 2.0,")
    print(f"      'z_exit': 0.5,")
    print(f"      'window_size': 100,")
    print(f"  }},")


if __name__ == '__main__':
    prices_df = load_prices(DATA_DIR)
    analyze_and_generate_registry(
        'SNACK_DUO', ['SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA'], prices_df)
    analyze_and_generate_registry(
        'SNACK_TRIO', ['SNACKPACK_PISTACHIO', 'SNACKPACK_RASPBERRY', 'SNACKPACK_STRAWBERRY'], prices_df)

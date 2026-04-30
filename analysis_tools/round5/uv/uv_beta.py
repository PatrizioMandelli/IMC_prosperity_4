import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / 'data' / 'round5'
PRODUCTS = [
    'UV_VISOR_MAGENTA',
    'UV_VISOR_ORANGE',
    'UV_VISOR_YELLOW',
    'UV_VISOR_AMBER',
    'UV_VISOR_RED',
]
REGRESSORS = ['UV_VISOR_ORANGE', 'UV_VISOR_YELLOW', 'UV_VISOR_AMBER', 'UV_VISOR_RED']


def load_and_prepare_data(log_dir):
    """Loads prices and returns a wide pivot indexed by continuous timestamp."""
    files = sorted(Path(log_dir).glob('prices_*.csv'))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in '{Path(log_dir).resolve()}'")
    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    if 'day' in df.columns:
        df['ts'] = df['day'] * 1_000_000 + df['timestamp']
        df = df.sort_values('ts')
        index_col = 'ts'
    else:
        index_col = 'timestamp'
    df = df[df['product'].isin(PRODUCTS)].copy()
    pivot = df.pivot(index=index_col, columns='product', values='mid_price')
    return pivot.dropna()


def run_ols(df):
    """OLS: MAGENTA ~ const + ORANGE + YELLOW + AMBER + RED."""
    y = df['UV_VISOR_MAGENTA']
    X = sm.add_constant(df[REGRESSORS])
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


def plot_spread(df, model):
    """Plots OLS residual spread with ±2σ bands."""
    coef = model.params
    predicted = coef['const'] + sum(coef[c] * df[c] for c in REGRESSORS)
    spread = df['UV_VISOR_MAGENTA'] - predicted
    std_dev = spread.std()

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, spread, label='OLS residual', color='purple', alpha=0.7)
    plt.axhline(0, color='black', ls='--', lw=1)
    plt.axhline(std_dev * 2, color='red', ls='--', label='+2σ')
    plt.axhline(-std_dev * 2, color='green', ls='--', label='−2σ')
    plt.title('UV_VISOR Intra-Basket OLS Spread')
    plt.xlabel('Timestamp')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    try:
        data = load_and_prepare_data(LOG_DIR)
        model = run_ols(data)
        print("\nBasket weights for bot:")
        print("basket_weights = {")
        print("    'UV_VISOR_MAGENTA': 1.0,")
        for col in REGRESSORS:
            print(f"    '{col}': {-model.params[col]:.4f},")
        print("}")
        plot_spread(data, model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Column mismatch: {e}")

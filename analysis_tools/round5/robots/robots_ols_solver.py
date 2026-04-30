import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / 'data' / 'round5'
POS_LIMIT = 7


def load_prices(data_dir, max_days=None):
    """Loads price CSVs and returns a wide pivot indexed by (day, timestamp)."""
    files = sorted(Path(data_dir).glob('prices_*.csv'))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {data_dir}")
    if max_days is not None:
        files = files[:max_days]
    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    wide = df.pivot_table(
        index=['day', 'timestamp'], columns='product',
        values='mid_price', aggfunc='last',
    )
    # Forward-fill within each day to handle missing ticks.
    return wide.groupby(level='day').ffill().dropna(how='all')


def analyze_robot_vs_chips(robot_name, chips, prices):
    """OLS of robot ~ chips; prints spread diagnostics and trader config tuple."""
    cols = [robot_name] + chips
    missing = [c for c in cols if c not in prices.columns]
    if missing:
        print(f"Missing products: {missing}")
        return

    data = prices[cols].dropna()
    if len(data) < 100:
        print(f"Insufficient data for {robot_name}: {len(data)} ticks")
        return

    model = sm.OLS(data[robot_name], sm.add_constant(data[chips])).fit()
    spread = model.resid
    adf_p = adfuller(spread, autolag='AIC')[1]

    intercept = float(model.params['const'])
    betas = {chip: float(model.params[chip]) for chip in chips}

    print(f"\n{'-' * 70}")
    print(f"{robot_name} ~ {chips}")
    print(f"  R²  = {model.rsquared:.6f}")
    print(f"  ADF = {adf_p:.6f}  ({'stationary' if adf_p < 0.05 else 'non-stationary'})")
    print(f"  std = {spread.std():.4f}")
    print(f"  intercept: {intercept:.4f}")
    for chip, beta in betas.items():
        print(f"  beta {chip}: {beta:.4f}")

    beta_oval = betas.get('MICROCHIP_OVAL', 0.0)
    beta_square = betas.get('MICROCHIP_SQUARE', 0.0)
    print(f"\n  '{robot_name}': ({intercept:.2f}, {beta_oval:.4f}, {beta_square:.4f}, {spread.std():.2f}),")


if __name__ == '__main__':
    try:
        prices_df = load_prices(DATA_DIR, max_days=2)
        robots = [
            'ROBOT_VACUUMING',
            'ROBOT_MOPPING',
            'ROBOT_DISHES',
            'ROBOT_LAUNDRY',
            'ROBOT_IRONING',
        ]
        chips = ['MICROCHIP_OVAL', 'MICROCHIP_SQUARE']
        for robot in robots:
            analyze_robot_vs_chips(robot, chips, prices_df)
    except Exception as e:
        print(f"Error: {e}")

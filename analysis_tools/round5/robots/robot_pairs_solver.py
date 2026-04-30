import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / 'data' / 'round5'
POS_LIMIT = 7

ROBOTS = [
    'ROBOT_VACUUMING',
    'ROBOT_MOPPING',
    'ROBOT_DISHES',
    'ROBOT_LAUNDRY',
    'ROBOT_IRONING',
]


def load_prices(data_dir, max_days=None):
    files = sorted(Path(data_dir).glob('prices_*.csv'))
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
    return wide.groupby(level='day').ffill().dropna(how='all')


def analyze_pair(p1, p2, prices):
    """OLS spread p1 ~ p2; returns stationarity stats and OU half-life."""
    cols = [p1, p2]
    if any(c not in prices.columns for c in cols):
        return None
    data = prices[cols].dropna()
    if len(data) < 200:
        return None

    Y = data[p1]
    X = sm.add_constant(data[p2])
    model = sm.OLS(Y, X).fit()
    alpha = float(model.params['const'])
    beta = float(model.params[p2])
    spread = Y - (alpha + beta * data[p2])

    adf_p = adfuller(spread, autolag='AIC')[1]
    coint_p = coint(Y, data[p2])[1]

    s_lag = spread.shift(1).dropna()
    s_diff = spread.diff().dropna()
    common = s_lag.index.intersection(s_diff.index)
    half_life = float('inf')
    if len(common) > 50:
        ols_hl = sm.OLS(s_diff.loc[common], sm.add_constant(s_lag.loc[common])).fit()
        theta = -ols_hl.params.iloc[1]
        if theta > 0:
            half_life = float(np.log(2) / theta)

    return {
        'p1': p1, 'p2': p2,
        'alpha': alpha, 'beta': beta,
        'spread_mean': float(spread.mean()),
        'spread_std': float(spread.std()),
        'r2': float(model.rsquared),
        'adf_p': float(adf_p),
        'coint_p': float(coint_p),
        'half_life': half_life,
    }


def analyze_basket(target, basket, prices):
    """OLS of target ~ basket of other assets; returns spread stationarity stats."""
    cols = [target] + basket
    if any(c not in prices.columns for c in cols):
        return None
    data = prices[cols].dropna()
    if len(data) < 200:
        return None
    model = sm.OLS(data[target], sm.add_constant(data[basket])).fit()
    spread = model.resid
    adf_p = adfuller(spread, autolag='AIC')[1]
    s_lag = spread.shift(1).dropna()
    s_diff = spread.diff().dropna()
    common = s_lag.index.intersection(s_diff.index)
    half_life = float('inf')
    if len(common) > 50:
        ols_hl = sm.OLS(s_diff.loc[common], sm.add_constant(s_lag.loc[common])).fit()
        theta = -ols_hl.params.iloc[1]
        if theta > 0:
            half_life = float(np.log(2) / theta)
    return {
        'target': target,
        'basket': basket,
        'alpha': float(model.params['const']),
        'betas': {c: float(model.params[c]) for c in basket},
        'spread_std': float(spread.std()),
        'r2': float(model.rsquared),
        'adf_p': float(adf_p),
        'half_life': half_life,
    }


def main():
    prices = load_prices(DATA_DIR, max_days=3)

    print('\n>>> ROBOT vs ROBOT PAIRS (3 days)\n')
    results = []
    for p1, p2 in combinations(ROBOTS, 2):
        r = analyze_pair(p1, p2, prices)
        if r is not None:
            results.append(r)
    results.sort(key=lambda x: x['adf_p'])

    print('\n>>> ROBOT vs BASKET (other robots) – 3 days\n')
    basket_results = []
    for target in ROBOTS:
        others = [r for r in ROBOTS if r != target]
        b = analyze_basket(target, others, prices)
        if b is not None:
            basket_results.append(b)
    basket_results.sort(key=lambda x: x['adf_p'])
    print(f"{'TARGET':<20} {'R2':>6} {'ADF_p':>8} {'STD':>8} {'HL':>8}")
    for b in basket_results:
        hl = f"{b['half_life']:.0f}" if np.isfinite(b['half_life']) else 'inf'
        print(f"{b['target']:<20} {b['r2']:>6.3f} {b['adf_p']:>8.4f} "
              f"{b['spread_std']:>8.2f} {hl:>8}")
        for k, v in b['betas'].items():
            print(f"    beta {k}: {v:+.4f}")
        print(f"    alpha: {b['alpha']:.4f}")

    print('\n>>> ROBOT vs ROBOTS+MICROCHIPS – 3 days\n')
    full_basket = ROBOTS + ['MICROCHIP_OVAL', 'MICROCHIP_SQUARE']
    for target in ROBOTS:
        others = [c for c in full_basket if c != target]
        b = analyze_basket(target, others, prices)
        if b is None:
            continue
        hl = f"{b['half_life']:.0f}" if np.isfinite(b['half_life']) else 'inf'
        print(f"\n{target}: R2={b['r2']:.3f} ADF_p={b['adf_p']:.4f} STD={b['spread_std']:.2f} HL={hl}")
        for k, v in b['betas'].items():
            print(f"    beta {k}: {v:+.4f}")
        print(f"    alpha: {b['alpha']:.4f}")

    print(f"\n{'=' * 100}")
    print(f"{'PAIR':<45} {'R2':>6} {'ADF_p':>8} {'COINT_p':>8} {'STD':>8} {'HL':>8} {'BETA':>7}")
    print(f"{'=' * 100}")
    for r in results:
        pair = f"{r['p1']} ~ {r['p2']}"
        hl_str = f"{r['half_life']:.0f}" if np.isfinite(r['half_life']) else 'inf'
        print(f"{pair:<45} {r['r2']:>6.3f} {r['adf_p']:>8.4f} {r['coint_p']:>8.4f} "
              f"{r['spread_std']:>8.2f} {hl_str:>8} {r['beta']:>7.3f}")

    print(f"\n{'=' * 100}")
    print("BEST STATIONARY PAIRS (adf_p < 0.05) — config for trader:")
    print(f"{'=' * 100}")
    for r in results:
        if r['adf_p'] < 0.05:
            print(f"# {r['p1']} ~ {r['p2']}: HL={r['half_life']:.0f}, std={r['spread_std']:.2f}")
            print(f"('{r['p1']}', '{r['p2']}', "
                  f"{r['alpha']:.4f}, {r['beta']:.4f}, "
                  f"{r['spread_mean']:.4f}, {r['spread_std']:.4f}),")


if __name__ == '__main__':
    main()

"""
Panel pair / basket / OU solver.

Goals:
  1. Single-asset mean reversion test (especially PANEL_4X4).
  2. All pairwise OLS spreads (PANEL_i / PANEL_j).
  3. Structural baskets:
       - Vertical (narrow) vs wider:  1X2 + 1X4   vs  2X2 + 2X4
       - "x2" group vs rest:           1X2 + 2X2   vs  1X4 + 2X4 + 4X4
       - 4X4 vs lower-order panels (replication candidates).
  4. Fit Ornstein-Uhlenbeck to every spread:
        d(spread) = -theta (spread - mu) dt + sigma dW
     report half-life, mu, sigma, ADF p-value.

Output: pretty table on stdout + machine-readable file
   analysis_tools/round5/panel/pairs_output.txt
"""

import sys
import math
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "round5"
OUT_FILE = REPO_ROOT / "analysis_tools" / "round5" / "panel" / "pairs_output.txt"

PANELS = [
    'PANEL_1X2',
    'PANEL_1X4',
    'PANEL_2X2',
    'PANEL_2X4',
    'PANEL_4X4',
]


def load_prices(data_dir, max_days=None):
    files = sorted(Path(data_dir).glob("prices_*.csv"))
    if max_days is not None:
        files = files[:max_days]
    df_list = [pd.read_csv(f, sep=';') for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    wide = df.pivot_table(index=['day', 'timestamp'], columns='product',
                          values='mid_price', aggfunc='last')
    return wide.groupby(level='day').ffill().dropna(how='all')


def fit_ou(spread):
    """OU half-life, mu, sigma via OLS on d(spread) ~ a + b * spread_lag."""
    s_lag = spread.shift(1).dropna()
    s_diff = spread.diff().dropna()
    common = s_lag.index.intersection(s_diff.index)
    if len(common) < 100:
        return {'half_life': float('inf'), 'mu': float(spread.mean()),
                'theta': 0.0, 'sigma': float(spread.std())}
    X = sm.add_constant(s_lag.loc[common])
    res = sm.OLS(s_diff.loc[common], X).fit()
    a = float(res.params.iloc[0])
    b = float(res.params.iloc[1])
    theta = -b
    mu = a / theta if theta != 0 else float(spread.mean())
    half_life = float(np.log(2) / theta) if theta > 1e-9 else float('inf')
    sigma = float(np.std(res.resid))
    return {'half_life': half_life, 'mu': mu, 'theta': theta, 'sigma': sigma}


def hurst(ts, lags=range(2, 80)):
    ts = np.asarray(ts, dtype=float)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


def analyze_single(prod, prices):
    if prod not in prices.columns:
        return None
    s = prices[prod].dropna()
    if len(s) < 200:
        return None
    adf_p = float(adfuller(s, autolag='AIC')[1])
    ou = fit_ou(s)
    h = hurst(s.values)
    return {'name': prod, 'mean': float(s.mean()), 'std': float(s.std()),
            'adf_p': adf_p, 'half_life': ou['half_life'], 'hurst': h}


def analyze_spread(name, spread):
    spread = spread.dropna()
    if len(spread) < 200:
        return None
    adf_p = float(adfuller(spread, autolag='AIC')[1])
    ou = fit_ou(spread)
    return {
        'name': name,
        'mean': float(spread.mean()),
        'std': float(spread.std()),
        'adf_p': adf_p,
        'half_life': ou['half_life'],
        'theta': ou['theta'],
        'mu_ou': ou['mu'],
        'sigma_ou': ou['sigma'],
    }


def analyze_pair(p1, p2, prices):
    cols = [p1, p2]
    if any(c not in prices.columns for c in cols):
        return None
    data = prices[cols].dropna()
    if len(data) < 200:
        return None
    Y = data[p1]; X = sm.add_constant(data[p2])
    model = sm.OLS(Y, X).fit()
    alpha = float(model.params['const']); beta = float(model.params[p2])
    spread = Y - (alpha + beta * data[p2])
    coint_p = float(coint(Y, data[p2])[1])
    out = analyze_spread(f"{p1} - ({alpha:+.2f} + {beta:+.4f} * {p2})", spread)
    if out is None:
        return None
    out.update({'p1': p1, 'p2': p2, 'alpha': alpha, 'beta': beta,
                'r2': float(model.rsquared), 'coint_p': coint_p,
                'simple_diff_std': float((Y - data[p2]).std())})
    return out


def analyze_basket(name, weights, prices):
    """weights: dict {product: weight}. Builds spread = sum_i w_i * mid_i."""
    cols = list(weights.keys())
    if any(c not in prices.columns for c in cols):
        return None
    data = prices[cols].dropna()
    if len(data) < 200:
        return None
    spread = sum(w * data[c] for c, w in weights.items())
    return analyze_spread(name, spread)


def fmt(x, kind='f'):
    if isinstance(x, float):
        if not np.isfinite(x):
            return 'inf'
        if kind == 'p':
            return f"{x:.4f}"
        if kind == '0':
            return f"{x:.0f}"
        return f"{x:.3f}"
    return str(x)


def main():
    prices = load_prices(DATA_DIR)
    print(f"Loaded {len(prices)} ticks across days {sorted(prices.index.get_level_values('day').unique())}")

    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    # ----- 1. Single-asset MR check -----
    emit("\n" + "=" * 92)
    emit("SINGLE-ASSET MEAN REVERSION TEST")
    emit("=" * 92)
    emit(f"{'PANEL':<18} {'MEAN':>10} {'STD':>8} {'ADF_p':>8} {'HALF_LIFE':>10} {'HURST':>7}")
    singles = []
    for p in PANELS:
        r = analyze_single(p, prices)
        if r is None:
            continue
        singles.append(r)
        emit(f"{r['name']:<18} {r['mean']:>10.2f} {r['std']:>8.2f} "
             f"{r['adf_p']:>8.4f} {fmt(r['half_life'], '0'):>10} {r['hurst']:>7.3f}")
    emit("(ADF_p<0.05 -> stationary;  Hurst<0.5 -> mean-reverting;  short HL = fast reversion)")

    # ----- 2. Pairs (OLS hedge) -----
    emit("\n" + "=" * 92)
    emit("PAIRS — OLS hedge ratio  spread = p1 - (alpha + beta * p2)")
    emit("=" * 92)
    emit(f"{'PAIR':<32} {'R2':>5} {'ADF':>7} {'COINT':>7} {'STD':>8} {'HL':>8} {'BETA':>8}")
    pair_results = []
    for p1, p2 in combinations(PANELS, 2):
        r = analyze_pair(p1, p2, prices)
        if r is None:
            continue
        pair_results.append(r)
    pair_results.sort(key=lambda x: x['adf_p'])
    for r in pair_results:
        hl = fmt(r['half_life'], '0')
        emit(f"{r['p1']+' / '+r['p2']:<32} {r['r2']:>5.2f} {r['adf_p']:>7.4f} "
             f"{r['coint_p']:>7.4f} {r['std']:>8.2f} {hl:>8} {r['beta']:>8.4f}")

    # ----- 3. Structural baskets -----
    emit("\n" + "=" * 92)
    emit("STRUCTURAL BASKETS (fixed unit weights — replication candidates)")
    emit("=" * 92)
    structural = [
        # User's groupings
        ("VERT (1X2+1X4) - WIDE (2X2+2X4)",
         {'PANEL_1X2': 1, 'PANEL_1X4': 1, 'PANEL_2X2': -1, 'PANEL_2X4': -1}),
        ("X2_GROUP (1X2+2X2) - REST (1X4+2X4+4X4)",
         {'PANEL_1X2': 1, 'PANEL_2X2': 1, 'PANEL_1X4': -1, 'PANEL_2X4': -1, 'PANEL_4X4': -1}),

        # Replication candidates by cell-count
        ("4X4 - 2*2X4",
         {'PANEL_4X4': 1, 'PANEL_2X4': -2}),
        ("4X4 - 4*2X2",
         {'PANEL_4X4': 1, 'PANEL_2X2': -4}),
        ("4X4 - 4*1X4",
         {'PANEL_4X4': 1, 'PANEL_1X4': -4}),
        ("2X4 - 2*1X4",
         {'PANEL_2X4': 1, 'PANEL_1X4': -2}),
        ("2X4 - 4*1X2",
         {'PANEL_2X4': 1, 'PANEL_1X2': -4}),
        ("2X2 - 2*1X2",
         {'PANEL_2X2': 1, 'PANEL_1X2': -2}),
        ("2X2 - 1X4",
         {'PANEL_2X2': 1, 'PANEL_1X4': -1}),
        ("4X4 - (2X4+2X2+1X4+1X2)",
         {'PANEL_4X4': 1, 'PANEL_2X4': -1, 'PANEL_2X2': -1, 'PANEL_1X4': -1, 'PANEL_1X2': -1}),
        # User's "x2 + 2x2 vs rest" alternative reading: 1X2 + 2X2 vs the rest with cell-weights
        ("CELLS  (1X2+2X2)*3 - (1X4+2X4+4X4)",
         {'PANEL_1X2': 3, 'PANEL_2X2': 3, 'PANEL_1X4': -1, 'PANEL_2X4': -1, 'PANEL_4X4': -1}),
    ]
    emit(f"{'BASKET':<46} {'MEAN':>10} {'STD':>8} {'ADF_p':>8} {'HL':>8}")
    basket_results = []
    for name, weights in structural:
        r = analyze_basket(name, weights, prices)
        if r is None:
            continue
        basket_results.append((name, weights, r))
        emit(f"{name:<46} {r['mean']:>10.2f} {r['std']:>8.2f} "
             f"{r['adf_p']:>8.4f} {fmt(r['half_life'], '0'):>8}")

    # ----- 4. OLS basket: target ~ rest -----
    emit("\n" + "=" * 92)
    emit("PANEL vs BASKET-of-others (OLS, all panels)")
    emit("=" * 92)
    emit(f"{'TARGET':<14} {'R2':>5} {'ADF_p':>8} {'STD':>8} {'HL':>8}    BETAS")
    for target in PANELS:
        others = [p for p in PANELS if p != target]
        cols = [target] + others
        if any(c not in prices.columns for c in cols):
            continue
        data = prices[cols].dropna()
        if len(data) < 200:
            continue
        Y = data[target]; X = sm.add_constant(data[others])
        m = sm.OLS(Y, X).fit()
        spread = m.resid
        adf_p = float(adfuller(spread, autolag='AIC')[1])
        ou = fit_ou(spread)
        betas_str = ', '.join(f"{c}:{m.params[c]:+.3f}" for c in others)
        hl = fmt(ou['half_life'], '0')
        emit(f"{target:<14} {m.rsquared:>5.2f} {adf_p:>8.4f} {float(spread.std()):>8.2f} "
             f"{hl:>8}    a={float(m.params['const']):+.2f}  {betas_str}")

    # ----- 5. Picks -----
    emit("\n" + "=" * 92)
    emit("BEST CANDIDATES (ranked by ADF_p among stationary)")
    emit("=" * 92)
    all_candidates = []
    for r in pair_results:
        all_candidates.append(('PAIR', r['p1'] + ' / ' + r['p2'], r['adf_p'],
                               r['half_life'], r['std'], r))
    for name, weights, r in basket_results:
        all_candidates.append(('BASKET', name, r['adf_p'], r['half_life'], r['std'],
                               {**r, 'weights': weights}))
    all_candidates.sort(key=lambda x: (x[2], x[3]))
    for kind, name, adfp, hl, std, _ in all_candidates[:15]:
        emit(f"  {kind:<7} {name:<46} adf={adfp:.4f} HL={fmt(hl, '0')} std={std:.2f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(out_lines))
    print(f"\nWritten {OUT_FILE}")


if __name__ == "__main__":
    main()

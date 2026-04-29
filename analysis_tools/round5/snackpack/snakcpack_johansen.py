import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR  = REPO_ROOT / 'data' / 'round5'

def half_life(spread: np.ndarray) -> float:
    s     = pd.Series(spread)
    delta = s.diff().dropna()
    lag   = s.shift(1).dropna()
    model = OLS(delta, add_constant(lag)).fit()
    lam   = model.params.iloc[1]
    return -np.log(2) / lam if lam < 0 else float('inf')

def load_prices(data_dir):
    files = sorted(Path(data_dir).glob("prices_*.csv"))
    df    = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    df    = df.sort_values(['day', 'timestamp'])
    df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    wide  = df.pivot_table(index=['day','timestamp'], columns='product',
                           values='mid_price', aggfunc='last')
    return wide.groupby(level='day').ffill().dropna(how='all')

def analyze_basket(name, group, prices):
    print(f"\n{'═'*60}")
    print(f"BASKET: {name}  →  {group}")
    print(f"{'═'*60}")
    data = prices[group].dropna()

    # ── Johansen ──────────────────────────────────────────────────────────
    jres    = coint_johansen(data, det_order=0, k_ar_diff=5)
    traces  = jres.lr1
    cvs     = jres.cvt[:, 1]   # 95%
    n_coint = int((traces > cvs).sum())
    print(f"\nJohansen — cointegrating vectors at 5%: {n_coint}")
    for i, (tr, cv) in enumerate(zip(traces, cvs)):
        print(f"  r≤{i}: trace={tr:.2f}  cv={cv:.2f}  {'✓' if tr>cv else '✗'}")

    # ── OLS in ogni direzione → trova il target migliore ─────────────────
    print("\nOLS spread analysis (ogni prodotto come target):")
    best = {'r2': -1}
    for target in group:
        others = [p for p in group if p != target]
        model  = OLS(data[target], add_constant(data[others])).fit()
        spread = model.resid
        adf_p  = adfuller(spread, autolag='AIC')[1]
        hl     = half_life(spread.values)
        coeffs = dict(zip(others, model.params[1:].round(6)))
        print(f"\n  {target} ~ {others}")
        print(f"    coeffs : {coeffs}")
        print(f"    R²     : {model.rsquared:.6f}")
        print(f"    ADF p  : {adf_p:.6f}")
        print(f"    HL     : {hl:.1f} ticks")

        if model.rsquared > best['r2']:
            best = {
                'r2': model.rsquared, 'target': target,
                'weights': coeffs, 'intercept': round(model.params.iloc[0], 4),
                'adf_p': adf_p, 'hl': hl, 'spread_std': round(spread.std(), 4)
            }

    # ── Lot sizing ────────────────────────────────────────────────────────
    POS_LIMIT = 10
    max_w  = max(abs(v) for v in best['weights'].values())
    k      = int(POS_LIMIT / max_w)
    lots   = {best['target']: k}
    lots  |= {p: max(1, round(abs(w) * k)) for p, w in best['weights'].items()}

    print(f"\n{'─'*60}")
    print(f"BEST TARGET : {best['target']}")
    print(f"WEIGHTS     : {best['weights']}")
    print(f"R²={best['r2']:.6f} | ADF p={best['adf_p']:.6f} | HL={best['hl']:.1f}t")
    print(f"\nLOT SIZING (k={k}, POS_LIMIT={POS_LIMIT}):")
    print(f"  lots = {lots}")
    print(f"\n── Incolla nel registry ──────────────────────────────────────")
    print(f"'{name}': {{")
    print(f"    'target' : '{best['target']}',")
    print(f"    'weights': {best['weights']},")
    print(f"    'lots'   : {lots},")
    print(f"}},")
    return best

if __name__ == "__main__":
    prices = load_prices(DATA_DIR)

    analyze_basket('SNACKPACK_DUO',
                   ['SNACKPACK_CHOCOLATE', 'SNACKPACK_VANILLA'],
                   prices)

    analyze_basket('SNACKPACK_TRIO',
                   ['SNACKPACK_PISTACHIO', 'SNACKPACK_RASPBERRY', 'SNACKPACK_STRAWBERRY'],
                   prices)
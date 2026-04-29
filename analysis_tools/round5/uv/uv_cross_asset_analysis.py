"""
Cross-Asset Pattern Discovery
==============================
Pipeline:
  1. Load all price CSV files → pivot to wide mid-price matrix
  2. Return correlation matrix (Pearson + Spearman for robustness)
  3. Engle-Granger cointegration scan on all pairs → ranked by p-value
  4. Johansen test on top clusters (> 2 legs)
  5. OLS residual analysis → spread stationarity (ADF)
  6. PCA decomposition → identify synthetic factors
  7. Rolling correlation heatmap → detect regime changes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import combinations

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─── Config ───────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[3]
DATA_DIR   = REPO_ROOT / 'data' / 'round5'
COINT_PVAL = 0.05    # significance threshold for cointegration
MIN_OBS    = 500     # minimum overlapping observations to test a pair
ROLL_WIN   = 200     # rolling window for correlation regime analysis


# ══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_mid_prices(data_dir: Path) -> pd.DataFrame:
    """
    Returns a wide DataFrame: index = (day, timestamp), columns = products,
    values = mid_price. Mid = (best_bid + best_ask) / 2.
    """
    files = sorted(data_dir.glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_*.csv in {data_dir.resolve()}")

    frames = [pd.read_csv(f, sep=';') for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(['day', 'timestamp'])
    df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    df = df.dropna(subset=['mid_price'])

    wide = df.pivot_table(
        index=['day', 'timestamp'],
        columns='product',
        values='mid_price',
        aggfunc='last'
    )
    print(f"Loaded {len(wide)} ticks | {wide.shape[1]} products: {list(wide.columns)}")
    return wide


# ══════════════════════════════════════════════════════════════════════════
# 2. RETURN CORRELATION
# ══════════════════════════════════════════════════════════════════════════

def correlation_analysis(prices: pd.DataFrame):
    """Pearson + Spearman on log-returns. Returns both matrices."""
    rets = np.log(prices / prices.shift(1)).dropna()

    pearson  = rets.corr(method='pearson')
    spearman = rets.corr(method='spearman')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, corr, title in zip(axes, [pearson, spearman], ['Pearson', 'Spearman']):
        im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title(f'{title} Return Correlation', fontsize=12)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=6,
                        color='black' if abs(corr.iloc[i, j]) < 0.7 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    return pearson, spearman


# ══════════════════════════════════════════════════════════════════════════
# 3. ENGLE-GRANGER PAIRWISE COINTEGRATION SCAN
# ══════════════════════════════════════════════════════════════════════════

def cointegration_scan(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Tests all pairs for cointegration (both directions: y~x and x~y).
    Returns a ranked DataFrame of cointegrated pairs.
    """
    products = prices.columns.tolist()
    results  = []

    for p1, p2 in combinations(products, 2):
        series  = prices[[p1, p2]].dropna()
        if len(series) < MIN_OBS:
            continue
        s1, s2 = series[p1].values, series[p2].values

        # Test both directions — OLS hedge ratio depends on which is y
        for y, x, yn, xn in [(s1, s2, p1, p2), (s2, s1, p2, p1)]:
            try:
                score, pval, _ = coint(y, x)
                # OLS hedge ratio
                model  = OLS(y, add_constant(x)).fit()
                hedge  = model.params[1]
                spread = y - hedge * x - model.params[0]
                adf_p  = adfuller(spread, maxlags=10, autolag='AIC')[1]

                results.append({
                    'y': yn, 'x': xn,
                    'coint_pval': round(pval, 4),
                    'adf_pval':   round(adf_p, 4),
                    'hedge_ratio': round(hedge, 4),
                    'spread_mean': round(spread.mean(), 4),
                    'spread_std':  round(spread.std(), 4),
                    'half_life':   round(_half_life(spread), 1),
                    'n_obs':       len(series),
                })
            except Exception:
                continue

    df = pd.DataFrame(results)
    if df.empty:
        print("No cointegrated pairs found.")
        return df

    # Rank: cointegrated AND spread is stationary
    df['is_coint'] = (df['coint_pval'] < COINT_PVAL) & (df['adf_pval'] < COINT_PVAL)
    df = df.sort_values(['is_coint', 'coint_pval'], ascending=[False, True])

    print("\n── Top Cointegrated Pairs ──────────────────────────────────")
    print(df[df['is_coint']].to_string(index=False))
    return df


def _half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion."""
    s     = pd.Series(spread)
    lag   = s.shift(1).dropna()
    delta = s.diff().dropna()
    model = OLS(delta, add_constant(lag)).fit()
    lam   = model.params.iloc[1]
    if lam >= 0:
        return float('inf')
    return -np.log(2) / lam


# ══════════════════════════════════════════════════════════════════════════
# 4. JOHANSEN TEST (multi-leg baskets)
# ══════════════════════════════════════════════════════════════════════════

def johansen_test(prices: pd.DataFrame, product_group: list, max_lags: int = 5):
    """
    Johansen cointegration test on a group of products.
    Returns the cointegrating vectors (eigenvectors) if any are found.
    """
    data = prices[product_group].dropna()
    if len(data) < MIN_OBS:
        print(f"Not enough data for Johansen on {product_group}")
        return None

    result = coint_johansen(data, det_order=0, k_ar_diff=max_lags)

    # Compare trace stats with 95% critical values
    traces  = result.lr1
    cvs_95  = result.cvt[:, 1]
    n_coint = int((traces > cvs_95).sum())

    print(f"\n── Johansen: {product_group} ─────────────────────────────")
    print(f"  Cointegrating vectors at 5%: {n_coint}")
    for i, (tr, cv) in enumerate(zip(traces, cvs_95)):
        sig = "✓" if tr > cv else "✗"
        print(f"  r≤{i}: trace={tr:.2f}, cv={cv:.2f} {sig}")

    if n_coint > 0:
        print("  Cointegrating vectors (columns):")
        vecs = pd.DataFrame(result.evec[:, :n_coint],
                            index=product_group,
                            columns=[f'v{i}' for i in range(n_coint)])
        print(vecs.round(4))
        return vecs
    return None


# ══════════════════════════════════════════════════════════════════════════
# 5. SPREAD DIAGNOSTICS (ADF + half-life visualization)
# ══════════════════════════════════════════════════════════════════════════

def spread_diagnostics(prices: pd.DataFrame, pairs: pd.DataFrame, top_n: int = 4):
    """
    For the top N cointegrated pairs: plot spread + z-score + half-life.
    """
    top = pairs[pairs['is_coint']].head(top_n)
    if top.empty:
        print("No significant pairs to plot.")
        return

    fig, axes = plt.subplots(top_n, 2, figsize=(16, 4 * top_n))
    if top_n == 1:
        axes = axes[np.newaxis, :]

    for i, row in enumerate(top.itertuples()):
        data   = prices[[row.y, row.x]].dropna()
        y, x   = data[row.y].values, data[row.x].values
        model  = OLS(y, add_constant(x)).fit()
        spread = y - model.params[1] * x - model.params[0]
        z      = (spread - spread.mean()) / spread.std()
        idx    = np.arange(len(spread))

        # Spread
        axes[i, 0].plot(idx, spread, lw=0.8, color='steelblue')
        axes[i, 0].axhline(spread.mean(), color='red', ls='--', lw=1)
        axes[i, 0].set_title(
            f'{row.y} − {row.hedge_ratio}×{row.x} | '
            f'HL={row.half_life:.1f}t | ADF p={row.adf_pval:.4f}',
            fontsize=9
        )
        axes[i, 0].set_ylabel('Spread')
        axes[i, 0].grid(alpha=0.3)

        # Z-score
        axes[i, 1].plot(idx, z, lw=0.8, color='darkorange')
        axes[i, 1].axhline(2,  color='red', ls='--', lw=1, label='+2σ')
        axes[i, 1].axhline(-2, color='green', ls='--', lw=1, label='−2σ')
        axes[i, 1].axhline(0,  color='black', ls='-', lw=0.5)
        axes[i, 1].set_title(f'Z-score | {row.n_obs} obs', fontsize=9)
        axes[i, 1].legend(fontsize=8)
        axes[i, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('spread_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 6. PCA — SYNTHETIC FACTORS
# ══════════════════════════════════════════════════════════════════════════

def pca_factors(prices: pd.DataFrame, n_components: int = 5):
    """
    PCA on log-returns. Identifies synthetic market factors and
    shows which products load heavily on each factor.
    """
    rets  = np.log(prices / prices.shift(1)).dropna()
    scaler = StandardScaler()
    X      = scaler.fit_transform(rets.values)

    n_components = min(n_components, rets.shape[1])
    pca    = PCA(n_components=n_components)
    pca.fit(X)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=prices.columns,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    expl_var = pca.explained_variance_ratio_

    print("\n── PCA Explained Variance ───────────────────────────────────")
    for i, v in enumerate(expl_var):
        print(f"  PC{i+1}: {v:.1%} ({expl_var[:i+1].sum():.1%} cumulative)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scree plot
    axes[0].bar(range(1, n_components + 1), expl_var * 100, color='steelblue')
    axes[0].plot(range(1, n_components + 1), np.cumsum(expl_var) * 100,
                 'ro-', label='Cumulative')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Scree Plot')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loadings heatmap
    im = axes[1].imshow(loadings.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_xticks(range(n_components))
    axes[1].set_xticklabels(loadings.columns)
    axes[1].set_yticks(range(len(loadings.index)))
    axes[1].set_yticklabels(loadings.index, fontsize=8)
    axes[1].set_title('PCA Loadings (product × factor)')
    for i in range(len(loadings.index)):
        for j in range(n_components):
            axes[1].text(j, i, f'{loadings.iloc[i, j]:.2f}',
                         ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig('pca_factors.png', dpi=150, bbox_inches='tight')
    plt.show()

    return loadings, expl_var


# ══════════════════════════════════════════════════════════════════════════
# 7. ROLLING CORRELATION — REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════

def rolling_correlation(prices: pd.DataFrame, pairs: pd.DataFrame,
                        window: int = ROLL_WIN, top_n: int = 4):
    """
    Rolling Pearson correlation for top pairs.
    Breaks in correlation = regime change = pairs trade blows up here.
    """
    top   = pairs[pairs['is_coint']].head(top_n)
    rets  = np.log(prices / prices.shift(1)).dropna()

    if top.empty:
        return

    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3 * top_n), sharex=False)
    if top_n == 1:
        axes = [axes]

    for i, row in enumerate(top.itertuples()):
        if row.y not in rets.columns or row.x not in rets.columns:
            continue
        roll_corr = rets[row.y].rolling(window).corr(rets[row.x]).dropna()
        axes[i].plot(roll_corr.values, lw=0.9, color='purple')
        axes[i].axhline(roll_corr.mean(), color='black', ls='--', lw=0.8)
        axes[i].axhline(0, color='gray', ls='-', lw=0.5)
        axes[i].fill_between(range(len(roll_corr)), roll_corr.values,
                             roll_corr.mean(), alpha=0.2, color='purple')
        axes[i].set_title(f'Rolling {window}-tick corr: {row.y} vs {row.x}', fontsize=9)
        axes[i].set_ylim(-1.1, 1.1)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('rolling_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 8. AUTO-DETECT UV_VISOR CLUSTER + JOHANSEN
# ══════════════════════════════════════════════════════════════════════════

def detect_clusters(pearson: pd.DataFrame, threshold: float = 0.7) -> dict:
    """
    Simple single-linkage clustering on return correlation.
    Identifies groups of highly correlated assets (potential basket candidates).
    """
    clusters = {}
    products = list(pearson.columns)
    visited  = set()

    for p in products:
        if p in visited:
            continue
        cluster = [p]
        for q in products:
            if q != p and q not in visited:
                if abs(pearson.loc[p, q]) >= threshold:
                    cluster.append(q)
                    visited.add(q)
        visited.add(p)
        if len(cluster) > 1:
            key = cluster[0]
            clusters[key] = cluster

    print("\n── Detected Clusters (|corr| ≥ {:.0%}) ──────────────────────".format(threshold))
    for k, v in clusters.items():
        print(f"  {k}: {v}")
    return clusters


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    prices = load_mid_prices(DATA_DIR)

    # Fill forward within each day (sparse ticks for illiquid products)
    prices = prices.groupby(level='day').ffill().dropna(how='all')

    # ── Step 1: Correlation ───────────────────────────────────────────────
    pearson, spearman = correlation_analysis(prices)

    # ── Step 2: Cluster detection ─────────────────────────────────────────
    clusters = detect_clusters(pearson, threshold=0.7)

    # ── Step 3: Johansen on each cluster ──────────────────────────────────
    johansen_vectors = {}
    for name, group in clusters.items():
        vecs = johansen_test(prices, group)
        if vecs is not None:
            johansen_vectors[name] = vecs

    # ── Step 4: Pairwise cointegration scan ───────────────────────────────
    coint_df = cointegration_scan(prices)

    # ── Step 5: Spread diagnostics on top pairs ───────────────────────────
    if not coint_df.empty:
        spread_diagnostics(prices, coint_df, top_n=min(4, coint_df['is_coint'].sum()))

    # ── Step 6: PCA ───────────────────────────────────────────────────────
    loadings, expl_var = pca_factors(prices, n_components=min(6, prices.shape[1]))

    # ── Step 7: Rolling correlation on top pairs ──────────────────────────
    if not coint_df.empty:
        rolling_correlation(prices, coint_df, window=ROLL_WIN, top_n=4)

    # ── Summary table ─────────────────────────────────────────────────────
    if not coint_df.empty:
        print("\n═══════════════════════════════════════════════════════════")
        print("TRADEABLE PAIRS SUMMARY (coint p<0.05 AND ADF p<0.05)")
        print("═══════════════════════════════════════════════════════════")
        cols = ['y', 'x', 'hedge_ratio', 'half_life', 'coint_pval',
                'adf_pval', 'spread_std', 'n_obs']
        print(coint_df[coint_df['is_coint']][cols].to_string(index=False))
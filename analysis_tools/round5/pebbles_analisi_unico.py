import pandas as pd
import numpy as np
from pathlib import Path
import glob
from statsmodels.tsa.stattools import adfuller

# ==============================================================================
# CONFIGURAZIONE UNIFICATA PEBBLES
# ==============================================================================
PEBBLES = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']
ORACLE_MAP = {
    "PEBBLES_XS": "UV_VISOR_AMBER",
    "PEBBLES_S": "GALAXY_SOUNDS_BLACK_HOLES",
    "PEBBLES_M": "OXYGEN_SHAKE_MORNING_BREATH",
    "PEBBLES_L": "TRANSLATOR_GRAPHITE_MIST",
    "PEBBLES_XL": "PANEL_2X4"
}

def load_data(data_dir: Path):
    print(f"📡 Caricamento dati in corso...")
    files = sorted(glob.glob(str(data_dir / "prices_round_5_day_*.csv")))
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f, sep=";")
        temp_df.columns = [c.strip().lower() for c in temp_df.columns]
        df_list.append(temp_df)
    return pd.concat(df_list, ignore_index=True).sort_values(by=['day', 'timestamp'])

def run_unified_analysis():
    DATA_PATH = Path("data/round5")
    df_all = load_data(DATA_PATH)
    
    # Pivot dei prezzi necessari
    all_needed = PEBBLES + list(ORACLE_MAP.values())
    pivot = df_all[df_all['product'].isin(all_needed)].pivot_table(
        index=['day', 'timestamp'], columns='product', values='mid_price'
    ).ffill().dropna()

    print("\n" + "=" * 90)
    print("   PEBBLES UNIFIED ANALYZER: COMPLETE STATS AND ARB ANALYSIS")
    print("=" * 90)

    # 1. Verifica Vincolo di Somma
    cluster_sum = pivot[PEBBLES].sum(axis=1)
    print(f"\n[1] VINCOLO DI SOMMA CLUSTER (XS+S+M+L+XL)")
    print(f"  Media Somma: {cluster_sum.mean():.2f} (Target: 50000)")
    print(f"  Deviazione Standard: {cluster_sum.std():.4f}")
    print(f"  Stazionarietà Somma (ADF p-value): {adfuller(cluster_sum)[1]:.4e}")

    # 2. Parametri Oracoli Cross-Cluster (Beta-Adjustment)
    print(f"\n[2] PARAMETRI ORACOLI ESTERNI (Hedge Ratios)")
    print("-" * 60)
    for target, anchor in ORACLE_MAP.items():
        poly = np.polyfit(pivot[anchor], pivot[target], 1)
        beta = poly[0]
        corr = pivot[target].corr(pivot[anchor])
        
        # Calcolo Spread per verifica
        spread = pivot[target] - (beta * pivot[anchor])
        adf_p = adfuller(spread.dropna())[1]
        
        print(f"Asset: {target:12} | Lead: {anchor:30}")
        print(f"  BETA: {beta:8.4f} | Correlazione: {corr:8.4f} | ADF Spread p-value: {adf_p:.4f}")

    # 3. Microstruttura (Esecuzione)
    print(f"\n[3] MICROSTRUTTURA E SPREAD MEDI")
    print("-" * 60)
    df_pebbles = df_all[df_all['product'].isin(PEBBLES)].copy()
    df_pebbles['spread'] = df_pebbles['ask_price_1'] - df_pebbles['bid_price_1']
    ms = df_pebbles.groupby('product')['spread'].mean()
    print(ms.to_string())

    # 4. Basket Arbitrage Analysis (Crossable)
    print(f"\n[4] BASKET ARBITRAGE ANALYSIS (Crossable)")
    print("-" * 60)
    pivot_bid = df_all[df_all['product'].isin(PEBBLES)].pivot_table(
        index=['day', 'timestamp'], columns='product', values='bid_price_1'
    ).ffill().dropna()
    
    pivot_ask = df_all[df_all['product'].isin(PEBBLES)].pivot_table(
        index=['day', 'timestamp'], columns='product', values='ask_price_1'
    ).ffill().dropna()

    sum_bid = pivot_bid.sum(axis=1)
    sum_ask = pivot_ask.sum(axis=1)

    print(f"Total Ticks: {len(sum_bid)}")
    print(f"Ticks where sum(bid) > 50000 (Sell Arb): {len(sum_bid[sum_bid > 50000])}")
    print(f"Ticks where sum(ask) < 50000 (Buy Arb): {len(sum_ask[sum_ask < 50000])}")
    
    if len(sum_bid[sum_bid > 50000]) > 0:
        print(f"Max sum(bid): {sum_bid.max()}")
    if len(sum_ask[sum_ask < 50000]) > 0:
        print(f"Min sum(ask): {sum_ask.min()}")

    # 5. Lead-Lag Relationships
    print(f"\n[5] LEAD-LAG RELATIONSHIPS (XS focus)")
    print("-" * 60)
    print("Correlations with Lagged XS (XS leads others?)")
    for target in PEBBLES:
        if target == 'PEBBLES_XS': continue
        corr_0 = pivot[target].corr(pivot['PEBBLES_XS'])
        corr_1 = pivot[target].corr(pivot['PEBBLES_XS'].shift(1))
        corr_2 = pivot[target].corr(pivot['PEBBLES_XS'].shift(2))
        print(f"XS -> {target:12}: Lag 0: {corr_0:.4f}, Lag 1: {corr_1:.4f}, Lag 2: {corr_2:.4f}")

    print("\nCorrelations with Lead XS (Others lead XS?)")
    for target in PEBBLES:
        if target == 'PEBBLES_XS': continue
        corr_1 = pivot['PEBBLES_XS'].corr(pivot[target].shift(1))
        print(f"{target:12} -> XS: Lag 1: {corr_1:.4f}")

if __name__ == "__main__":
    run_unified_analysis()

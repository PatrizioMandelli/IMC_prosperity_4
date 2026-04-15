import pandas as pd
import numpy as np
import os
import time
from itertools import product

PRODUCT = "INTARIAN_PEPPER_ROOT"


def load_data(data_dir):
    """Carica e unisce trades e prices."""
    print("Caricamento dati...")
    dfs_t, dfs_p = [], []
    for d in [-2, -1, 0]:
        tf = os.path.join(data_dir, f'trades_round_1_day_{d}.csv')
        pf = os.path.join(data_dir, f'prices_round_1_day_{d}.csv')
        if os.path.exists(tf):
            dfs_t.append(pd.read_csv(tf, sep=';').assign(day=d))
        if os.path.exists(pf):
            dfs_p.append(pd.read_csv(pf, sep=';').assign(day=d))

    df_trades = pd.concat(dfs_t).rename(columns={'symbol': 'product'})
    df_prices = pd.concat(dfs_p).sort_values(['day', 'timestamp']).reset_index(drop=True)
    return df_trades, df_prices


def clean_pepper(df_prices):
    """Filtra pepper root, rimuove tick con book vuoto o mid_price anomalo."""
    df = df_prices[df_prices['product'] == PRODUCT].copy()
    n_before = len(df)

    # Rimuovi tick con book vuoto (mid=0 o NaN su bid/ask)
    df = df[df['mid_price'] > 0].copy()
    # Rimuovi tick dove bid o ask sono NaN (one-sided book → mid non affidabile)
    df = df[df['bid_price_1'].notna() & df['ask_price_1'].notna()].copy()

    n_after = len(df)
    print(f"Tick rimossi (book vuoto/one-sided): {n_before - n_after} / {n_before}")

    df['spread'] = df['ask_price_1'] - df['bid_price_1']
    return df.reset_index(drop=True)


# ==========================================
# PARTE 1: ANALISI COMPORTAMENTALE
# ==========================================
def analyze_patterns(df):
    print("\n" + "=" * 50)
    print("=== STEP 1: BEHAVIORAL ANALYSIS (DATI PULITI) ===")
    print("=" * 50)

    # Indicatori base
    df['sma5'] = df.groupby('day')['mid_price'].transform(lambda x: x.rolling(5).mean())
    df['dev'] = df['mid_price'] - df['sma5']
    df['next_ret'] = df.groupby('day')['mid_price'].shift(-1) - df['mid_price']
    df.dropna(subset=['dev', 'next_ret'], inplace=True)

    # 1. Statistiche deviazione
    print(f"\n1. DISTRIBUZIONE DEVIAZIONE (mid - SMA5):")
    print(f"   Mean: {df['dev'].mean():.2f}, Std: {df['dev'].std():.2f}")
    print(f"   Mediana: {df['dev'].median():.2f}")
    for t in [2, 3, 4, 5, 6]:
        pct = (df['dev'].abs() > t).mean() * 100
        print(f"   |dev| > {t}: {pct:.1f}%")

    # 2. Spike reversion per soglia
    print(f"\n2. SPIKE REVERSION PER SOGLIA:")
    for t in [3, 4, 5, 6]:
        spk = df[df['dev'].abs() > t]
        if len(spk) > 5:
            corr = spk['dev'].corr(spk['next_ret'])
            avg_rev = -(spk['dev'] * spk['next_ret']).mean()  # positivo = reversion
            print(f"   |dev| > {t}: n={len(spk)}, corr={corr:.4f}, avg_reversion_pnl={avg_rev:.2f}")

    # 3. Spread medio e impatto
    print(f"\n3. SPREAD:")
    print(f"   Mean: {df['spread'].mean():.1f}, Median: {df['spread'].median():.1f}")
    print(f"   P25: {df['spread'].quantile(0.25):.1f}, P75: {df['spread'].quantile(0.75):.1f}")

    # 4. Spike timing
    df['is_spike'] = df['dev'].abs() > 4
    df['spike_id'] = (df['is_spike'] != df['is_spike'].shift()).cumsum()
    spike_events = df[df['is_spike']].groupby(['day', 'spike_id']).first().reset_index()
    spike_events['gap'] = spike_events.groupby('day')['timestamp'].diff() / 100

    print(f"\n4. TIMING SPIKE (|dev|>4):")
    print(f"   Frequenza media: ogni {spike_events['gap'].mean():.1f} tick")
    print(f"   Frequenza mediana: ogni {spike_events['gap'].median():.1f} tick")
    print(f"   P25: {spike_events['gap'].quantile(0.25):.1f} tick (clusterizzazione)")

    # 5. Reversion multi-tick (quanto dura il revert?)
    print(f"\n5. REVERSION MULTI-TICK (da spike |dev|>4):")
    for horizon in [1, 2, 3, 5]:
        df[f'ret_{horizon}'] = df.groupby('day')['mid_price'].shift(-horizon) - df['mid_price']
        spk = df[df['dev'].abs() > 4]
        corr = spk['dev'].corr(spk[f'ret_{horizon}'])
        print(f"   Orizzonte {horizon} tick: corr={corr:.4f}")

    # 6. Trend strength
    df['fast12'] = df.groupby('day')['mid_price'].transform(lambda x: x.rolling(12).mean())
    df['slow30'] = df.groupby('day')['mid_price'].transform(lambda x: x.rolling(30).mean())
    df['trend'] = df['fast12'] - df['slow30']
    trend_clean = df['trend'].dropna()

    print(f"\n6. TREND (fast12 - slow30):")
    print(f"   Mean: {trend_clean.mean():.2f}, Std: {trend_clean.std():.2f}")
    for t in [0.5, 1.0, 1.5, 2.0]:
        pct = (trend_clean.abs() > t).mean() * 100
        print(f"   |trend| > {t}: {pct:.1f}%")

    # 7. Profittabilità trend semplice
    print(f"\n7. TREND FOLLOW PNL (next-tick):")
    for t in [0.5, 1.0, 1.5, 2.0]:
        long_mask = df['trend'] > t
        short_mask = df['trend'] < -t
        long_pnl = df.loc[long_mask, 'next_ret'].sum()
        short_pnl = -df.loc[short_mask, 'next_ret'].sum()
        n = long_mask.sum() + short_mask.sum()
        print(f"   thresh={t}: pnl={long_pnl + short_pnl:.0f}, trades={n}")

    return df


# ==========================================
# PARTE 2: GRID SEARCH (CORRETTO)
# ==========================================
def simulate_strategy(df, fast_len, slow_len, trend_thresh, spike_thresh, pos_limit=80):
    """Simulazione allineata alla logica reale del trader."""
    df = df.copy()

    df['fast_sma'] = df.groupby('day')['mid_price'].transform(lambda x: x.rolling(fast_len).mean())
    df['slow_sma'] = df.groupby('day')['mid_price'].transform(lambda x: x.rolling(slow_len).mean())
    df['sma5'] = df.groupby('day')['mid_price'].transform(lambda x: x.rolling(5).mean())
    df['deviation'] = df['mid_price'] - df['sma5']
    df['trend_s'] = df['fast_sma'] - df['slow_sma']

    # Logica combinata: trend + spike satellite (come nel trader, non override)
    core = np.where(df['trend_s'] > trend_thresh, 70,
           np.where(df['trend_s'] < -trend_thresh, -70, 0))

    spike_bias = np.where(df['deviation'] > spike_thresh, -30,
                 np.where(df['deviation'] > spike_thresh * 0.6, -15,
                 np.where(df['deviation'] < -spike_thresh, 30,
                 np.where(df['deviation'] < -spike_thresh * 0.6, 15, 0))))

    # Se spike è nella stessa direzione del trend, dimezza
    same_dir = (core > 0) & (spike_bias > 0) | (core < 0) & (spike_bias < 0)
    spike_bias = np.where(same_dir, spike_bias // 2, spike_bias)

    target = np.clip(core + spike_bias, -pos_limit, pos_limit)
    df['target_pos'] = target

    # PnL con costo realistico (half-spread + 1 tick slippage per ordini aggressivi)
    df['pos_change'] = df.groupby('day')['target_pos'].diff().fillna(0)
    df['cost'] = abs(df['pos_change']) * (df['spread'] / 2 + 1)  # +1 tick slippage
    df['cashflow'] = -df['pos_change'] * df['mid_price']

    total_pnl = 0
    for day, g in df.groupby('day'):
        cash = g['cashflow'].sum() - g['cost'].sum()
        final_val = g['target_pos'].iloc[-1] * g['mid_price'].iloc[-1]
        total_pnl += (cash + final_val)

    trades = (df['pos_change'] != 0).sum()
    return total_pnl, trades


def run_grid_search(df):
    print("\n" + "=" * 50)
    print("=== STEP 2: GRID SEARCH (CORRETTO) ===")
    print("=" * 50)

    fast_lengths = [5, 8, 10, 12, 15]
    slow_lengths = [25, 30, 35, 40]
    trend_thresholds = [0.5, 1.0, 1.5, 2.0]
    spike_thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]

    combos = [(f, s, t, sp) for f, s, t, sp in product(fast_lengths, slow_lengths, trend_thresholds, spike_thresholds) if f < s]
    print(f"Testing {len(combos)} combinazioni...")

    results = []
    t0 = time.time()

    for fast, slow, trend, spike in combos:
        pnl, trades = simulate_strategy(df, fast, slow, trend, spike)
        results.append({
            'Fast': fast, 'Slow': slow, 'Trend_T': trend,
            'Spike_T': spike, 'Trades': trades, 'PnL': round(pnl, 0)
        })

    res = pd.DataFrame(results).sort_values('PnL', ascending=False)
    print(f"Completato in {time.time()-t0:.1f}s\n")

    print("TOP 10:")
    print(res.head(10).to_string(index=False))

    # Robustezza: stabilità del cluster
    top20 = res.head(20)
    print(f"\nCLUSTER TOP-20:")
    print(f"  Fast: {top20['Fast'].mode().values}")
    print(f"  Slow: {top20['Slow'].mode().values}")
    print(f"  Trend_T: {top20['Trend_T'].mode().values}")
    print(f"  Spike_T: {top20['Spike_T'].mode().values}")

    # Out-of-sample: train su day -2,-1, test su day 0
    print(f"\n=== OUT-OF-SAMPLE (train: day -2,-1 | test: day 0) ===")
    df_train = df[df['day'].isin([-2, -1])]
    df_test = df[df['day'] == 0]

    oos_results = []
    for _, row in res.head(10).iterrows():
        pnl_train, _ = simulate_strategy(df_train, int(row['Fast']), int(row['Slow']), row['Trend_T'], row['Spike_T'])
        pnl_test, _ = simulate_strategy(df_test, int(row['Fast']), int(row['Slow']), row['Trend_T'], row['Spike_T'])
        oos_results.append({
            'Fast': int(row['Fast']), 'Slow': int(row['Slow']),
            'Trend_T': row['Trend_T'], 'Spike_T': row['Spike_T'],
            'PnL_Train': round(pnl_train, 0), 'PnL_Test': round(pnl_test, 0),
            'Ratio': round(pnl_test / pnl_train, 2) if pnl_train > 0 else 0
        })

    oos_df = pd.DataFrame(oos_results)
    print(oos_df.to_string(index=False))


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'round1')
    df_trades, df_prices = load_data(data_dir)
    df = clean_pepper(df_prices)
    df = analyze_patterns(df)
    run_grid_search(df)
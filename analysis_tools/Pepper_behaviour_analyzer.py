import pandas as pd
import numpy as np
import os


def analyze_pepper_root_patterns():
    # Percorsi relativi alla root del progetto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trades_files = [os.path.join(project_root, f'data/round1/trades_round_1_day_{d}.csv') for d in [-2, -1, 0]]
    prices_files = [os.path.join(project_root, f'data/round1/prices_round_1_day_{d}.csv') for d in [-2, -1, 0]]

    # 1. Caricamento e Preparazione
    dfs_t = [pd.read_csv(f, sep=';').assign(day=i - 2) for i, f in enumerate(trades_files)]
    df_trades = pd.concat(dfs_t).rename(columns={'symbol': 'product'})

    dfs_p = [pd.read_csv(f, sep=';') for f in prices_files]
    df_prices = pd.concat(dfs_p).sort_values(['day', 'timestamp'])

    # 2. Join e Classificazione
    trades = pd.merge(df_trades, df_prices, on=['day', 'timestamp', 'product'], how='left')

    def classify_trade(row):
        if pd.isna(row['ask_price_1']): return 'neutral'
        if row['price'] >= row['ask_price_1']: return 'buy'
        if row['price'] <= row['bid_price_1']: return 'sell'
        return 'neutral'

    trades['direction'] = trades.apply(classify_trade, axis=1)
    pepper_trades = trades[trades['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    pepper_prices = df_prices[df_prices['product'] == 'INTARIAN_PEPPER_ROOT'].copy()

    print("=== FINAL EXPLOIT ANALYSIS: INTARIAN_PEPPER_ROOT ===\n")

    # [SCENARI 1 e 2 omessi per brevità, ma rimangono validi nel tuo file]

    # -------------------------------------------------------------------------
    # SCENARIO 5: DETTAGLIO BOOK EXHAUSTION (Sopra/Sotto e Livelli)
    # -------------------------------------------------------------------------
    def get_exhaustion_details(row):
        side = 'none'
        levels = 0
        qty = row['quantity']

        if row['direction'] == 'buy':
            side = 'ABOVE (Ask)'
            v1 = row.get('ask_volume_1', 0)
            v2 = row.get('ask_volume_2', 0)
            v3 = row.get('ask_volume_3', 0)
            # Pulizia NaN
            v1, v2, v3 = [0 if pd.isna(v) else v for v in [v1, v2, v3]]

            if qty >= v1: levels = 1
            if v2 > 0 and qty >= (v1 + v2): levels = 2
            if v3 > 0 and qty >= (v1 + v2 + v3): levels = 3

        elif row['direction'] == 'sell':
            side = 'BELOW (Bid)'
            v1 = row.get('bid_volume_1', 0)
            v2 = row.get('bid_volume_2', 0)
            v3 = row.get('bid_volume_3', 0)
            v1, v2, v3 = [0 if pd.isna(v) else v for v in [v1, v2, v3]]

            if qty >= v1: levels = 1
            if v2 > 0 and qty >= (v1 + v2): levels = 2
            if v3 > 0 and qty >= (v1 + v2 + v3): levels = 3

        return pd.Series([side, levels])

    pepper_trades[['exh_side', 'exh_levels']] = pepper_trades.apply(get_exhaustion_details, axis=1)

    total_trades = len(pepper_trades)
    exh_total = pepper_trades[pepper_trades['exh_levels'] > 0]

    print(f"5. ANALISI DETTAGLIATA SVUOTAMENTO BOOK:")
    print(f"   Trade totali analizzati: {total_trades}")
    print(f"   Svuotamenti totali: {len(exh_total)} ({len(exh_total) / total_trades * 100:.2f}%)")

    for side in ['ABOVE (Ask)', 'BELOW (Bid)']:
        side_df = pepper_trades[pepper_trades['exh_side'] == side]
        side_exh = side_df[side_df['exh_levels'] > 0]
        print(f"\n   --- Direzione: {side} ---")
        print(f"   Frequenza: {len(side_exh)} volte ({len(side_exh) / total_trades * 100:.2f}%)")
        for L in [1, 2, 3]:
            count = len(side_exh[side_exh['exh_levels'] == L])
            if count > 0:
                print(f"   - Livelli svuotati [{L}]: {count} volte")

    # SCENARIO 6: Mean Reversion (Il pattern più forte)
    pepper_prices['sma5'] = pepper_prices.groupby('day')['mid_price'].transform(lambda x: x.rolling(5).mean())
    pepper_prices['dev'] = pepper_prices['mid_price'] - pepper_prices['sma5']
    pepper_prices['next_ret'] = pepper_prices.groupby('day')['mid_price'].shift(-1) - pepper_prices['mid_price']
    pepper_prices['is_spike'] = pepper_prices['dev'].abs() > 4
    spikes = pepper_prices[pepper_prices['is_spike']]
    print(f"\n6. Spike Reversion (Deviazione > 4):")
    print(f"   Istanze: {len(spikes)} | Correlazione Reversione: {spikes['dev'].corr(spikes['next_ret']):.4f}")

    # -------------------------------------------------------------------------
    # SCENARIO 7: PERIODICITÀ E PREDIZIONE
    # -------------------------------------------------------------------------

    # A. Analisi della frequenza (Ogni quanto?)
    # Calcoliamo il tempo (in tick) dall'ultima spike
    pepper_prices['spike_id'] = (pepper_prices['is_spike'] != pepper_prices['is_spike'].shift()).cumsum()
    spike_events = pepper_prices[pepper_prices['is_spike']].groupby(['day', 'spike_id']).first().reset_index()
    spike_events['ticks_since_last'] = spike_events.groupby('day')['timestamp'].diff() / 100

    avg_period = spike_events['ticks_since_last'].mean()
    median_period = spike_events['ticks_since_last'].median()

    print(f"7. ANALISI TEMPORALE DELLE SPIKE:")
    print(f"   - Frequenza media: Una spike ogni {avg_period:.1f} tick")
    print(f"   - Frequenza mediana: Una spike ogni {median_period:.1f} tick")
    print(
        f"   - Clusterizzazione: Il 25% delle spike avviene entro {spike_events['ticks_since_last'].quantile(0.25):.1f} tick dalla precedente.")

    # B. Analisi dei precursori (Cosa succede PRIMA?)
    # Guardiamo i 2 tick precedenti a una spike
    pepper_prices['prev_dev_1'] = pepper_prices.groupby('day')['dev'].shift(1)
    pepper_prices['prev_spread'] = (pepper_prices['ask_price_1'] - pepper_prices['bid_price_1']).shift(1)

    spikes = pepper_prices[pepper_prices['is_spike']].copy()

    # Analizziamo la "Velocità di avvicinamento"
    print("\n   PRECURSORI DELLE SPIKE (1 tick prima):")
    # Se la deviazione precedente era già alta ma sotto la soglia, è un segnale di "caricamento"
    pre_spike_dev = spikes['prev_dev_1'].abs().mean()
    pre_spike_spread = spikes['prev_spread'].mean()

    print(f"   - Deviazione media pre-spike: {pre_spike_dev:.2f}")
    print(f"   - Spread medio pre-spike: {pre_spike_spread:.2f}")

    # C. Predittore: Probabilità di Spike
    # Se la deviazione attuale è tra 2.5 e 4.0, quanto è probabile che il prossimo tick sia una spike > 4.0?
    pepper_prices['next_is_spike'] = pepper_prices.groupby('day')['is_spike'].shift(-1).fillna(False)
    potential_setup = pepper_prices[(pepper_prices['dev'].abs() > 2.5) & (pepper_prices['dev'].abs() <= 4.0)]
    next_is_spike = pepper_prices.loc[potential_setup.index, 'next_is_spike'] if not potential_setup.empty else pd.Series([], dtype=bool)

    prob = (next_is_spike.sum() / len(potential_setup)) * 100 if len(potential_setup) > 0 else 0
    print(f"\n   POTERE PREDITTIVO:")
    print(f"   - Se la deviazione supera 2.5, probabilità di spike nel tick dopo: {prob:.1f}%")


if __name__ == "__main__":
    analyze_pepper_root_patterns()
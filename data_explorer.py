import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- CONFIGURATION ---
DATA_DIR = "./data/round1/"  # Change this when new rounds drop


def load_data():
    print("--- LOADING ROUND 1 DATA ---")

    days = ['-2', '-1', '0']

    df_prices_list = []
    df_trades_list = []
    TICKS_PER_DAY = 1000000

    for i, day in enumerate(days):
        # The exact names the script is looking for
        price_file = f"data/round1/prices_round_1_day_{day}.csv"
        trade_file = f"data/round1/trades_round_1_day_{day}.csv"

        print(f"Checking for: {price_file}")
        if os.path.exists(price_file):
            print(f"  -> Found {price_file}!")
            df_p = pd.read_csv(price_file, sep=';')
            df_p['global_time'] = df_p['timestamp'] + (i * TICKS_PER_DAY)
            df_p['day'] = int(day)
            df_prices_list.append(df_p)
        else:
            print(f"  -> MISSING: {price_file}")

        print(f"Checking for: {trade_file}")
        if os.path.exists(trade_file):
            print(f"  -> Found {trade_file}!")
            df_t = pd.read_csv(trade_file, sep=';')
            df_t['global_time'] = df_t['timestamp'] + (i * TICKS_PER_DAY)
            df_t['day'] = int(day)
            df_trades_list.append(df_t)
        else:
            print(f"  -> MISSING: {trade_file}")

    # Safely combine, falling back to empty DataFrames if nothing was found
    df_prices = pd.concat(
        df_prices_list, ignore_index=True) if df_prices_list else pd.DataFrame()
    df_trades = pd.concat(
        df_trades_list, ignore_index=True) if df_trades_list else pd.DataFrame()

    print(
        f"Loaded {len(df_prices)} price rows and {len(df_trades)} trade rows across {len(days)} days.")

    return df_prices, df_trades


def analyze_correlations(df_prices):
    print("\n--- 1. CROSS-ASSET CORRELATION ---")
    # Pivot the data so each column is an asset and each row is a timestamp
    pivot_prices = df_prices.pivot_table(
        index=['day', 'timestamp'], columns='product', values='mid_price')

    # Calculate daily returns (percentage change tick-to-tick)
    returns = pivot_prices.pct_change().dropna()

    corr_matrix = returns.corr()
    print("Correlation Matrix (Returns):")
    print(corr_matrix)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                center=0, vmin=-1, vmax=1)
    plt.title('Asset Return Correlation')
    plt.tight_layout()
    plt.show()


def analyze_bot_behavior(df_trades):
    print("\n--- 2. EXCHANGE BOT PROFILING ---")

    # Clean up names just in case
    df_trades['buyer'] = df_trades['buyer'].fillna(
        'UNKNOWN').astype(str).str.strip()
    df_trades['seller'] = df_trades['seller'].fillna(
        'UNKNOWN').astype(str).str.strip()

    # Who buys the most?
    buy_stats = df_trades.groupby(['buyer', 'symbol']).agg(
        total_buy_volume=('quantity', 'sum'),
        avg_buy_size=('quantity', 'mean'),
        buy_trades=('quantity', 'count')
    ).reset_index()

    # Who sells the most?
    sell_stats = df_trades.groupby(['seller', 'symbol']).agg(
        total_sell_volume=('quantity', 'sum'),
        avg_sell_size=('quantity', 'mean'),
        sell_trades=('quantity', 'count')
    ).reset_index()

    print("\nTop Buyers by Volume:")
    print(buy_stats.sort_values(by='total_buy_volume', ascending=False).head(10))

    print("\nTop Sellers by Volume:")
    print(sell_stats.sort_values(by='total_sell_volume', ascending=False).head(10))

# look if volumes spike at certain times of the day (e.g. bots that only trade at the start or end of the day)


def analyze_intraday_seasonality(df_trades):
    print("\n--- 3. VOLUME SEASONALITY ---")

    # Group trades into "buckets" to see if volume spikes at specific times
    # Assuming 1,000,000 timestamps per day, let's bucket by 10,000s
    df_trades['time_bucket'] = (df_trades['timestamp'] // 10000) * 10000

    vol_seasonality = df_trades.groupby(['symbol', 'time_bucket'])[
        'quantity'].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=vol_seasonality, x='time_bucket',
                 y='quantity', hue='symbol')
    plt.title('Trade Volume throughout the Day')
    plt.xlabel('Timestamp')
    plt.ylabel('Total Volume Traded')
    plt.tight_layout()
    plt.show()

# look for bots


def analyze_trade_signatures(df_prices, df_trades):
    print("\n--- NEW: BOT SIGNATURE & MARKOUT ANALYSIS ---")

    # Prep the price dataframe for merging
    df_prices_merge = df_prices[['global_time', 'product', 'mid_price']].copy()
    df_prices_merge.rename(columns={'product': 'symbol'}, inplace=True)

    # 1. Merge current price into the trades (matching exact asset and closest time)
    df_merged = pd.merge_asof(
        df_trades,
        df_prices_merge,
        on='global_time',
        by='symbol',
        direction='backward'
    )

    # 2. Look into the Future! (10 ticks = 1000 timestamp units in Prosperity)
    df_prices_future = df_prices_merge.copy()
    df_prices_future.rename(
        columns={'mid_price': 'future_price'}, inplace=True)
    df_merged['future_time'] = df_merged['global_time'] + 1000

    # Re-sort for the second merge
    df_merged.sort_values(by='future_time', inplace=True)

    df_merged = pd.merge_asof(
        df_merged,
        df_prices_future,
        left_on='future_time',
        right_on='global_time',
        by='symbol',
        direction='backward'
    )

    # 3. Calculate Markout (Did the price go up or down after they traded?)
    df_merged['buyer_markout'] = df_merged['future_price'] - \
        df_merged['mid_price']

    # 4. Profile the behavior by Trade Size
    size_profiler = df_merged.groupby(['symbol', 'quantity']).agg(
        trade_count=('quantity', 'count'),
        avg_buyer_profit_10_ticks=('buyer_markout', 'mean')
    ).reset_index()

    # Filter out the noise (only look at sizes that happen more than 50 times)
    significant_sizes = size_profiler[size_profiler['trade_count'] > 50].sort_values(
        by='avg_buyer_profit_10_ticks', ascending=False)

    print("\nTop Predictor Trade Sizes (If you see these quantities, copy them!):")
    print(significant_sizes.head(15))


def analyze_lagged_correlations(df_prices, max_lag=5):
    print("\n--- NEW: LAGGED CORRELATION (LEAD-LAG) ---")

    # Pivot to get prices side-by-side on the global timeline
    pivot_prices = df_prices.pivot_table(
        index='global_time', columns='product', values='mid_price')

    # We correlate returns (pct_change), not raw prices.
    # Correlating raw prices creates false positives if both just happen to go up over a day.
    returns = pivot_prices.pct_change().dropna()

    assets = returns.columns
    found_lead_lag = False

    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            asset1 = assets[i]
            asset2 = assets[j]

            best_corr = 0
            best_lag = 0

            # Shift the series from -max_lag to +max_lag ticks
            # Positive lag means asset1 LEADS asset2.
            for lag in range(-max_lag, max_lag + 1):
                corr = returns[asset1].corr(returns[asset2].shift(-lag))
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            # Filter out pure noise (only show somewhat meaningful correlations)
            if abs(best_corr) > 0.2:
                found_lead_lag = True
                if best_lag > 0:
                    print(
                        f"🔥 CRYSTAL BALL: {asset1} leads {asset2} by {best_lag} ticks! (Corr: {best_corr:.3f})")
                elif best_lag < 0:
                    print(
                        f"🔥 CRYSTAL BALL: {asset2} leads {asset1} by {-best_lag} ticks! (Corr: {best_corr:.3f})")
                else:
                    print(
                        f"🤝 Simultaneous move: {asset1} and {asset2} move exactly together (Corr: {best_corr:.3f})")

    if not found_lead_lag:
        print("🤷 No significant lead-lag relationships found among current assets.")


def analyze_simulation_determinism(df_trades):
    print("\n--- NEW: SIMULATION LAZINESS (DETERMINISM) CHECK ---")

    # --- CHECK 1: The "Groundhog Day" Exact Timestamp Match ---
    # We want to look at the daily 'timestamp', NOT the 'global_time'

    # Filter for meaningful trades (ignore 1-lot noise)
    large_trades = df_trades[df_trades['quantity'] >= 5]

    # Count how many different days a specific timestamp sees a large trade
    time_frequency = large_trades.groupby(['symbol', 'timestamp']).agg(
        days_active=('day', 'nunique'),
        total_volume=('quantity', 'sum'),
        avg_price=('price', 'mean')
    ).reset_index()

    # If a timestamp has large trades on multiple days, the my_bot is hardcoded to a clock!
    suspicious_times = time_frequency[time_frequency['days_active'] > 1].sort_values(
        by='total_volume', ascending=False)

    if not suspicious_times.empty:
        print("🚨 HARDCODED BOT DETECTED: Trades happening at the EXACT same timestamp across multiple days!")
        # Print top 5 worst offenders
        print(suspicious_times.head(5).to_string(index=False))
    else:
        print("✅ Groundhog Check: Clear. Bots do not trade at the exact same timestamps every day.")

    # --- CHECK 2: The "Metronome" Modulo Check ---
    # Do bots prefer round numbers? We check if volume spikes on ticks ending in 00.

    # Extract the last two digits of the timestamp
    df_trades['tick_ending'] = df_trades['timestamp'] % 100
    modulo_counts = df_trades.groupby(
        'tick_ending')['quantity'].sum().reset_index()

    # Calculate the average volume per tick ending
    avg_vol = modulo_counts['quantity'].mean()

    # Find spikes that are at least 3x larger than normal volume
    spikes = modulo_counts[modulo_counts['quantity'] > (avg_vol * 3)]

    if not spikes.empty:
        print("\n🚨 METRONOME BOT DETECTED: Massive volume spikes occurring on predictable clock cycles!")
        print(spikes.to_string(index=False))
    else:
        print("\n✅ Metronome Check: Clear. Volume is smoothly distributed across all clock ticks.")


def plot_bot_signatures(df_prices, df_trades):
    print("\n--- VISUALIZING BOT SIGNATURES ---")

    # The exact signatures you identified as the "Dumb Money"
    signatures = [
        {"symbol": "ASH_COATED_OSMIUM", "size": 5},
        {"symbol": "INTARIAN_PEPPER_ROOT", "size": 6}
    ]

    for sig in signatures:
        sym = sig["symbol"]
        target_size = sig["size"]

        # 1. Filter the data for this specific asset
        df_p = df_prices[df_prices['product'] == sym].copy()
        if df_p.empty:
            continue

        # 2. Filter the trades for exactly this size
        df_t = df_trades[(df_trades['symbol'] == sym) & (
            df_trades['quantity'] == target_size)].copy()

        if df_t.empty:
            print(f"No trades of size {target_size} found for {sym}.")
            continue

        print(
            f"Plotting {len(df_t)} trades of size {target_size} for {sym}...")

        # 3. Create the plot
        plt.figure(figsize=(15, 7))

        # Plot the continuous Mid Price in blue
        plt.plot(df_p['global_time'], df_p['mid_price'],
                 label=f"{sym} Mid Price", color='royalblue', alpha=0.7)

        # Overlay the exact trades as giant Red Xs
        # We use the trade execution price for the Y-axis so we see exactly where they got filled
        plt.scatter(df_t['global_time'], df_t['price'], color='red', marker='X', s=120, zorder=5,
                    label=f"The 'Dumb Bot' (Size {target_size})")

        plt.title(
            f"Hunting the Vulnerable Bot: {sym} (Trade Size {target_size})", fontsize=16, fontweight='bold')
        plt.xlabel("Simulation Time (Ticks)", fontsize=12)
        plt.ylabel("Price (SeaShells)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def profile_bot_behavior(df_prices, df_trades):
    print("\n--- PROFILING DUMB BOT BEHAVIOR ---")

    signatures = [
        {"symbol": "ASH_COATED_OSMIUM", "size": 5},
        {"symbol": "INTARIAN_PEPPER_ROOT", "size": 6}
    ]

    # We only need mid_price from the prices dataframe
    df_p_lite = df_prices[['global_time', 'product', 'mid_price']].copy()

    for sig in signatures:
        sym = sig["symbol"]
        target_size = sig["size"]

        # 1. Isolate the specific my_bot's trades
        df_bot = df_trades[(df_trades['symbol'] == sym) & (
            df_trades['quantity'] == target_size)].copy()

        if df_bot.empty:
            continue

        # 2. Merge the trade with the exact mid_price at that timestamp
        df_bot = df_bot.merge(
            df_p_lite[df_p_lite['product'] == sym],
            on='global_time',
            how='left'
        )

        # 3. Calculate "Carelessness" (Execution Price vs. Mid Price)
        # If they buy, how much OVER mid did they pay? If they sell, how much UNDER?
        df_bot['dist_from_mid'] = df_bot.apply(
            lambda row: (row['price'] - row['mid_price']
                         ) if row['buyer'] == "BOT_NAME_HERE" else (row['mid_price'] - row['price']),
            axis=1
        )
        # Note: Since Prosperity anonymizes counterparty names in training data,
        # we simplify: Absolute distance from mid shows how far they crossed the spread.
        df_bot['abs_dist_from_mid'] = abs(
            df_bot['price'] - df_bot['mid_price'])

        # 4. Calculate Frequency (Time between trades)
        df_bot = df_bot.sort_values('global_time')
        df_bot['ticks_since_last'] = df_bot['global_time'].diff()

        print(f"\n[ Profile: {sym} | Size: {target_size} ]")
        print(f"Total Trades Observed: {len(df_bot)}")

        avg_dist = df_bot['abs_dist_from_mid'].mean()
        max_dist = df_bot['abs_dist_from_mid'].max()
        print(f"Average Slippage (Ticks from Mid): {avg_dist:.2f}")
        print(f"Worst Execution (Ticks from Mid): {max_dist:.2f}")

        avg_time = df_bot['ticks_since_last'].mean()
        print(f"Average Ticks Between Trades: {avg_time:.0f}")

        # Look for deterministic timers (standard deviation near 0 means it's a fixed timer)
        time_std = df_bot['ticks_since_last'].std()
        if time_std < 100:
            print(
                f"⚠️ HIGHLY PREDICTABLE TIMER DETECTED! (StdDev: {time_std:.1f})")


def main():
    df_prices, df_trades = load_data()
    if df_prices is not None:
        # analyze_correlations(df_prices)
        # analyze_bot_behavior(df_trades)
        # analyze_intraday_seasonality(df_trades)

        # --- NEW FUNCTIONS ---
        analyze_trade_signatures(df_prices, df_trades)
        analyze_lagged_correlations(df_prices)
        # analyze_simulation_determinism(df_trades)
        plot_bot_signatures(df_prices, df_trades)
        profile_bot_behavior(df_prices, df_trades)


if __name__ == "__main__":
    main()

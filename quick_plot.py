import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import glob
import os
import re


def main():
    # 1. Find the latest log file automatically
    list_of_files = glob.glob('./*.log')
    # If your logs are in a folder, use this instead: list_of_files = glob.glob('./backtests/*.log')
    if not list_of_files:
        print("No .log files found in the current directory.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📈 Generating quick plots from: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. Extract sections
    activities_text = ""
    trades_text = ""

    if "Activities log:" in content:
        parts = content.split("Activities log:")
        raw_activities = parts[-1].strip()
        activities_text = raw_activities.split("Trade History:")[0].strip()

    if "Trade History:" in content:
        trades_text = content.split("Trade History:")[-1].strip()

    if not activities_text:
        print("❌ No activities log found. Did the backtest fail?")
        return

    # 3. Parse Activities
    df = pd.read_csv(io.StringIO(activities_text), sep=";",
                     on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.strip()
    df = df[df['product'] != 'product']

    df['profit_and_loss'] = pd.to_numeric(
        df['profit_and_loss'], errors='coerce')
    df['mid_price'] = pd.to_numeric(df['mid_price'], errors='coerce')

    # We use the raw timestamp since the engine already made it continuous
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # 4. Parse Trades (Bulletproof Regex Parser)
    trades_list = []
    if trades_text:
        matches = re.finditer(r'\{[^{}]+\}', trades_text)
        for match in matches:
            block = match.group(0)
            try:
                # Extract the pre-stitched timestamp directly
                ts = int(re.search(r'"timestamp":\s*(\d+)', block).group(1))
                buyer = re.search(r'"buyer":\s*"([^"]*)"', block).group(1)
                seller = re.search(r'"seller":\s*"([^"]*)"', block).group(1)
                symbol = re.search(r'"symbol":\s*"([^"]*)"', block).group(1)
                qty = int(re.search(r'"quantity":\s*(\d+)', block).group(1))

                trades_list.append({
                    "timestamp": ts,
                    "buyer": buyer,
                    "seller": seller,
                    "symbol": symbol,
                    "quantity": qty
                })
            except Exception:
                pass

    trades_df = pd.DataFrame(trades_list)

    # 5. Process and Plot per Asset
    products = df['product'].dropna().unique()
    BOT_NAME = "SUBMISSION"

    for product in products:
        prod_df = df[df['product'] == product].copy()

        # Calculate Cumulative Inventory
        prod_df['inventory'] = 0
        if not trades_df.empty and 'symbol' in trades_df.columns:
            sym_trades = trades_df[trades_df['symbol'] == product].copy()
            if not sym_trades.empty:
                sym_trades['trade_delta'] = 0
                sym_trades.loc[sym_trades['buyer'] == BOT_NAME,
                               'trade_delta'] = sym_trades['quantity']
                sym_trades.loc[sym_trades['seller'] == BOT_NAME,
                               'trade_delta'] = -sym_trades['quantity']

                # Group strictly by the continuous timestamp
                net_trades = sym_trades.groupby(
                    'timestamp')['trade_delta'].sum().reset_index()

                prod_df = prod_df.merge(net_trades, on='timestamp', how='left')
                prod_df['trade_delta'] = prod_df['trade_delta'].fillna(0)
                prod_df['inventory'] = prod_df['trade_delta'].cumsum()

        # Build the Interactive Plotly Figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f"{product}: Profit & Loss",
                            "Mid Price", "Inventory Position")
        )

        # Panel 1: Profit & Loss
        fig.add_trace(go.Scatter(x=prod_df['timestamp'], y=prod_df['profit_and_loss'],
                                 mode='lines', name='PnL', line=dict(color='#2ca02c', width=2)), row=1, col=1)

        # Panel 2: Mid Price
        fig.add_trace(go.Scatter(x=prod_df['timestamp'], y=prod_df['mid_price'], mode='lines',
                                 name='Mid Price', line=dict(color='#1f77b4', width=2)), row=2, col=1)

        # Panel 3: Inventory Tracker
        fig.add_trace(go.Scatter(x=prod_df['timestamp'], y=prod_df['inventory'], mode='lines', name='Inventory', line=dict(
            color='#17becf', shape='hv', width=2)), row=3, col=1)

        # Add limit lines for the inventory (Adjust y=20 to your actual limits if needed)
        fig.add_hline(y=0, line_dash="dash", line_color="black",
                      opacity=0.3, row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="red",
                      opacity=0.5, row=3, col=1)
        fig.add_hline(y=-80, line_dash="dot", line_color="red",
                      opacity=0.5, row=3, col=1)

        fig.update_layout(
            title_text=f"Quick Plot: {product}",
            height=800,
            hovermode="x unified",
            showlegend=False
        )

        fig.show()


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import glob
import os
import json
import re

# --- CONFIGURATION ---
POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
    # You can add future round assets here!
}
DEFAULT_LIMIT = 80  # Fallback just in case


st.set_page_config(page_title="Prosperity 4 Visualizer", layout="wide")
st.title("Prosperity 4 Local Visualizer")

# 1. Initialize variables safely
activities_text = None
trades_text = None
content = None

# 2. File Selection: Auto-load or Upload
uploaded_file = st.file_uploader(
    "Upload your backtest .log file (Optional)", type="log")

if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8")
    st.success("Loaded uploaded file.")
else:
    list_of_files = glob.glob('./*.log')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        st.info(f"Automatically loaded latest run: {latest_file}")
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()

# 3. Extract the Sections
if content:
    if content and content.strip().startswith('{'):
        # formato JSON ufficiale
        data = json.loads(content)
        content = data.get('activitiesLog', '')
        # converti \n in newline reali
        content = "Activities log:\n" + content.replace('\\n', '\n')
    # Extract Activities
    if "Activities log:" in content:
        parts = content.split("Activities log:")
        raw_activities = parts[-1].strip()
        # Make sure we don't grab the Trade History by accident
        activities_text = raw_activities.split("Trade History:")[0].strip()

    # Extract Trades
    if "Trade History:" in content:
        trades_text = content.split("Trade History:")[-1].strip()

# 4. Parse Data and Draw Graphs
if activities_text:
    try:
        # Load Activities
        df = pd.read_csv(io.StringIO(activities_text), sep=";",
                         on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip()
        df = df[df['product'] != 'product']

        # --- BULLETPROOF TRADE EXTRACTOR ---
        trades_list = []
        if trades_text:
            try:
                # 1. Try standard JSON first (This works for official IMC server logs)
                trades_list = json.loads(trades_text)
            except json.JSONDecodeError:
                # 2. Fallback: Regex parser for local backtester's pseudo-JSON
                matches = re.finditer(r'\{[^{}]+\}', trades_text)
                for match in matches:
                    block = match.group(0)
                    try:
                        ts = int(
                            re.search(r'"timestamp":\s*(\d+)', block).group(1))
                        buyer = re.search(
                            r'"buyer":\s*"([^"]*)"', block).group(1)
                        seller = re.search(
                            r'"seller":\s*"([^"]*)"', block).group(1)
                        symbol = re.search(
                            r'"symbol":\s*"([^"]*)"', block).group(1)
                        qty = int(
                            re.search(r'"quantity":\s*(\d+)', block).group(1))

                        trades_list.append({
                            "timestamp": ts, "buyer": buyer, "seller": seller,
                            "symbol": symbol, "quantity": qty
                        })
                    except Exception:
                        pass  # Skip mangled blocks

        trades_df = pd.DataFrame(trades_list)

        if 'product' in df.columns:
            df['profit_and_loss'] = pd.to_numeric(
                df['profit_and_loss'], errors='coerce')

            # --- TOP DASHBOARD ---
            final_pnl = df.groupby('product')['profit_and_loss'].last().sum()
            st.metric(label="Total Strategy PnL",
                      value=f"{final_pnl:,.0f} XIRECs")
            st.divider()

            # --- ASSET SELECTOR ---
            products = df['product'].dropna().unique()
            selected_product = st.selectbox(
                "Select Asset to Analyze", products)

            # THE HEDGEHOG NORMALIZATION TOGGLE
            normalize_chart = st.checkbox(
                "Normalize Prices (Mean Reversion View)", value=False)

            # Filter activities for selected product
            prod_df = df[df['product'] == selected_product].copy()

            # --- CALCULATE INVENTORY ---
            prod_df['inventory'] = 0
            if not trades_df.empty and 'symbol' in trades_df.columns:

                # Strip invisible spaces from the names
                trades_df['buyer'] = trades_df['buyer'].astype(str).str.strip()
                trades_df['seller'] = trades_df['seller'].astype(
                    str).str.strip()

                # Isolate trades for this asset
                sym_trades = trades_df[trades_df['symbol']
                                       == selected_product].copy()

                if not sym_trades.empty:
                    sym_trades['trade_delta'] = 0

                    # The strict my_bot name we discovered earlier
                    BOT_NAME = "SUBMISSION"

                    sym_trades.loc[sym_trades['buyer'] == BOT_NAME,
                                   'trade_delta'] = sym_trades['quantity']
                    sym_trades.loc[sym_trades['seller'] == BOT_NAME,
                                   'trade_delta'] = -sym_trades['quantity']

                    # Group by timestamp in case of multiple fills in one tick
                    net_trades = sym_trades.groupby(
                        'timestamp')['trade_delta'].sum().reset_index()

                    # Merge with the main timeline and calculate cumulative sum
                    prod_df = prod_df.merge(
                        net_trades, on='timestamp', how='left')
                    prod_df['trade_delta'] = prod_df['trade_delta'].fillna(0)
                    prod_df['inventory'] = prod_df['trade_delta'].cumsum()

            # --- ASSET METRICS ---
            col1, col2, col3 = st.columns(3)
            with col1:
                asset_pnl = prod_df['profit_and_loss'].iloc[-1] if not prod_df.empty else 0
                st.metric(label=f"{selected_product} PnL",
                          value=f"{asset_pnl:,.0f}")
            with col2:
                mean_pos = prod_df['inventory'].mean()
                st.metric(label="Mean Position (Bias)",
                          value=f"{mean_pos:,.2f}", help="Close to 0 is neutral.")
            with col3:
                mean_abs_pos = prod_df['inventory'].abs().mean()
                st.metric(label="Mean Absolute Position (Risk)",
                          value=f"{mean_abs_pos:,.2f}", help="High numbers mean heavy inventory.")

            # --- GRAPH 1: PnL ---
            st.subheader("Profit & Loss")
            fig_pnl = px.line(prod_df, x='timestamp',
                              y='profit_and_loss', markers=True)
            # Replaced deprecated use_container_width
            st.plotly_chart(fig_pnl, width="stretch")

            # --- GRAPH 2: Market View (Order Book) ---
            st.subheader("Market View")
            prod_df = prod_df.ffill()
            fig_price = go.Figure()

            # Apply Normalization if checked!
            if normalize_chart:
                # Subtract the mid_price from the bids and asks
                mid_baseline = prod_df['mid_price']
                # This just becomes a flat line at 0
                plot_mid = prod_df['mid_price'] - mid_baseline
                plot_bid = prod_df['bid_price_1'] - mid_baseline
                plot_ask = prod_df['ask_price_1'] - mid_baseline
                y_title = "Deviation from Mid Price"
            else:
                plot_mid = prod_df['mid_price']
                plot_bid = prod_df['bid_price_1']
                plot_ask = prod_df['ask_price_1']
                y_title = "Actual Price"

            fig_price.add_trace(go.Scatter(
                x=prod_df['timestamp'], y=plot_mid, mode='lines', name='Mid Price', line=dict(color='white', width=1)))
            fig_price.add_trace(go.Scatter(x=prod_df['timestamp'], y=plot_bid, mode='markers', name='Market Bid', marker=dict(
                color='lightgreen', size=3, opacity=0.4)))
            fig_price.add_trace(go.Scatter(x=prod_df['timestamp'], y=plot_ask, mode='markers', name='Market Ask', marker=dict(
                color='salmon', size=3, opacity=0.4)))

            fig_price.update_layout(hovermode="x unified", yaxis_title=y_title, xaxis=dict(
                rangeslider=dict(visible=True), type="linear"))
            st.plotly_chart(fig_price, width="stretch")

            # --- GRAPH 3: INVENTORY TRACKER ---
            st.subheader("Inventory Position")
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(x=prod_df['timestamp'], y=prod_df['inventory'],
                              mode='lines', name='Inventory', line=dict(color='cyan', shape='hv', width=2)))

            fig_inv.add_hline(y=0, line_dash="dash",
                              line_color="gray", opacity=0.5)

            # Fetch the dynamic limit for the selected asset
            limit = POSITION_LIMITS.get(selected_product, DEFAULT_LIMIT)

            # Draw the dynamic boundary lines
            fig_inv.add_hline(y=limit, line_dash="dot",
                              line_color="red", opacity=0.3)
            fig_inv.add_hline(y=-limit, line_dash="dot",
                              line_color="red", opacity=0.3)

            fig_inv.update_layout(hovermode="x unified")
            st.plotly_chart(fig_inv, width="stretch")

        else:
            st.error("Missing 'product' column in Activities log.")

    except Exception as e:
        st.error(f"Failed during analysis: {e}")

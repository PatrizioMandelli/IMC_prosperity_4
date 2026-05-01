"""
Prosperity 4 Universal Framework Dashboard
Usage: streamlit run visualizer-framework/dashboard.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.pipeline.loader import UniversalLoader
from framework.pipeline.normalizer import LOBNormalizer
from framework.pipeline.registry import RoundRegistry

DARK_BG = "#0d0d0d"
PANEL   = "#12122a"
TEXT    = "#e8e8f0"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
BLUE    = "#3498db"

POSITION_LIMITS: dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}
DEFAULT_LIMIT = 80

st.set_page_config(page_title="Prosperity 4 Framework", layout="wide")
st.title("Prosperity 4 — Universal Analysis Framework")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar_loader():
    with st.sidebar:
        st.header("Data Source")
        source = st.radio("Source", ["Upload .log/.json", "CSV directory"], index=0)

        prices_df = pd.DataFrame()
        trades_df = pd.DataFrame()
        label = ""

        if source == "Upload .log/.json":
            uploaded = st.file_uploader("Upload backtest file", type=["log", "json"])
            if uploaded:
                try:
                    content = uploaded.getvalue().decode("utf-8", errors="replace")
                    prices_df, trades_df = UniversalLoader.from_content(content)
                    if not prices_df.empty:
                        label = uploaded.name
                        st.success(f"Loaded: {uploaded.name} ({len(prices_df):,} rows)")
                    else:
                        st.error("No price data found in the uploaded file.")
                except Exception as e:
                    st.error(f"Parse error: {e}")
        else:
            data_root = st.text_input("data/ directory path", value="data")
            registry = RoundRegistry.scan(data_root)
            if not registry:
                st.warning("No rounds found. Check the path.")
            else:
                round_n = st.selectbox("Round", list(registry.keys()))
                available_days = registry[round_n]
                selected_days = st.multiselect("Days", available_days, default=available_days)
                if st.button("Load Data") and selected_days:
                    with st.spinner("Loading CSV data..."):
                        try:
                            prices_df, trades_df = UniversalLoader.from_csv_dir(
                                data_root, round_n=round_n, days=selected_days
                            )
                            label = f"Round {round_n}, Days {selected_days}"
                            st.success(f"Loaded {len(prices_df):,} price rows, {len(trades_df):,} trade rows")
                            st.session_state["prices_df"] = prices_df
                            st.session_state["trades_df"] = trades_df
                            st.session_state["label"] = label
                        except Exception as e:
                            st.error(f"Load error: {e}")
                if "prices_df" in st.session_state and prices_df.empty:
                    prices_df = st.session_state["prices_df"]
                    trades_df = st.session_state["trades_df"]
                    label = st.session_state.get("label", "")

        return prices_df, trades_df, label


prices_df, trades_df, label = _sidebar_loader()

if prices_df.empty:
    st.info("Load data from the sidebar to begin analysis.")
    st.stop()

# Normalize once
@st.cache_data(show_spinner="Normalizing LOB data...")
def _normalize(prices_hash: str, prices_json: str) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(prices_json), orient="split")
    return LOBNormalizer.normalize(df)

prices_hash = f"{len(prices_df)}_{prices_df['global_time'].iloc[-1] if 'global_time' in prices_df.columns else 0}"
norm_df = _normalize(prices_hash, prices_df.to_json(orient="split"))

products = sorted(norm_df["product"].dropna().unique().tolist()) if "product" in norm_df.columns else []


# ── Tab: Overview ─────────────────────────────────────────────────────────────

st.subheader(f"Overview — {label}")

if "product" not in prices_df.columns:
    st.error("Missing 'product' column in price data.")
else:
    final_pnl = 0
    if "profit_and_loss" in prices_df.columns:
        final_pnl = prices_df.groupby("product")["profit_and_loss"].last().sum()
    st.metric("Total Strategy PnL", f"{final_pnl:,.0f} XIRECs")
    st.divider()

    selected_product = st.selectbox("Select Asset", products, key="overview_product")
    normalize_chart = st.checkbox("Normalize Prices (Mean Reversion View)", value=False)

    prod_df = prices_df[prices_df["product"] == selected_product].copy()

    prod_df["inventory"] = 0
    t_col = "symbol" if "symbol" in trades_df.columns else "product"
    if not trades_df.empty and t_col in trades_df.columns:
        t2 = trades_df.copy()
        t2["buyer"] = t2.get("buyer", pd.Series(dtype=str)).astype(str).str.strip()
        t2["seller"] = t2.get("seller", pd.Series(dtype=str)).astype(str).str.strip()
        sym_trades = t2[t2[t_col] == selected_product].copy()
        if not sym_trades.empty:
            sym_trades["trade_delta"] = 0
            sym_trades.loc[sym_trades["buyer"] == "SUBMISSION", "trade_delta"] = sym_trades["quantity"]
            sym_trades.loc[sym_trades["seller"] == "SUBMISSION", "trade_delta"] = -sym_trades["quantity"]
            net = sym_trades.groupby("timestamp")["trade_delta"].sum().reset_index()
            prod_df = prod_df.merge(net, on="timestamp", how="left")
            prod_df["trade_delta"] = prod_df["trade_delta"].fillna(0)
            prod_df["inventory"] = prod_df["trade_delta"].cumsum()

    col1, col2, col3 = st.columns(3)
    with col1:
        asset_pnl = prod_df["profit_and_loss"].iloc[-1] if not prod_df.empty and "profit_and_loss" in prod_df else 0
        st.metric(f"{selected_product} PnL", f"{asset_pnl:,.0f}")
    with col2:
        st.metric("Mean Position (Bias)", f"{prod_df['inventory'].mean():,.2f}")
    with col3:
        st.metric("Mean Abs Position (Risk)", f"{prod_df['inventory'].abs().mean():,.2f}")

    x_col = "global_time" if "global_time" in prod_df.columns else "timestamp"

    for c in ["bid_price_1", "ask_price_1", "bid_price_2", "ask_price_2",
              "bid_price_3", "ask_price_3", "mid_price", "micro_price"]:
        if c in prod_df.columns:
            prod_df[c] = prod_df[c].replace(0, np.nan)
    prod_df = prod_df.ffill()

    plot_df = prod_df.sort_values(x_col).reset_index(drop=True)

    day_boundaries = []
    if "day" in plot_df.columns and plot_df["day"].nunique() > 1:
        days = sorted(plot_df["day"].dropna().unique())
        for i in range(len(days) - 1):
            last_idx = int(plot_df[plot_df["day"] == days[i]].index.max())
            day_boundaries.append(last_idx + 0.5)

    x_disp = plot_df.index

    st.subheader("Profit & Loss")
    if "profit_and_loss" in plot_df.columns:
        fig_pnl = px.line(plot_df.assign(_t=x_disp), x="_t", y="profit_and_loss", markers=False)
        fig_pnl.update_layout(xaxis_title="Tick (sequential)")
        for b in day_boundaries:
            fig_pnl.add_vline(x=b, line_dash="dash", line_color="gray", opacity=0.4)
        st.plotly_chart(fig_pnl, use_container_width=True)

    st.subheader("Market View")
    plot_df["mid_smooth"] = plot_df["mid_price"].rolling(window=50, min_periods=1).mean()

    if normalize_chart:
        mid_baseline = float(plot_df["mid_price"].dropna().iloc[0]) if not plot_df["mid_price"].dropna().empty else 0.0
        plot_mid = plot_df["mid_price"] - mid_baseline
        plot_mid_s = plot_df["mid_smooth"] - mid_baseline
        plot_bid = (plot_df["bid_price_1"] if "bid_price_1" in plot_df.columns else plot_df["mid_price"]) - mid_baseline
        plot_ask = (plot_df["ask_price_1"] if "ask_price_1" in plot_df.columns else plot_df["mid_price"]) - mid_baseline
        y_title = "Deviation from First Price"
    else:
        plot_mid = plot_df["mid_price"]
        plot_mid_s = plot_df["mid_smooth"]
        plot_bid = plot_df["bid_price_1"] if "bid_price_1" in plot_df.columns else plot_df["mid_price"]
        plot_ask = plot_df["ask_price_1"] if "ask_price_1" in plot_df.columns else plot_df["mid_price"]
        y_title = "Price"

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=x_disp, y=plot_mid, mode="lines", name="Mid (raw)",
                                   line=dict(color="rgba(255,255,255,0.25)", width=1), connectgaps=True))
    fig_price.add_trace(go.Scatter(x=x_disp, y=plot_mid_s, mode="lines", name="Mid Smooth (50t)",
                                   line=dict(color="white", width=2), connectgaps=True))
    fig_price.add_trace(go.Scatter(x=x_disp, y=plot_bid, mode="markers", name="Bid L1",
                                   marker=dict(color="lightgreen", size=3, opacity=0.35)))
    fig_price.add_trace(go.Scatter(x=x_disp, y=plot_ask, mode="markers", name="Ask L1",
                                   marker=dict(color="salmon", size=3, opacity=0.35)))
    for b in day_boundaries:
        fig_price.add_vline(x=b, line_dash="dash", line_color="gray", opacity=0.4)
    fig_price.update_layout(
        hovermode="x unified", yaxis_title=y_title,
        xaxis=dict(title="Tick (sequential)", rangeslider=dict(visible=True)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL, font=dict(color=TEXT),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Inventory Position")
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=x_disp, y=plot_df["inventory"], mode="lines",
                                 name="Inventory", line=dict(color="cyan", shape="hv", width=2)))
    fig_inv.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    limit = POSITION_LIMITS.get(selected_product, DEFAULT_LIMIT)
    fig_inv.add_hline(y=limit, line_dash="dot", line_color="red", opacity=0.3)
    fig_inv.add_hline(y=-limit, line_dash="dot", line_color="red", opacity=0.3)
    for b in day_boundaries:
        fig_inv.add_vline(x=b, line_dash="dash", line_color="gray", opacity=0.4)
    fig_inv.update_layout(
        hovermode="x unified", xaxis_title="Tick (sequential)",
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL, font=dict(color=TEXT),
    )
    st.plotly_chart(fig_inv, use_container_width=True)

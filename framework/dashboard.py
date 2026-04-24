"""
Prosperity 4 Universal Framework Dashboard
Usage: streamlit run framework/dashboard.py
"""

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.pipeline.loader import UniversalLoader
from framework.pipeline.normalizer import LOBNormalizer
from framework.pipeline.registry import RoundRegistry
from framework.charts.heatmap import lob_heatmap
from framework.charts.bot_report import bot_detection_figure, detection_summary_table
from framework.charts.quant_panels import mm_panel, statarb_panel, vol_panel
from framework.charts.arbitrage_panels import basket_arb_panel, vol_arb_panel
from framework.quant.forensics import handle_discover, handle_forensic_report, handle_impact_test
from framework.detectors.hidden_taker import HiddenTakerDetector
from framework.detectors.pseudo_directional import PseudoDirectionalDetector
from framework.detectors.insider import InsiderSimulator
from framework.detectors.spoofing_detector import SpoofingDetector
from framework.quant.statarb import pair_spread

# ── Color palette ─────────────────────────────────────────────────────────────
DARK_BG = "#0d0d0d"
PANEL   = "#12122a"
TEXT    = "#e8e8f0"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
BLUE    = "#3498db"
YELLOW  = "#f1c40f"
PURPLE  = "#9b59b6"

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
                    # Read content directly from memory
                    content = uploaded.getvalue().decode("utf-8", errors="replace")
                    prices_df, trades_df = UniversalLoader.from_content(content)
                    
                    if not prices_df.empty:
                        label = uploaded.name
                        st.success(f"Loaded: {uploaded.name} ({len(prices_df):,} rows)")
                    else:
                        st.error("No price data found in the uploaded file. Check the format.")
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
                # Use cached data if available
                if "prices_df" in st.session_state and prices_df.empty:
                    prices_df = st.session_state["prices_df"]
                    trades_df = st.session_state["trades_df"]
                    label = st.session_state.get("label", "")

        st.divider()
        st.subheader("Detector Settings")
        agent_id = st.text_input(
            "Agent ID for Insider Simulator",
            value="",
            help="Leave blank to auto-detect top-volume agent",
        )
        st.session_state["agent_id"] = agent_id or None

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

@st.cache_data(show_spinner="Classifying trades...")
def _classify(trades_json: str, prices_json: str) -> pd.DataFrame:
    t = pd.read_json(io.StringIO(trades_json), orient="split")
    p = pd.read_json(io.StringIO(prices_json), orient="split")
    if t.empty:
        return t
    return LOBNormalizer.classify_trades(t, p)

prices_hash = f"{len(prices_df)}_{prices_df['global_time'].iloc[-1] if 'global_time' in prices_df.columns else 0}"
norm_df = _normalize(prices_hash, prices_df.to_json(orient="split"))

classified_trades = _classify(
    trades_df.to_json(orient="split"),
    norm_df.to_json(orient="split"),
) if not trades_df.empty else trades_df

products = sorted(norm_df["product"].dropna().unique().tolist()) if "product" in norm_df.columns else []

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["Overview", "LOB Heatmap", "Bot Detectors", "Quant Signals", "Arbitrage", "Forensics", "ML Predictor", "Data Explorer"])

# ── Tab 0: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Overview — {label}")

    # Replicate visualizer.py behavior exactly
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

        # Inventory calculation (mirrors visualizer.py exactly)
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

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            asset_pnl = prod_df["profit_and_loss"].iloc[-1] if not prod_df.empty and "profit_and_loss" in prod_df else 0
            st.metric(f"{selected_product} PnL", f"{asset_pnl:,.0f}")
        with col2:
            st.metric("Mean Position (Bias)", f"{prod_df['inventory'].mean():,.2f}")
        with col3:
            st.metric("Mean Abs Position (Risk)", f"{prod_df['inventory'].abs().mean():,.2f}")

        x_col = "global_time" if "global_time" in prod_df.columns else "timestamp"

        # Fix spikes: replace 0 with NaN in all price columns, then forward-fill
        for c in ["bid_price_1", "ask_price_1", "bid_price_2", "ask_price_2",
                  "bid_price_3", "ask_price_3", "mid_price", "micro_price"]:
            if c in prod_df.columns:
                prod_df[c] = prod_df[c].replace(0, np.nan)
        prod_df = prod_df.ffill()

        # Sort once and build a truly sequential index so inter-day gaps disappear
        plot_df = prod_df.sort_values(x_col).reset_index(drop=True)

        # Day boundary lines in sequential space (vertical separators)
        day_boundaries = []
        if "day" in plot_df.columns and plot_df["day"].nunique() > 1:
            days = sorted(plot_df["day"].dropna().unique())
            for i in range(len(days) - 1):
                last_idx = int(plot_df[plot_df["day"] == days[i]].index.max())
                day_boundaries.append(last_idx + 0.5)

        # Sequential tick index used as x-axis for all charts (0, 1, 2, …)
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


# ── Tab 1: LOB Heatmap ────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("LOB Heatmap")
    col_l, col_r = st.columns([1, 3])
    with col_l:
        hm_product = st.selectbox("Product", products, key="hm_product")
        time_bins = st.slider("Time Bins", 50, 500, 200, step=25)
        price_bins = st.slider("Price Bins", 20, 100, 50, step=5)

    @st.cache_data(show_spinner="Building LOB heatmap...")
    def _build_heatmap(p_json: str, t_json: str, product: str, tb: int, pb: int) -> go.Figure:
        p = pd.read_json(io.StringIO(p_json), orient="split")
        t = pd.read_json(io.StringIO(t_json), orient="split") if t_json else pd.DataFrame()
        return lob_heatmap(p, t, product=product, time_bins=tb, price_bins=pb)

    t_json = classified_trades.to_json(orient="split") if not classified_trades.empty else ""
    fig_hm = _build_heatmap(norm_df.to_json(orient="split"), t_json, hm_product, time_bins, price_bins)
    st.plotly_chart(fig_hm, use_container_width=True)


# ── Tab 2: Bot Detectors ──────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Adversarial Bot Detectors")
    det_product = st.selectbox("Product", products, key="det_product")

    agent_id_setting = st.session_state.get("agent_id")

    @st.cache_data(show_spinner="Running bot detectors...")
    def _run_detectors(p_json: str, t_json: str, product: str, agent_id: str | None):
        p = pd.read_json(io.StringIO(p_json), orient="split")
        t = pd.read_json(io.StringIO(t_json), orient="split") if t_json else pd.DataFrame()
        pp = p[p["product"] == product] if "product" in p.columns else p
        tt = t[t["product"] == product] if not t.empty and "product" in t.columns else t

        results = []
        for detector in [
            HiddenTakerDetector(),
            PseudoDirectionalDetector(),
            InsiderSimulator(agent_id=agent_id),
            SpoofingDetector(),
        ]:
            try:
                results.append(detector.detect(pp, tt))
            except Exception as e:
                from framework.detectors.base import DetectionResult
                results.append(DetectionResult(
                    detector_name=type(detector).__name__,
                    product=product,
                    confidence=0.0,
                    pattern_description=f"Error: {e}",
                    counter_strategy="N/A",
                ))
        return results

    t_json = classified_trades.to_json(orient="split") if not classified_trades.empty else ""
    results = _run_detectors(norm_df.to_json(orient="split"), t_json, det_product, agent_id_setting)

    st.plotly_chart(detection_summary_table(results), use_container_width=True)

    for r in results:
        st.divider()
        conf_label = "HIGH" if r.confidence >= 0.7 else "MED" if r.confidence >= 0.4 else "LOW"
        with st.expander(f"{r.detector_name} — Confidence: {r.confidence:.0%} ({conf_label})"):
            st.info(f"**Pattern:** {r.pattern_description}")
            st.warning(f"**Counter-Strategy:** {r.counter_strategy}")
            pp = norm_df[norm_df["product"] == det_product] if "product" in norm_df.columns else norm_df
            if not pp.empty and r.flagged_timestamps:
                st.plotly_chart(bot_detection_figure(r, pp), use_container_width=True)
            else:
                st.json(r.evidence)


# ── Tab 3: Quant Signals ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Quantitative Signal Panels")
    module = st.radio("Module", ["Market Making", "Stat Arb", "Volatility"], horizontal=True)

    if module == "Market Making":
        mm_product = st.selectbox("Product", products, key="mm_product")
        fast = st.slider("EMA Fast", 3, 50, 12)
        slow = st.slider("EMA Slow", 10, 100, 30)
        pos_lim = st.slider("Position Limit", 10, 200, POSITION_LIMITS.get(mm_product, DEFAULT_LIMIT))
        pp = norm_df[norm_df["product"] == mm_product] if "product" in norm_df.columns else norm_df
        if not pp.empty:
            st.plotly_chart(mm_panel(pp, fast=fast, slow=slow, pos_limit=pos_lim), use_container_width=True)

    elif module == "Stat Arb":
        if len(products) < 2:
            st.warning("Stat Arb requires at least 2 products.")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                asset_a = st.selectbox("Asset A", products, index=0)
            with col2:
                asset_b = st.selectbox("Asset B", products, index=min(1, len(products) - 1))
            with col3:
                use_coint = st.checkbox("Auto-Discover Beta", value=False, help="Runs Engle-Granger test to find optimal hedge ratio")
            
            window = st.slider("Z-Score Window", 20, 500, 100)
            
            h_ratio = None
            if use_coint and asset_a != asset_b:
                from framework.quant.forensics import AssetForensics
                from framework.quant.statarb import calculate_ou_half_life
                
                pivot = norm_df.pivot_table(index="global_time", columns="product", values="mid_price").ffill()
                if asset_a in pivot.columns and asset_b in pivot.columns:
                    res = AssetForensics.engle_granger_cointegration(pivot[asset_a], pivot[asset_b])
                    if res['is_cointegrated']:
                        spread_for_hl = pair_spread(norm_df, asset_a, asset_b, hedge_ratio=res['hedge_ratio'])
                        half_life = calculate_ou_half_life(spread_for_hl)
                        hl_text = f" | Half-life: {half_life:.1f} ticks" if half_life > 0 else ""
                        st.success(f"Cointegration found! (p={res['p_value']:.4f}){hl_text}")
                    else:
                        st.warning(f"No cointegration (p={res['p_value']:.4f}). Using OLS beta.")
                    h_ratio = res['hedge_ratio']

            if asset_a != asset_b:
                st.plotly_chart(statarb_panel(norm_df, asset_a, asset_b, window=window, hedge_ratio=h_ratio), use_container_width=True)
            else:
                st.warning("Select two different assets.")

    else:  # Volatility
        vol_product = st.selectbox("Product", products, key="vol_product")
        pp = norm_df[norm_df["product"] == vol_product] if "product" in norm_df.columns else norm_df
        if not pp.empty:
            st.plotly_chart(vol_panel(pp, vol_product), use_container_width=True)


# ── Tab 4: Arbitrage Analysis ───────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Arbitrage Opportunity Scanners")
    arb_module = st.radio("Arbitrage Module", ["Basket/ETF Arbitrage", "Volatility Arbitrage"], horizontal=True)

    if arb_module == "Basket/ETF Arbitrage":
        from framework.arbitrage.basket import DEFAULT_BASKETS

        avail_prods = sorted(norm_df["product"].dropna().unique().tolist())

        # Check which predefined baskets are fully covered by loaded products
        valid_predef = [
            b for b, bdef in DEFAULT_BASKETS.items()
            if all(c in avail_prods for c in bdef["constituents"])
        ]

        default_mode = "Predefined" if valid_predef else "Custom (from loaded products)"
        basket_mode = st.radio(
            "Basket Definition",
            ["Predefined", "Custom (from loaded products)"],
            index=0 if valid_predef else 1,
            horizontal=True,
            key="arb_basket_mode",
        )

        if basket_mode == "Predefined":
            if not valid_predef:
                st.warning("Nessun basket predefinito copre i prodotti caricati. Usa la modalità Custom.")
            else:
                sel_basket = st.selectbox("Basket Asset", valid_predef, key="arb_predef_basket")
                basket_def = DEFAULT_BASKETS[sel_basket]
                st.json(basket_def, expanded=False)
                st.plotly_chart(basket_arb_panel(norm_df, sel_basket, basket_def), use_container_width=True)

        else:  # Custom builder
            st.markdown("#### Definisci il tuo basket dai prodotti caricati")
            c1, c2 = st.columns(2)
            with c1:
                etf_asset = st.selectbox("ETF / Basket product", avail_prods, key="arb_etf")
            with c2:
                custom_fee = st.number_input("Fee / offset teorico", value=0, step=1, key="arb_fee")

            remaining = [p for p in avail_prods if p != etf_asset]
            constituents_sel = st.multiselect("Prodotti componenti", remaining, key="arb_constituents")

            weights: dict = {}
            if constituents_sel:
                cols = st.columns(min(len(constituents_sel), 5))
                for i, prod in enumerate(constituents_sel):
                    with cols[i % len(cols)]:
                        weights[prod] = st.number_input(f"Peso: {prod}", value=1, min_value=1,
                                                         step=1, key=f"arb_wt_{prod}")

            if constituents_sel and weights:
                custom_def = {"constituents": weights, "etf_fee": int(custom_fee)}
                st.json(custom_def, expanded=False)
                st.plotly_chart(basket_arb_panel(norm_df, etf_asset, custom_def), use_container_width=True)
            else:
                st.info("Seleziona almeno un prodotto componente per avviare l'analisi.")

    elif arb_module == "Volatility Arbitrage":
        vol_arb_product = st.selectbox("Select Asset for Volatility Analysis", products, key="vol_arb_prod")
        pp = norm_df[norm_df["product"] == vol_arb_product]
        if not pp.empty:
            st.plotly_chart(vol_arb_panel(pp), use_container_width=True)

# ── Tab 5: Asset Forensics ────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Asset Forensics & DNA Profiling")
    
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        st.markdown("### Forensic Tools")
        f_asset = st.selectbox("Target Asset", products, key="f_asset")
        
        if st.button("🔍 Run DNA Discovery (/discover)"):
            with st.spinner("Analyzing Asset DNA..."):
                result = handle_discover(norm_df, f_asset)
            st.code(result, language="text")
            
        if st.button("⚖️ Run Impact Test (/impact-test)"):
            with st.spinner("Calculating Market Impact..."):
                result = handle_impact_test(norm_df, f_asset)
            st.code(result, language="text")
            
        if st.button("🧬 Cross-Asset Co-integration (/forensic-report)"):
            with st.spinner("Running Engle-Granger tests..."):
                result = handle_forensic_report(norm_df)
            st.code(result, language="text")

    with f_col2:
        st.markdown("### Metrics Deep-Dive")
        from framework.quant.forensics import AssetForensics
        pp = norm_df[norm_df["product"] == f_asset] if "product" in norm_df.columns else norm_df
        if not pp.empty:
            forensics = AssetForensics(pp)
            
            f_sub_col1, f_sub_col2 = st.columns(2)
            with f_sub_col1:
                moments = forensics.return_moments()
                st.metric("Skewness", f"{moments['skewness']:.4f}")
                st.metric("Kurtosis", f"{moments['kurtosis']:.4f}")
                
                fft = forensics.dominant_frequency_fft()
                st.metric("Dominant Cycle", f"{fft['dominant_period_ticks']:.1f} ticks")
                
            with f_sub_col2:
                try:
                    hurst = forensics.rolling_hurst_exponent(window=200).iloc[-1]
                except IndexError:
                    hurst = 0.5
                st.metric("Hurst Exponent", f"{hurst:.3f}", help="<0.5 MR, >0.5 Trend")
                
                obi = forensics.order_book_imbalance().mean()
                st.metric("Avg OBI", f"{obi:.3f}", help="Order Book Imbalance")

            # Trade Flow Toxicity
            from framework.microstructure.toxicity import calculate_pin_proxy
            toxicity = calculate_pin_proxy(classified_trades[classified_trades['product'] == f_asset], window=60)
            if not toxicity.empty:
                st.markdown("**Trade Flow Toxicity (PIN Proxy)**")
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=toxicity.index, y=toxicity.rolling(10).mean(), name="Toxicity (smooth)", line=dict(color=PURPLE)))
                fig_t.update_layout(height=250, margin=dict(t=20, b=40), yaxis_title="PIN Proxy", paper_bgcolor="#0d0d0d", plot_bgcolor="#12122a", font=dict(color="#e8e8f0"))
                st.plotly_chart(fig_t, use_container_width=True)

            # Volatility Comparison
            adv_vol = forensics.advanced_volatility(bar_ticks=100)
            if not adv_vol.empty:
                st.markdown("**Advanced Volatility (Parkinson vs Garman-Klass)**")
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(y=adv_vol['parkinson_vol'], name="Parkinson", line=dict(color="#2ecc71")))
                fig_v.add_trace(go.Scatter(y=adv_vol['garman_klass_vol'], name="Garman-Klass", line=dict(color="#e74c3c")))
                fig_v.update_layout(height=300, margin=dict(t=20, b=20), paper_bgcolor="#0d0d0d", plot_bgcolor="#12122a", font=dict(color="#e8e8f0"))
                st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.info("Select an asset to see the deep-dive metrics.")


# ── Tab 6: ML Price Predictor ─────────────────────────────────────────────────
with tabs[6]:
    st.subheader("🤖 Machine Learning Price Predictor")
    
    ml_col1, ml_col2 = st.columns([1, 2])
    with ml_col1:
        st.markdown("#### Model Configuration")
        ml_asset = st.selectbox("Select Asset to Model", products, key="ml_asset")
        ml_horizon = st.slider("Prediction Horizon (ticks)", 5, 50, 10)
        
        if st.button("🚀 Train & Evaluate Model"):
            pp = norm_df[norm_df["product"] == ml_asset]
            if pp.empty:
                st.warning("No data for the selected asset.")
            else:
                from framework.ml.predictor import PricePredictor
                predictor = PricePredictor(pp, horizon=ml_horizon)
                
                with st.spinner(f"Training LightGBM model for {ml_asset}..."):
                    report, f_importance, predictions = predictor.train_and_evaluate()

                if report is None:
                    st.error(f"Training failed: {f_importance}")
                else:
                    st.session_state['ml_report'] = report
                    st.session_state['ml_importance'] = f_importance
                    st.session_state['ml_predictions'] = predictions
                    st.success("Modello allenato con successo!")
                
    with ml_col2:
        st.markdown("#### Model Evaluation")
        if 'ml_report' in st.session_state and st.session_state['ml_report'] is not None:
            report = st.session_state['ml_report']
            asset = st.session_state['ml_asset']
            st.markdown(f"**Classification Report for {asset}**")

            acc = report['accuracy']
            up_precision = report.get('Up', {}).get('precision', 0.0)
            down_precision = report.get('Down', {}).get('precision', 0.0)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Overall Accuracy", f"{acc:.2%}")
            m_col2.metric("'Up' Precision", f"{up_precision:.2%}")
            m_col3.metric("'Down' Precision", f"{down_precision:.2%}")

            # Mostra importanza delle feature
            st.markdown("**Feature Importance**")
            st.dataframe(st.session_state['ml_importance'])

    if 'ml_predictions' in st.session_state and st.session_state['ml_predictions'] is not None:
        st.markdown("---")
        st.markdown("### Prediction Visualization")
        preds_df = st.session_state['ml_predictions'].copy().reset_index(drop=True)

        # Clean spikes and build sequential index
        preds_df["mid_price"] = preds_df["mid_price"].replace(0, np.nan).ffill()
        mid_smooth = preds_df["mid_price"].rolling(30, min_periods=1).mean()
        x_seq = preds_df.index

        correct_up   = preds_df[(preds_df['prediction'] == 2) & (preds_df['target'] == 2)]
        correct_down = preds_df[(preds_df['prediction'] == 0) & (preds_df['target'] == 0)]

        fig_preds = go.Figure()
        fig_preds.add_trace(go.Scatter(x=x_seq, y=preds_df["mid_price"], name="Mid (raw)",
            line=dict(color="rgba(232,232,240,0.2)", width=1), connectgaps=True))
        fig_preds.add_trace(go.Scatter(x=x_seq, y=mid_smooth, name="Mid Smooth (30t)",
            line=dict(color=TEXT, width=1.8), connectgaps=True))
        fig_preds.add_trace(go.Scatter(x=correct_up.index, y=correct_up["mid_price"], mode="markers",
            name="Correct Up", marker=dict(symbol="triangle-up", color=GREEN, size=8)))
        fig_preds.add_trace(go.Scatter(x=correct_down.index, y=correct_down["mid_price"], mode="markers",
            name="Correct Down", marker=dict(symbol="triangle-down", color=RED, size=8)))
        fig_preds.update_layout(
            height=500, title="Model Predictions vs. Actual Price",
            xaxis_title="Tick (sequential)",
            paper_bgcolor=DARK_BG, plot_bgcolor=PANEL, font=dict(color=TEXT),
            hovermode="x unified",
        )
        st.plotly_chart(fig_preds, use_container_width=True)


# ── Tab 7: Data Explorer ──────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Normalized Data Explorer")
    exp_product = st.selectbox("Product (blank = all)", ["ALL"] + products, key="exp_product")
    view_df = norm_df if exp_product == "ALL" else norm_df[norm_df["product"] == exp_product]

    st.write(f"{len(view_df):,} rows × {len(view_df.columns)} columns")
    st.write(view_df.describe(include="all").round(4))
    st.dataframe(view_df.head(500), use_container_width=True)

    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"normalized_{exp_product}.csv",
        mime="text/csv",
    )

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from framework.arbitrage.basket import BasketArbitrage
from framework.quant.forensics import AssetForensics

DARK_BG = "#0d0d0d"
PANEL = "#12122a"
TEXT = "#e8e8f0"
GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"
YELLOW = "#f1c40f"
PURPLE = "#9b59b6"


def _seq_x(series_or_index, gt_to_seq: dict) -> np.ndarray:
    """Map a global_time index → sequential integers."""
    return np.array([gt_to_seq.get(gt, np.nan) for gt in series_or_index], dtype=float)


def basket_arb_panel(df: pd.DataFrame, basket_asset: str, basket_definition: dict) -> go.Figure:
    """
    Basket/ETF arbitrage on a sequential time axis:
      Row 1: Basket price vs NAV
      Row 2: Spread (Premium/Discount)
      Row 3: Spread Z-Score
    """
    arb = BasketArbitrage(df, basket_asset, basket_definition)
    result_df = arb.calculate_nav_and_spread()

    if result_df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Dati insufficienti per il basket {basket_asset}",
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Sequential index — result_df is already sorted by global_time from BasketArbitrage
    result_df = result_df.sort_values("global_time").reset_index(drop=True)

    # Clean price spikes
    for c in ("basket_price", "nav"):
        result_df[c] = result_df[c].replace(0, np.nan)
    result_df = result_df.ffill()

    x = result_df.index  # sequential

    # Day boundaries (propagate 'day' from original df if available)
    day_boundaries: list[float] = []
    if "day" in df.columns:
        day_map = (df[["global_time", "day"]].drop_duplicates("global_time")
                   .set_index("global_time")["day"].to_dict())
        result_df["_day"] = result_df["global_time"].map(day_map)
        if result_df["_day"].nunique() > 1:
            days = sorted(result_df["_day"].dropna().unique())
            for i in range(len(days) - 1):
                last_idx = int(result_df[result_df["_day"] == days[i]].index.max())
                day_boundaries.append(last_idx + 0.5)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.1,
        subplot_titles=["Basket Price vs. Net Asset Value (NAV)",
                        "Spread (Premium/Discount)", "Spread Z-Score"],
    )

    fig.add_trace(go.Scatter(x=x, y=result_df["basket_price"], name=f"{basket_asset} Price",
        line=dict(color=TEXT, width=2), connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=result_df["nav"], name="Net Asset Value (NAV)",
        line=dict(color=YELLOW, width=1.5, dash="dash"), connectgaps=True), row=1, col=1)

    spread_colors = [GREEN if v >= 0 else RED for v in result_df["spread"].fillna(0)]
    fig.add_trace(go.Bar(x=x, y=result_df["spread"], marker_color=spread_colors, name="Spread"),
        row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=result_df["spread_zscore"], name="Z-Score",
        line=dict(color=PURPLE, width=1.5), connectgaps=True), row=3, col=1)
    for level in [2.0, -2.0, 0.0]:
        fig.add_hline(y=level, line_dash="dash",
                      line_color=RED if abs(level) > 1 else TEXT, line_width=1, row=3, col=1)

    for b in day_boundaries:
        for row in range(1, 4):
            fig.add_vline(x=b, line_dash="dash", line_color="rgba(150,150,150,0.35)", row=row, col=1)

    fig.update_layout(
        title=f"Basket Arbitrage Analysis: {basket_asset}", height=850,
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL, font=dict(color=TEXT),
        hovermode="x unified", xaxis3=dict(title="Tick (sequential)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=80, b=50, l=50, r=50),
    )
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="#1e1e3a", row=i, col=1)
        fig.update_xaxes(gridcolor="#1e1e3a", row=i, col=1)
    return fig


def vol_arb_panel(df: pd.DataFrame) -> go.Figure:
    """
    Volatility arbitrage on a sequential time axis:
      Row 1: Historical vol vs Garman-Klass vol
      Row 2: Volatility spread
    """
    df = df.sort_values("global_time").reset_index(drop=True)
    for c in ("mid_price",):
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    df = df.ffill()

    forensics = AssetForensics(df)
    hist_vol = df["mid_price"].pct_change().rolling(100).std() * (100 ** 0.5)
    adv_vol = forensics.advanced_volatility(bar_ticks=10)

    x_price = df.index  # sequential

    # Day boundaries
    day_boundaries: list[float] = []
    if "day" in df.columns and df["day"].nunique() > 1:
        days = sorted(df["day"].dropna().unique())
        for i in range(len(days) - 1):
            last_idx = int(df[df["day"] == days[i]].index.max())
            day_boundaries.append(last_idx + 0.5)

    # adv_vol is indexed by bar_id (0,1,2,…); map to price sequential space
    # Each bar covers bar_ticks=10 price rows, so bar i → price rows [i*10, (i+1)*10)
    bar_ticks = 10
    x_bars = np.array([i * bar_ticks + bar_ticks // 2 for i in adv_vol.index], dtype=float)
    x_bars = x_bars.clip(0, len(df) - 1)

    # Align adv_vol to price df index for spread computation
    adv_vol_aligned = adv_vol.reindex(df.index // bar_ticks).ffill()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.15,
        subplot_titles=["Volatility Estimators Comparison",
                        "Volatility Spread (Garman-Klass - Historical)"],
    )

    fig.add_trace(go.Scatter(x=x_price, y=hist_vol, name="Historical Vol (Rolling Std)",
        line=dict(color=BLUE, width=1.5), connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_bars, y=adv_vol["garman_klass_vol"].values,
        name="Garman-Klass Vol", line=dict(color=YELLOW, width=1.5),
        mode="lines", connectgaps=True), row=1, col=1)

    gk_aligned = adv_vol_aligned["garman_klass_vol"] if "garman_klass_vol" in adv_vol_aligned.columns else pd.Series(np.nan, index=df.index)
    vol_spread = gk_aligned.values - hist_vol.values
    spread_colors = [GREEN if v >= 0 else RED for v in np.nan_to_num(vol_spread)]
    fig.add_trace(go.Bar(x=x_price, y=vol_spread, marker_color=spread_colors,
        name="Volatility Spread"), row=2, col=1)

    for b in day_boundaries:
        for row in range(1, 3):
            fig.add_vline(x=b, line_dash="dash", line_color="rgba(150,150,150,0.35)", row=row, col=1)

    product_name = df["product"].iloc[0] if "product" in df.columns else ""
    fig.update_layout(
        title=f"Volatility Arbitrage Analysis: {product_name}", height=850,
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL, font=dict(color=TEXT),
        hovermode="x unified", xaxis2=dict(title="Tick (sequential)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=80, b=50, l=50, r=50),
    )
    for i in range(1, 3):
        fig.update_yaxes(gridcolor="#1e1e3a", row=i, col=1)
        fig.update_xaxes(gridcolor="#1e1e3a", row=i, col=1)
    return fig

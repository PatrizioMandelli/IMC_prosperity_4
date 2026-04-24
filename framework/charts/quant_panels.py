import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from framework.quant.signals import ema_fair_value, zscore_signal
from framework.quant.statarb import pair_spread, spread_zscore

DARK_BG = "#0d0d0d"
PANEL = "#12122a"
TEXT = "#e8e8f0"
GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"
YELLOW = "#f1c40f"
PURPLE = "#9b59b6"


def _prep_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Sort, clean zero-spikes, reset to sequential index. Returns (df, day_boundaries)."""
    df = df.sort_values("global_time").reset_index(drop=True)
    for c in ("mid_price", "micro_price", "bid_price_1", "ask_price_1",
              "bid_price_2", "ask_price_2", "bid_price_3", "ask_price_3"):
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    df = df.ffill()

    boundaries: list[float] = []
    if "day" in df.columns and df["day"].nunique() > 1:
        days = sorted(df["day"].dropna().unique())
        for i in range(len(days) - 1):
            last_idx = int(df[df["day"] == days[i]].index.max())
            boundaries.append(last_idx + 0.5)
    return df, boundaries


def mm_panel(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 30,
    pos_limit: int = 80,
) -> go.Figure:
    """Market Making panel: price + EMAs, deviation z-score, position pressure."""
    df, day_boundaries = _prep_df(df)
    df = ema_fair_value(df, fast=fast, slow=slow)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.12,
        subplot_titles=["Price + EMA Fair Value", "Deviation Z-Score", "Position Pressure"],
    )

    x = df.index  # sequential — no inter-day gaps
    smooth = df["mid_price"].rolling(30, min_periods=1).mean()

    # Row 1: faint raw mid + smooth mid + EMAs
    fig.add_trace(go.Scatter(x=x, y=df["mid_price"], mode="lines", name="Mid (raw)",
        line=dict(color="rgba(232,232,240,0.2)", width=1), connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=smooth, mode="lines", name="Mid Smooth (30t)",
        line=dict(color=TEXT, width=1.8), connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["ema_fast"], mode="lines", name=f"EMA{fast}",
        line=dict(color=GREEN, width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["ema_slow"], mode="lines", name=f"EMA{slow}",
        line=dict(color=RED, width=1, dash="dot")), row=1, col=1)

    # Row 2: deviation z-score bars
    dz = df["deviation_z"].fillna(0)
    bar_colors = [GREEN if v >= 0 else RED for v in dz]
    fig.add_trace(go.Bar(x=x, y=dz, marker_color=bar_colors, name="Dev Z-Score"), row=2, col=1)
    for level in [2, -2]:
        fig.add_hline(y=level, line_dash="dash", line_color=YELLOW, line_width=0.8, row=2, col=1)

    # Row 3: position pressure
    if "pos_smooth" in df.columns:
        pressure = df["pos_smooth"].fillna(0) / pos_limit
        fig.add_trace(go.Scatter(x=x, y=pressure, mode="lines", name="Position Pressure",
            line=dict(color=PURPLE, width=1.5)), row=3, col=1)
        for level in [1, -1, 0.8, -0.8]:
            fig.add_hline(y=level, line_dash="dot",
                line_color=RED if abs(level) == 1 else YELLOW, line_width=0.7, row=3, col=1)

    _apply_dark_layout(fig, f"Market Making Signals (fast={fast}, slow={slow})", day_boundaries, n_rows=3)
    return fig


def statarb_panel(
    df: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = 100,
    hedge_ratio: float | None = None,
) -> go.Figure:
    """Stat Arb panel: normalized prices, spread, z-score — all on sequential x-axis."""
    if "product" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Multi-product DataFrame required", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    prods = df["product"].unique()
    if asset_a not in prods or asset_b not in prods:
        fig = go.Figure()
        fig.add_annotation(text=f"Products {asset_a} and/or {asset_b} not found",
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Clean spikes on full multi-product df before pivot
    df = df.sort_values("global_time").copy()
    for c in ("mid_price", "micro_price"):
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    df = df.ffill()

    # Spread and z (indexed on global_time)
    spread = pair_spread(df, asset_a, asset_b, hedge_ratio=hedge_ratio)
    z = spread_zscore(spread, window=window)

    # Pivot for normalized prices
    pivot = df.pivot_table(index="global_time", columns="product", values="mid_price").ffill()
    std_a = pivot[asset_a].std() or 1.0
    std_b = pivot[asset_b].std() or 1.0
    norm_a = (pivot[asset_a] - pivot[asset_a].mean()) / std_a
    norm_b = (pivot[asset_b] - pivot[asset_b].mean()) / std_b

    # Map global_time → sequential pivot row index (eliminates inter-day gaps)
    gt_to_seq = {gt: i for i, gt in enumerate(pivot.index)}
    x_norm = np.arange(len(pivot))
    x_sp = np.array([gt_to_seq.get(gt, np.nan) for gt in spread.index], dtype=float)
    x_z  = np.array([gt_to_seq.get(gt, np.nan) for gt in z.index],      dtype=float)

    # Day boundaries in pivot sequential space
    day_col = (df[["global_time", "day"]].drop_duplicates("global_time")
               .sort_values("global_time").reset_index(drop=True))
    day_boundaries: list[float] = []
    if "day" in day_col.columns and day_col["day"].nunique() > 1:
        days = sorted(day_col["day"].dropna().unique())
        for i in range(len(days) - 1):
            last_idx = int(day_col[day_col["day"] == days[i]].index.max())
            day_boundaries.append(last_idx + 0.5)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.35, 0.3], vertical_spacing=0.12,
        subplot_titles=["Normalized Prices", f"Spread ({asset_a} vs {asset_b})", "Spread Z-Score"],
    )

    fig.add_trace(go.Scatter(x=x_norm, y=norm_a.values, mode="lines", name=asset_a,
        line=dict(color=BLUE, width=1.5), connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_norm, y=norm_b.values, mode="lines", name=asset_b,
        line=dict(color=YELLOW, width=1.5), connectgaps=True), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_sp, y=spread.values, mode="lines", name="Spread",
        line=dict(color=PURPLE, width=1.5), connectgaps=True), row=2, col=1)

    fig.add_trace(go.Scatter(x=x_z, y=z.values, mode="lines", name="Z-Score",
        line=dict(color=TEXT, width=1), connectgaps=True), row=3, col=1)
    for level, color in [(2, RED), (-2, RED), (0, TEXT)]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, line_width=0.8, row=3, col=1)

    hr_label = f" (beta={hedge_ratio:.3f})" if hedge_ratio else " (OLS fit)"
    _apply_dark_layout(fig, f"Stat Arb: {asset_a} vs {asset_b}{hr_label}", day_boundaries, n_rows=3)
    return fig


def vol_panel(df: pd.DataFrame, product: str) -> go.Figure:
    """Volatility panel: realized vol (multi-window) + regime bars."""
    if "product" in df.columns:
        df = df[df["product"] == product]
    df, day_boundaries = _prep_df(df)
    returns = df["mid_price"].pct_change()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.12,
        subplot_titles=[f"Realized Volatility — {product}", "Volatility Regime"],
    )

    x = df.index
    for w, color in [(20, GREEN), (50, YELLOW), (100, RED)]:
        rvol = returns.rolling(w, min_periods=5).std() * np.sqrt(w)
        fig.add_trace(go.Scatter(x=x, y=rvol, mode="lines", name=f"RVol-{w}",
            line=dict(color=color, width=1)), row=1, col=1)

    rvol50 = returns.rolling(50, min_periods=5).std()
    q33, q67 = rvol50.quantile(0.33), rvol50.quantile(0.67)
    regime = pd.cut(rvol50, bins=[-np.inf, q33, q67, np.inf], labels=[0, 0.5, 1])
    regime_colors = [GREEN if v == 0 else YELLOW if v == 0.5 else RED for v in regime]
    fig.add_trace(go.Bar(x=x, y=regime.astype(float), marker_color=regime_colors,
        name="Regime (0=low, 1=high)"), row=2, col=1)

    _apply_dark_layout(fig, f"Volatility Analysis — {product}", day_boundaries, n_rows=2)
    return fig


def _apply_dark_layout(fig: go.Figure, title: str, day_boundaries: list | None = None, n_rows: int = 3):
    fig.update_layout(
        title=title, height=850,
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(t=80, b=50, l=50, r=50),
        xaxis=dict(title="Tick (sequential)"),
    )
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].gridcolor = "#1e1e3a"
    if day_boundaries:
        for b in day_boundaries:
            for row in range(1, n_rows + 1):
                fig.add_vline(x=b, line_dash="dash",
                              line_color="rgba(150,150,150,0.35)", row=row, col=1)

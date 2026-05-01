import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK_BG = "#0d0d0d"
PANEL   = "#12122a"
TEXT    = "#e8e8f0"
GREEN   = "#2ecc71"
RED     = "#e74c3c"


def lob_heatmap(
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame | None = None,
    product: str | None = None,
    time_bins: int = 200,
    price_bins: int = 50,
) -> go.Figure:
    """
    3-row interactive LOB heatmap on a sequential (gap-free) time axis:
      Row 1 (60%): Heatmap of bid/ask volume density + micro_price + mid_price + trade markers
      Row 2 (20%): OFI bar chart
      Row 3 (20%): Spread + rolling mean
    """
    if product:
        prices_df = prices_df[prices_df["product"] == product]
        if trades_df is not None and not trades_df.empty:
            trades_df = trades_df[trades_df["product"] == product]

    df = prices_df.sort_values("global_time").reset_index(drop=True)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # ── Clean spikes ─────────────────────────────────────────────────────────────
    for c in ("mid_price", "micro_price", "spread",
              "bid_price_1", "ask_price_1", "bid_price_2", "ask_price_2",
              "bid_price_3", "ask_price_3"):
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    df = df.ffill()

    # Sequential index (row number) used as x-axis everywhere
    df["_seq"] = df.index.astype(float)

    # Day boundaries in sequential space
    day_boundaries: list[float] = []
    if "day" in df.columns and df["day"].nunique() > 1:
        days = sorted(df["day"].dropna().unique())
        for i in range(len(days) - 1):
            last_idx = int(df[df["day"] == days[i]].index.max())
            day_boundaries.append(last_idx + 0.5)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=["LOB Heatmap + Micro-Price", "Order Flow Imbalance", "Bid-Ask Spread"],
    )

    # ── Row 1: Heatmap ───────────────────────────────────────────────────────────
    z, t_edges, p_edges = _pivot_lob_volume(df, time_bins, price_bins)
    fig.add_trace(
        go.Heatmap(z=z.T, x=t_edges, y=p_edges, colorscale="Plasma", zsmooth="best",
                   showscale=True, colorbar=dict(len=0.55, y=0.72, thickness=10, title="Volume"),
                   name="LOB Depth",
                   hovertemplate="Seq Tick: %{x}<br>Price: %{y}<br>Volume: %{z}<extra></extra>"),
        row=1, col=1,
    )

    # Smooth mid-price overlay (30-tick rolling mean, no spikes)
    mid_smooth = df["mid_price"].rolling(30, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(x=df["_seq"], y=mid_smooth, mode="lines",
                   line=dict(color="rgba(180,180,180,0.7)", width=1.5, dash="dot"),
                   name="Mid Smooth", connectgaps=True),
        row=1, col=1,
    )

    # Micro-price overlay
    if "micro_price" in df.columns:
        micro_smooth = df["micro_price"].rolling(20, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=df["_seq"], y=micro_smooth, mode="lines",
                       line=dict(color="white", width=1.5), name="Micro-Price (smooth)",
                       connectgaps=True),
            row=1, col=1,
        )

    # Trade markers mapped to sequential positions — direction from dist_from_mid
    if trades_df is not None and not trades_df.empty and "maker_taker" in trades_df.columns:
        gt_arr = df["global_time"].values
        takers = trades_df[trades_df["maker_taker"] == "TAKER"]
        if not takers.empty and "price" in takers.columns and "dist_from_mid" in takers.columns:
            buys  = takers[takers["dist_from_mid"] > 0]
            sells = takers[takers["dist_from_mid"] < 0]
            for sub, sym, color, label in [
                (buys,  "triangle-up",   GREEN, "Taker Buy"),
                (sells, "triangle-down", RED,   "Taker Sell"),
            ]:
                if not sub.empty:
                    seq_pos = np.searchsorted(gt_arr, sub["global_time"].values).clip(0, len(gt_arr) - 1)
                    fig.add_trace(
                        go.Scatter(x=seq_pos, y=sub["price"].values, mode="markers",
                                   marker=dict(symbol=sym, size=7, color=color, opacity=0.8),
                                   name=label,
                                   hovertemplate=f"{label}<br>SeqTick: %{{x}}<br>Price: %{{y}}<extra></extra>"),
                        row=1, col=1,
                    )
        elif not takers.empty and "price" in takers.columns:
            # fallback when dist_from_mid not available
            seq_pos = np.searchsorted(gt_arr, takers["global_time"].values).clip(0, len(gt_arr) - 1)
            fig.add_trace(
                go.Scatter(x=seq_pos, y=takers["price"].values, mode="markers",
                           marker=dict(symbol="circle", size=5, color=RED, opacity=0.7),
                           name="Taker trade",
                           hovertemplate="Taker<br>SeqTick: %{x}<br>Price: %{y}<extra></extra>"),
                row=1, col=1,
            )

    # Day boundary lines on all rows
    for b in day_boundaries:
        for row in range(1, 4):
            fig.add_vline(x=b, line_dash="dash", line_color="rgba(150,150,150,0.4)", row=row, col=1)

    # ── Row 2: OFI ───────────────────────────────────────────────────────────────
    if "ofi" in df.columns:
        ofi = df["ofi"].fillna(0)
        colors = [GREEN if v >= 0 else RED for v in ofi]
        fig.add_trace(
            go.Bar(x=df["_seq"], y=ofi, marker_color=colors, name="OFI",
                   hovertemplate="OFI: %{y:.3f}<extra></extra>"),
            row=2, col=1,
        )

    # ── Row 3: Spread ────────────────────────────────────────────────────────────
    if "spread" in df.columns:
        spread = df["spread"].ffill()
        roll_mean = spread.rolling(30, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=df["_seq"], y=spread, mode="lines",
                       line=dict(color="#3498db", width=1), name="Spread"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["_seq"], y=roll_mean, mode="lines",
                       line=dict(color="#f1c40f", width=1.5, dash="dash"),
                       name="Spread (30t MA)"),
            row=3, col=1,
        )

    # ── Layout ───────────────────────────────────────────────────────────────────
    title = f"LOB Heatmap — {product}" if product else "LOB Heatmap"
    fig.update_layout(
        title=title, height=800,
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis3=dict(title="Tick (sequential)"),
    )
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="#1e1e3a", row=i, col=1)
        fig.update_xaxes(gridcolor="#1e1e3a", row=i, col=1)

    return fig


def _pivot_lob_volume(
    df: pd.DataFrame,
    time_bins: int,
    price_bins: int,
    time_col: str = "_seq",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (z_matrix [time x price], time_edges, price_edges).
    Uses sequential index as time axis; filters out zero/NaN prices.
    """
    records_t, records_p, records_v = [], [], []

    level_map = [
        ("bid_price_1", "bid_volume_1"), ("bid_price_2", "bid_volume_2"), ("bid_price_3", "bid_volume_3"),
        ("ask_price_1", "ask_volume_1"), ("ask_price_2", "ask_volume_2"), ("ask_price_3", "ask_volume_3"),
    ]

    for price_col, vol_col in level_map:
        if price_col in df.columns and vol_col in df.columns:
            # Filter: price must be positive and non-NaN, volume must be positive
            mask = (df[price_col].notna() & df[vol_col].notna()
                    & (df[price_col] > 0) & (df[vol_col] > 0))
            records_t.extend(df.loc[mask, time_col].tolist())
            records_p.extend(df.loc[mask, price_col].tolist())
            records_v.extend(df.loc[mask, vol_col].tolist())

    t_range = (df[time_col].min(), df[time_col].max())
    mid_col = df["mid_price"].dropna()
    p_range = (mid_col.min(), mid_col.max()) if not mid_col.empty else (0, 1)

    if t_range[0] == t_range[1]:
        t_range = (t_range[0] - 1, t_range[1] + 1)
    if p_range[0] == p_range[1]:
        p_range = (p_range[0] - 1, p_range[1] + 1)

    if not records_t:
        t_edges = np.linspace(*t_range, time_bins + 1)
        p_edges = np.linspace(*p_range, price_bins + 1)
        return np.zeros((time_bins, price_bins)), t_edges[:-1], p_edges[:-1]

    p_arr = np.array(records_p)
    unique_prices = np.unique(p_arr[~np.isnan(p_arr)])
    actual_price_bins = min(price_bins, len(unique_prices))
    actual_p_range = (np.nanmin(p_arr), np.nanmax(p_arr))
    if actual_p_range[0] == actual_p_range[1]:
        actual_p_range = (actual_p_range[0] - 1, actual_p_range[1] + 1)

    z, t_edges, p_edges = np.histogram2d(
        np.array(records_t, dtype=float), p_arr,
        bins=[time_bins, actual_price_bins],
        range=[t_range, actual_p_range],
        weights=np.array(records_v, dtype=float),
    )

    return z, t_edges[:-1], p_edges[:-1]

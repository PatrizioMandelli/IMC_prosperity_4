import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from framework.detectors.base import DetectionResult

DARK_BG = "#0d0d0d"
PANEL = "#12122a"
TEXT = "#e8e8f0"
GREEN = "#2ecc71"
YELLOW = "#f1c40f"
RED = "#e74c3c"
BLUE = "#3498db"


def bot_detection_figure(result: DetectionResult, prices_df: pd.DataFrame) -> go.Figure:
    """
    2-row figure on a sequential (gap-free) time axis:
      Row 1: mid_price (raw faint + smooth) with flagged timestamp spans
      Row 2: confidence gauge | evidence table
    """
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.6, 0.4],
        column_widths=[0.7, 0.3],
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "table"}]],
        subplot_titles=[f"{result.detector_name} — {result.product}", "Confidence", "Evidence"],
        vertical_spacing=0.15,
    )

    # ── Sort, clean spikes, build sequential index ────────────────────────────────
    df = prices_df.sort_values("global_time").reset_index(drop=True)
    for c in ("mid_price", "bid_price_1", "ask_price_1"):
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    df = df.ffill()

    x_seq = df.index  # 0, 1, 2, … — no inter-day gaps
    mid_smooth = df["mid_price"].rolling(30, min_periods=1).mean()

    # Map flagged global_time values → sequential positions via searchsorted
    gt_arr = df["global_time"].values
    flagged_seq = np.searchsorted(gt_arr, result.flagged_timestamps[:100]).clip(0, len(gt_arr) - 1)

    # ── Row 1: Price traces ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=x_seq, y=df["mid_price"], mode="lines", name="Mid (raw)",
                   line=dict(color="rgba(52,152,219,0.25)", width=1), connectgaps=True),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_seq, y=mid_smooth, mode="lines", name="Mid Smooth (30t)",
                   line=dict(color=BLUE, width=1.8), connectgaps=True),
        row=1, col=1,
    )

    # Flagged spans in sequential space
    half_width = max(1, len(gt_arr) // 500)  # ~0.2% of the series width
    for seq_pos in flagged_seq:
        fig.add_vrect(
            x0=int(seq_pos) - half_width, x1=int(seq_pos) + half_width,
            fillcolor=RED, opacity=0.15, line_width=0,
            row=1, col=1,
        )

    # Day boundaries
    if "day" in df.columns and df["day"].nunique() > 1:
        days = sorted(df["day"].dropna().unique())
        for i in range(len(days) - 1):
            last_idx = int(df[df["day"] == days[i]].index.max())
            fig.add_vline(x=last_idx + 0.5, line_dash="dash",
                          line_color="rgba(150,150,150,0.4)", row=1, col=1)

    # ── Row 2 left: Confidence gauge ─────────────────────────────────────────────
    conf_color = GREEN if result.confidence < 0.4 else YELLOW if result.confidence < 0.7 else RED
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=result.confidence * 100,
            number={"suffix": "%", "font": {"color": conf_color}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=TEXT),
                bar=dict(color=conf_color),
                bgcolor=PANEL,
                steps=[
                    dict(range=[0, 40],  color="#1a1a2e"),
                    dict(range=[40, 70], color="#1e2a1e"),
                    dict(range=[70, 100], color="#2a1a1a"),
                ],
            ),
            title={"text": "Bot Confidence", "font": {"color": TEXT}},
        ),
        row=2, col=1,
    )

    # ── Row 2 right: Evidence table ──────────────────────────────────────────────
    ev = result.evidence
    keys = [str(k) for k in ev.keys()]
    vals = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in ev.values()]
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="#1a1a2e",
                        font=dict(color=TEXT, size=11), align="left"),
            cells=dict(values=[keys, vals], fill_color=PANEL,
                       font=dict(color=TEXT, size=10), align="left"),
        ),
        row=2, col=2,
    )

    fig.update_layout(
        height=700,
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT), showlegend=False,
        margin=dict(t=80, b=60, l=50, r=50),
        xaxis=dict(title="Tick (sequential)"),
        annotations=[dict(
            text=f"<b>Pattern:</b> {result.pattern_description[:120]}...",
            xref="paper", yref="paper", x=0, y=-0.04,
            showarrow=False, font=dict(color=TEXT, size=10), align="left",
        )],
    )
    return fig


def detection_summary_table(results: list[DetectionResult]) -> go.Figure:
    """Color-coded summary table of all DetectionResults."""
    rows = {"Detector": [], "Product": [], "Confidence": [], "Pattern": [], "Counter-Strategy": []}
    row_colors = []

    for r in results:
        rows["Detector"].append(r.detector_name)
        rows["Product"].append(r.product)
        rows["Confidence"].append(f"{r.confidence:.0%}")
        rows["Pattern"].append(r.pattern_description[:80] + "..." if len(r.pattern_description) > 80 else r.pattern_description)
        rows["Counter-Strategy"].append(r.counter_strategy[:80] + "..." if len(r.counter_strategy) > 80 else r.counter_strategy)
        c = r.confidence
        row_colors.append(
            "rgba(231,76,60,0.25)" if c >= 0.7 else
            "rgba(241,196,15,0.2)"  if c >= 0.4 else
            "rgba(46,204,113,0.15)"
        )

    fig = go.Figure(go.Table(
        header=dict(values=list(rows.keys()), fill_color="#1a1a2e",
                    font=dict(color=TEXT, size=12), align="left", height=30),
        cells=dict(values=list(rows.values()), fill_color=[row_colors] * len(rows),
                   font=dict(color=TEXT, size=11), align="left", height=28),
    ))
    fig.update_layout(
        title="Bot Detection Summary", paper_bgcolor=DARK_BG, font=dict(color=TEXT),
        height=max(200, 60 + 30 * len(results)), margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

"""
Prosperity Analyzer v2
======================
Usage: python prosperity_analyzer.py <path_to_json>

Generates 5 separate PNG files per product inside <stem>_analysis/:
  1_liquidity    — bid/ask depth (all 3 levels stacked) + one-sided flags
  2_pnl_per_tick — PnL delta per tick + cumulative PnL
  3_spread       — spread over time + rolling average + wide-spread highlights
  4_position     — implied position (smoothed)
  5_midprice     — mid price + best bid/ask (outliers removed)
"""

import json, csv, io, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
BG     = "#0d0d0d"
PANEL  = "#12122a"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
BLUE   = "#3498db"
PURPLE = "#9b59b6"
YELLOW = "#f1c40f"
DIM    = "#888899"
WHITE  = "#e8e8f0"

LARGE_PNL = 10
ROLL_W    = 30

# ── Load & parse ───────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

def parse(log_str):
    df = pd.DataFrame(list(csv.DictReader(io.StringIO(log_str), delimiter=";")))
    nums = ["day","timestamp",
            "bid_price_1","bid_volume_1","bid_price_2","bid_volume_2","bid_price_3","bid_volume_3",
            "ask_price_1","ask_volume_1","ask_price_2","ask_volume_2","ask_price_3","ask_volume_3",
            "mid_price","profit_and_loss"]
    for c in nums:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["day","timestamp"]).reset_index(drop=True)

def timeline(df):
    max_ts = df.groupby("day")["timestamp"].max().max() + 200
    return (df["day"] * max_ts + df["timestamp"]).values

def features(df):
    df = df.copy()
    # Remove outlier mid prices
    median = df["mid_price"].median()
    df.loc[df["mid_price"] < median * 0.5, "mid_price"] = np.nan
    df["mid_price"] = df["mid_price"].interpolate()

    df["bid_depth"] = df[["bid_volume_1","bid_volume_2","bid_volume_3"]].sum(axis=1, min_count=1)
    df["ask_depth"] = df[["ask_volume_1","ask_volume_2","ask_volume_3"]].sum(axis=1, min_count=1)
    df["bid_l1"] = df["bid_volume_1"].fillna(0)
    df["bid_l2"] = df["bid_volume_2"].fillna(0)
    df["bid_l3"] = df["bid_volume_3"].fillna(0)
    df["ask_l1"] = df["ask_volume_1"].fillna(0)
    df["ask_l2"] = df["ask_volume_2"].fillna(0)
    df["ask_l3"] = df["ask_volume_3"].fillna(0)
    df["one_sided"] = df["bid_depth"].eq(0) | df["ask_depth"].eq(0)

    df["spread"]      = df["ask_price_1"] - df["bid_price_1"]
    df["spread_roll"] = df["spread"].rolling(ROLL_W, min_periods=1).mean()

    df["pnl_delta"] = df["profit_and_loss"].diff().fillna(0)
    df["pnl_cum"]   = df["profit_and_loss"]
    df["big_gain"]  = df["pnl_delta"] >  LARGE_PNL
    df["big_loss"]  = df["pnl_delta"] < -LARGE_PNL

    df["mid_delta"] = df["mid_price"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(df["mid_delta"].abs() > 0.1,
                       df["pnl_delta"] / df["mid_delta"], np.nan)
    s = pd.Series(raw, index=df.index).interpolate(limit_direction="both")
    df["pos_smooth"] = s.rolling(ROLL_W, min_periods=1, center=True).mean()
    return df

# ── Helpers ────────────────────────────────────────────────────────────────────

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.grid(True, color="#1e1e3a", linewidth=0.5, alpha=0.8)
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a2a4a")

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path.name}")

# ── Panel 1: Liquidity ─────────────────────────────────────────────────────────

def plot_liquidity(df, t, product, out_dir):
    fig, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(16, 8), facecolor=BG, sharex=True)
    fig.suptitle(f"{product}  —  Liquidity: Bid & Ask Depth (stacked levels 1-2-3)",
                 color=WHITE, fontsize=11, fontweight="bold", y=0.99)

    for ax, label, l1, l2, l3, col, os_mask in [
        (ax_b, "BID", df["bid_l1"], df["bid_l2"], df["bid_l3"], GREEN,
         (df["bid_depth"].eq(0) & df["one_sided"])),
        (ax_a, "ASK", df["ask_l1"], df["ask_l2"], df["ask_l3"], RED,
         (df["ask_depth"].eq(0) & df["one_sided"])),
    ]:
        style_ax(ax)
        ax.fill_between(t, 0,               l1.values,               color=col, alpha=0.80, label="Level 1")
        ax.fill_between(t, l1.values,       (l1+l2).values,          color=col, alpha=0.45, label="Level 2")
        ax.fill_between(t, (l1+l2).values,  (l1+l2+l3).values,      color=col, alpha=0.20, label="Level 3")
        for xt in t[os_mask.values]:
            ax.axvline(xt, color=YELLOW, lw=0.5, alpha=0.6)
        ax.set_ylabel(f"{label} depth (units)", color=DIM, fontsize=8)
        leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor=WHITE)
        ax.text(0.01, 0.94, f"One-sided ticks: {os_mask.sum()}",
                transform=ax.transAxes, color=YELLOW, fontsize=7.5, va="top")

    ax_a.set_xlabel("Timestamp", color=DIM, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, out_dir / f"{product}_1_liquidity.png")

# ── Panel 2: PnL per tick ──────────────────────────────────────────────────────

def plot_pnl(df, t, product, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), facecolor=BG, sharex=True)
    fig.suptitle(f"{product}  —  PnL per Tick & Cumulative PnL",
                 color=WHITE, fontsize=11, fontweight="bold", y=0.99)

    # per-tick delta
    style_ax(ax1)
    deltas = df["pnl_delta"].values
    cols   = np.where(deltas >= 0, GREEN, RED)
    gaps   = np.diff(t)
    bw     = np.median(gaps[gaps > 0]) * 0.55 if len(gaps) else 100
    ax1.bar(t, deltas, width=bw, color=cols, alpha=0.85)
    ax1.axhline(0, color=DIM, lw=0.6)
    ax1.scatter(t[df["big_gain"].values], deltas[df["big_gain"].values],
                color=WHITE, marker="*", s=70, zorder=6, label=f"Gain > +{LARGE_PNL}")
    ax1.scatter(t[df["big_loss"].values],  deltas[df["big_loss"].values],
                color=YELLOW, marker="*", s=70, zorder=6, label=f"Loss < -{LARGE_PNL}")
    ax1.set_ylabel("PnL Δ per tick", color=DIM, fontsize=8)
    ax1.legend(fontsize=7.5, framealpha=0.3, labelcolor=WHITE)
    ax1.text(0.99, 0.97,
             f"Big gain ticks: {df['big_gain'].sum()}  |  Big loss ticks: {df['big_loss'].sum()}  |  Net: {deltas.sum():+.2f}",
             transform=ax1.transAxes, color=DIM, fontsize=7.5, ha="right", va="top")

    # cumulative
    style_ax(ax2)
    pnl = df["pnl_cum"].values
    c   = GREEN if pnl[-1] >= pnl[0] else RED
    ax2.plot(t, pnl, color=c, lw=1.4)
    ax2.fill_between(t, pnl[0], pnl, alpha=0.15, color=c)
    ax2.axhline(pnl[0], color=DIM, lw=0.5, ls="--")
    ax2.set_ylabel("Cumulative PnL", color=DIM, fontsize=8)
    ax2.set_xlabel("Timestamp", color=DIM, fontsize=8)
    ax2.text(0.99, 0.97, f"Final: {pnl[-1]:+.2f}",
             transform=ax2.transAxes, color=c, fontsize=8, ha="right", va="top", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, out_dir / f"{product}_2_pnl_per_tick.png")

# ── Panel 3: Spread ────────────────────────────────────────────────────────────

def plot_spread(df, t, product, out_dir):
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    fig.suptitle(f"{product}  —  Bid-Ask Spread", color=WHITE, fontsize=11, fontweight="bold")
    style_ax(ax)

    spread = df["spread"].values
    roll   = df["spread_roll"].values
    mu, sigma = np.nanmean(spread), np.nanstd(spread)
    wide   = spread > (mu + 2 * sigma)

    ax.fill_between(t, 0, spread, color=PURPLE, alpha=0.30)
    ax.plot(t, spread, color=PURPLE, lw=0.7, alpha=0.8)
    ax.plot(t, roll,   color=WHITE,  lw=1.2,  ls="--",
            label=f"Rolling mean ({ROLL_W}t) = {mu:.1f}")
    ax.axhline(mu,         color=DIM,    lw=0.6, ls=":",  label=f"Mean = {mu:.1f}")
    ax.axhline(mu+2*sigma, color=YELLOW, lw=0.6, ls=":",  label=f"μ+2σ = {mu+2*sigma:.1f}")
    ax.scatter(t[wide], spread[wide], color=YELLOW, s=20, zorder=5,
               label=f"Wide ticks (>{mu+2*sigma:.0f}): {wide.sum()}")

    ax.set_ylabel("Ticks  (ask₁ − bid₁)", color=DIM, fontsize=8)
    ax.set_xlabel("Timestamp", color=DIM, fontsize=8)
    ax.legend(fontsize=7.5, framealpha=0.3, labelcolor=WHITE)
    ax.text(0.99, 0.97,
            f"Min: {np.nanmin(spread):.0f}  Max: {np.nanmax(spread):.0f}  Std: {sigma:.1f}",
            transform=ax.transAxes, color=DIM, fontsize=7.5, ha="right", va="top")

    fig.tight_layout()
    save(fig, out_dir / f"{product}_3_spread.png")

# ── Panel 4: Implied Position ──────────────────────────────────────────────────

def plot_position(df, t, product, out_dir):
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    fig.suptitle(f"{product}  —  Implied Position  (PnL Δ / Mid Δ, smoothed {ROLL_W}t)",
                 color=WHITE, fontsize=11, fontweight="bold")
    style_ax(ax)

    pos = df["pos_smooth"].values
    ax.fill_between(t, 0, pos, where=pos >= 0, color=GREEN, alpha=0.25, label="Long")
    ax.fill_between(t, 0, pos, where=pos <  0, color=RED,   alpha=0.25, label="Short")
    ax.plot(t, pos, color=BLUE, lw=1.2)
    ax.axhline(0, color=DIM, lw=0.7, ls="--")

    imax = int(np.nanargmax(pos)); imin = int(np.nanargmin(pos))
    ax.annotate(f"{pos[imax]:.1f}", xy=(t[imax], pos[imax]),
                xytext=(t[imax], pos[imax] + abs(pos[imax])*0.2 + 0.5),
                color=GREEN, fontsize=7.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
    ax.annotate(f"{pos[imin]:.1f}", xy=(t[imin], pos[imin]),
                xytext=(t[imin], pos[imin] - abs(pos[imin])*0.2 - 0.5),
                color=RED, fontsize=7.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.7))

    ax.set_ylabel("Units (approx)", color=DIM, fontsize=8)
    ax.set_xlabel("Timestamp", color=DIM, fontsize=8)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor=WHITE)

    fig.tight_layout()
    save(fig, out_dir / f"{product}_4_position.png")

# ── Panel 5: Mid Price ─────────────────────────────────────────────────────────

def plot_midprice(df, t, product, out_dir):
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    fig.suptitle(f"{product}  —  Mid Price & Best Bid/Ask",
                 color=WHITE, fontsize=11, fontweight="bold")
    style_ax(ax)

    mid = df["mid_price"].values
    b1  = df["bid_price_1"].values
    a1  = df["ask_price_1"].values

    ax.fill_between(t, b1, a1, alpha=0.08, color=BLUE)
    ax.plot(t, mid, color=BLUE,  lw=1.3, label="Mid price")
    ax.plot(t, b1,  color=GREEN, lw=0.7, alpha=0.6, label="Best bid")
    ax.plot(t, a1,  color=RED,   lw=0.7, alpha=0.6, label="Best ask")
    ax.scatter(t[df["big_gain"].values], mid[df["big_gain"].values],
               color=GREEN, marker="^", s=25, zorder=5, label=f"Big gain (>{LARGE_PNL})")
    ax.scatter(t[df["big_loss"].values], mid[df["big_loss"].values],
               color=RED,   marker="v", s=25, zorder=5, label=f"Big loss (<-{LARGE_PNL})")

    ax.set_ylabel("Price", color=DIM, fontsize=8)
    ax.set_xlabel("Timestamp", color=DIM, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.legend(fontsize=7.5, framealpha=0.3, labelcolor=WHITE, ncol=3)

    fig.tight_layout()
    save(fig, out_dir / f"{product}_5_midprice.png")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python prosperity_analyzer.py <path_to_json>")
        sys.exit(1)

    data = load(sys.argv[1])
    print(f"\nRound {data.get('round','?')} | Status: {data.get('status','?')} | Profit: {data.get('profit','?')}\n")

    df_all   = parse(data["activitiesLog"])
    products = sorted(df_all["product"].unique())

    stem    = Path(sys.argv[1]).stem
    out_dir = Path(f"{stem}_analysis")
    out_dir.mkdir(exist_ok=True)
    print(f"Output → {out_dir}/\n")

    for product in products:
        print(f"[{product}]")
        df = df_all[df_all["product"] == product].reset_index(drop=True)
        df = features(df)
        t  = timeline(df)
        plot_liquidity(df, t, product, out_dir)
        plot_pnl      (df, t, product, out_dir)
        plot_spread   (df, t, product, out_dir)
        plot_position (df, t, product, out_dir)
        plot_midprice (df, t, product, out_dir)
        print()

    print("Done.")

if __name__ == "__main__":
    main()
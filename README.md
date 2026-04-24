
# IMC Prosperity 4 — Algorithmic Trading Bot

> **Current standing: top 5% globally** in IMC Prosperity 4 (2026 edition)

## What is Prosperity?

[IMC Prosperity](https://prosperity.imc.com/) is a global algorithmic trading competition run by IMC Trading. Participants write Python bots that trade fictional assets on a simulated limit-order-book exchange. The competition runs in rounds, each introducing new products with different price dynamics. Bots are evaluated on realized PnL across thousands of ticks per day, over multiple simulated trading days.

## Our Approach

Each product demands a distinct strategy. We analyze historical order book data, identify the dominant price dynamic (mean-reversion, trend, arbitrage), then build a stateful bot that adapts in real time.

### Round 1 & 2 Products

| Product | Dynamic | Strategy |
|---|---|---|
| `ASH_COATED_OSMIUM` | Mean-reverting around a fair value | Arbitrage taker + two-sided market making |
| `INTARIAN_PEPPER_ROOT` | Intraday uptrend with sudden traps | Trend-following with adaptive circuit breaker |

### Key algorithmic ideas

**Arbitrage + Market Making (Osmium)**  
The bot scans the order book for mispriced offers relative to a known fair value. It takes those fills first, then deploys the remaining position capacity as passive two-tier quotes (tight + deep) to capture the spread. Inventory pressure dynamically scales both layers.

**Adaptive Trend Following with Circuit Breaker (Pepper Root)**  
The bot tracks a linear price trend (slope estimated from historical snapshots) and holds a directional core position to ride it. A persistent circuit-breaker fires after 50 consecutive ticks where the live price deviates more than ±100 from the model — it re-estimates the slope from recent data and re-anchors, rather than blindly holding. An aggressive startup phase accumulates inventory quickly at the beginning of each day before the trend gains speed.

**Stateful memory across ticks**  
State (trend slope, day open price, broken-tick counter, price snapshots) is serialised to JSON via `traderData` and restored each tick, surviving the stateless execution model imposed by the platform.

## Repository Structure

```
my_bot/
  round1/          # Round 1 trader iterations
  round2/          # Round 2 traders (current submission)
    Main_scale_arbitrage_osmium_trader.py   # Active bot

analysis_tools/
  Order_book_analyzer.py     # 5-panel PNG report per product (liquidity, PnL, spread, position, midprice)
  Pepper_behaviour_analyzer.py
  baseline_std_analyzer.py
  data_explorer.py
  quick_plot.py

Manual_trading/
  static_book_filler.py      # Single-price auction simulator for manual round analysis

prosperity4bt/               # Local backtest engine (fork of prosperity4bt)
data/
  round1/                    # Historical prices & trades CSVs
  round2/
backtests/                   # Timestamped backtest logs
visualizer.py                # PnL & trade visualisation
```

## Tooling

- **Backtester** — a local fork of `prosperity4bt` lets us iterate offline against historical CSV data before submitting
- **Order book analyzer** — generates per-product diagnostic plots (liquidity depth, spread history, PnL per tick, implied position, mid price) from submission JSON logs
- **Auction simulator** — models the manual-round single-price auction to optimise bid/ask placement and estimate clearing prices

## Tech Stack

- Python 3.11
- NumPy, Pandas, Matplotlib, Plotly
- Custom backtesting engine

## Results

Currently ranked in the **top 5%** of all global participants after Round 2.

---

*Competition page: [prosperity.imc.com](https://prosperity.imc.com/)*

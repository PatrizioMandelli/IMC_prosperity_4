# IMC Prosperity 4 — Quantitative Trading Systems

[![Prosperity 4](https://img.shields.io/badge/Competition-IMC%20Prosperity%204-orange)](https://prosperity.imc.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

> **Professional-grade algorithmic trading suite for the IMC Prosperity 4 global challenge.**  
> Ranked in the top *% globally, featuring an ensemble of statistical arbitrage, market making, and trend-following strategies.

---

## 🏗️ Quantitative Infrastructure

Before diving into the strategies, it's essential to understand the tools used to validate our financial theses and monitor real-time performance.

### 1. Local Backtesting Engine (`prosperity4bt/`)
A high-fidelity local simulation environment forked from `prosperity4bt`. 
- **Worse-case Execution**: Simulated using `--match-trades worse` to account for adverse selection and fill-probability in thin markets.
- **Micro-price Analysis**: Tracks order book imbalance and weighted mid-prices to estimate short-term alpha.
- **Batch Validation**: Automates testing across all historical days to ensure strategy robustness.

### 2. Performance Visualizer (`visualizer.py`)
A custom **Streamlit** dashboard used to analyze submission logs and backtest results.
- **Inventory Heatmaps**: Monitors position risk and bias relative to mid-price.
- **Order Book Replay**: Visualizes market bid/ask spreads alongside bot executions.
- **PnL Attribution**: Breaks down profit by product and execution type (Maker vs. Taker).

---

## 📈 Algorithmic strategy Evolution by Round

### 🪐 Rounds 1 & 2: The Intarian Outpost
The objective was to qualify for Phase 2 by generating 200,000 XIRECs.

*   **ASH_COATED_OSMIUM**: Implemented an **FV-Anchored Market Maker**. Recognizing that Osmium mean-reverts tightly around 10,000, we used inventory-scaled quoting (quadratic scaling) to capture the spread while keeping a neutral delta.
*   **INTARIAN_PEPPER_ROOT**: 
    - **The Challenge**: A slow-growing root with a persistent upward drift.
    - **Our Edge**: We outperformed a simple **Buy & Hold** strategy by implementing a **God-Mode Drift Anchor**. 
    - **Mechanism**: The bot calculated a macro-slope (0.001) but instead of just holding, it deployed **"Whale Harpoons"** (sell traps) at precisely calculated offsets (Fair Value + 16, +20, +25). This allowed us to ride the trend while capturing massive PnL from the periodic +20 tick spikes discovered in our historical logs.

### 🌿 Round 3: Solvenar & The Ascension Trials
*   **Asset Class**: Delta-1 assets and 10 European-style Vouchers.
*   **Algorithm**: **Hybrid Black-Scholes + EMA Trend**.
    - For the underlying, we used a dual-EMA crossover to capture momentum.
    - For the **Vouchers**, we developed a custom **BS-Pricing Engine** with a Newton-Raphson solver to find Implied Volatility (IV) in real-time, allowing us to identify mispriced option premiums.

### 🛡️ Round 4: Counterparty Alpha
*   **The Shift**: We transitioned from trend-following to **Pure Mean Reversion** using an **Ornstein-Uhlenbeck (OU)** approach.
*   **Optimization**: We realized that ultra-slow EMAs (α=0.00005) were necessary to avoid "chasing" the price during intraday noise.
*   **Counterparty Signals**: 
    - **Mark 14 (Informed Taker)**: Our "Alpha Signal." When Mark 14 traded, we adjusted our edge to follow their direction.
    - **Mark 38 (Noise Trader)**: Our "Liquidity Source." We faded Mark 38's trades, providing liquidity against their noise.

### 🚀 Round 5: The Final Frontier
The system evolved into a namespaced master-orchestrator managing 10 distinct clusters:

| Cluster | Bot Strategy | Key Logic |
| :--- | :--- | :--- |
| **Robots** | **Oracle Stat-Arb** | `ROBOT ~ OVAL + SQUARE`. Exploits the latency between chip price discovery and robot updates. |
| **Pebbles** | **Structural Arb** | Maintains `Sum(Pebbles) = 50,000`. XL receives 68% of the signal correction based on variance. |
| **Galaxy** | **OU Mean Reversion** | Statistical anchoring using Ornstein-Uhlenbeck parameters for Dark Matter and Solar Winds. |
| **Translators** | **Basket Mean Rev** | RLS-driven regression against Microchips combined with relative basket mean reversion. |
| **Oxygen** | **Pair Trading** | Mean reversion on structural (Morning/Evening) and statistical (Garlic/Chocolate) pairs. |
| **Panels** | **Structural Basket** | Arbitrage on the synthetic spread between "Couple vs Couple" (1x2 + 1x4 vs 2x2 + 2x4). |
| **Sleep Pods** | **Adaptive Trend** | Market making with a trend-filter and a specific fix for the Day 4 trend inversion. |
| **Snack Packs** | **Weight Deviation** | Statistical arbitrage on flavor-weight deviations from the cluster equilibrium. |
| **Microchips** | **Directional Trend** | Passive market making with a short-term trend filter to avoid being run over during momentum shifts. |
| **UV Visors** | **Pairwise Z-Score** | Mean reversion on pairwise differences across the entire visor spectrum. |

---

## 🧠 Manual Trading Challenges

While our bots handled high-frequency execution, the **Manual Trading Challenges** required deep quantitative research into game theory, exotic option pricing, and constrained optimization.

### Round 1: An Intarian Welcome (Auctions)
- **Challenge**: Participating in single-price auctions for `DRYLAND_FLAX` and `EMBER_MUSHROOM` with a guaranteed merchant buyback.
- **Strategy**: Developed a volume-profit sweep in `manual_challenges/round1/static_book_filler.py`. We simulated the clearing price mechanism to identify the optimal bid price that maximizes the spread against the buyback price while accounting for the trade fees (0.10 for Mushrooms).

### Round 3: The Celestial Gardeners' Guild (Game Theory)
- **Challenge**: Bidding against 50 counterparties with uniform reserve prices (670-920). A cubic penalty was applied for second bids below the population mean.
- **Strategy**: Solved in `manual_challenges/round3/manual3.py` using a **Mean-Field Game** framework. We computed the symmetric Nash Equilibrium through three convergent paths:
    - **Quantal Response Equilibrium (QRE)**: Logit-based fixed-point iteration.
    - **Fictitious Play**: Perturbed best-response to empirical history.
    - **Monte Carlo Replicator Dynamics**: Finite-population evolutionary simulation.
- **Result**: Robust consensus bid (b1=700, b2=915) optimized for expected volume vs. penalty risk.

### Round 4: Vanilla Just Isn’t Exotic Enough (Options)
- **Challenge**: Portfolio optimization using `AETHER_CRYSTAL` spot, vanilla options, and exotic derivatives (Chooser, Binary, and Knock-out puts).
- **Strategy**: Implemented a high-performance Monte Carlo search engine in `manual_challenges/round4/manual4.py`.
    - **Global Search**: Explored 2M+ unique portfolios using coarse grid search and the **Cross-Entropy Method (CEM)**.
    - **Risk Management**: Optimized for the **Sortino Ratio**, using downside semi-std to manage the non-linear risk of exotic payouts.
    - **Greeks**: Used **Common Random Numbers (CRN)** to calculate bump-and-reprice Greeks (Delta, Gamma, Vega, Theta).

### Round 5: Extra! Extra! (News & KKT)
- **Challenge**: Distributing a 1,000,000 budget across 9 products based on asymmetric news catalysts, subject to a quadratic fee: `fee = (v/100)² * Budget`.
- **Strategy**: Solved in `manual_challenges/round5/manual5.py`.
    - **Analytical Optimum**: Used **Karush-Kuhn-Tucker (KKT)** conditions to derive the closed-form allocation: $a_i^* = (|\mu_i| - \lambda) / 2$.
    - **Distribution Modeling**: Modeled catalysts using Jump-Diffusion, Markov regimes, and Bimodal distributions to estimate $|\mu_i|$ before applying the KKT allocation.

---

## 👥 Team & Credits

- **Algorithmic Trading Systems**: Developed entirely by **Emanuele** and **Alessandro**.
- **Manual Trading Challenges**: Research and strategy primarily led by **Patrizio** and **Tommaso**.

---

## 🔬 Quantitative Research Pipeline (`analysis_tools/`)

During Round 5, we conducted several advanced analyses to find our edge:
- **Engle-Granger & Johansen Cointegration**: Scanned all 50 products to find stationary baskets. This led to the discovery of the **Pebbles Sum Invariant** and the **Robot-Chip link**.
- **PCA Decomposition**: Identified that 80% of Robot variance is explained by the first two principal components (Chips).
- **Regime Detection**: Used rolling correlation heatmaps to identify when statistical pairs were "breaking," allowing the bot to automatically deleverage.
- **Half-Life Estimation**: Used OU process regression to calibrate the EMA windows for the Galaxy sounds recorders.

---

## 🏁 Final Results

- **Global Standing**: Top 3*% (out of 28.000 initial teams).
- **National Standing**: 8#* (out of 400 initial teams, top *%)
- **Sharpe Ratio**: Consistently > 15 across core Stat-Arb modules.
- **Daily Performance**: 100% profitable days in all 5 rounds backtests under "Worse-Match" conditions.

---

*This project was developed for the IMC Prosperity 4 Global Challenge (2026). All strategies are for educational/simulated purposes.*

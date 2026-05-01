# TRANSLATOR Final Analysis Report

## 1. Executive Summary
The TRANSLATOR cluster consists of 5 highly correlated products: `ASTRO_BLACK`, `ECLIPSE_CHARCOAL`, `GRAPHITE_MIST`, `SPACE_GRAY`, and `VOID_BLUE`. These products are traded as a basket (identical trade timestamps and quantities). While absolute prices and simple spreads are non-stationary and shift significantly between days, the **relative price of each product to the basket mean, adjusted by a short-term rolling mean**, is extremely stationary and mean-reverting.

## 2. Key Findings
- **Intra-Basket Cointegration:** Johansen tests confirm the existence of multiple cointegrating relationships within the cluster.
- **Dynamic Relationships:** Fixed-coefficient spreads (like pairs) are unstable across days due to structural shifts in the products' relative value.
- **Stationary Signal:** The signal $S_i = (P_i - \bar{P}_{basket}) - EMA(P_i - \bar{P}_{basket})$ is highly stationary across all products and days.
  - Typical p-value: $10^{-7}$ to $10^{-10}$
  - Standard Deviation: ~70 ticks
  - Execution Cost: ~20 ticks (full spread)
- **Self-Balancing Property:** Since $\sum S_i = 0$ by construction, a strategy that takes positions proportional to $-S_i$ is naturally delta-neutral within the cluster.

## 3. Financial Thesis
The TRANSLATOR products represent variants of a single underlying economic value (likely a "translator" device in different colors). Market participants trade them as a basket, but temporary liquidity imbalances in individual products cause them to deviate from their fair relative value. These deviations are short-lived and mean-reverting. By trading the residual of each product relative to the basket's local equilibrium, we capture these "micro-arbitrage" opportunities.

## 4. Proposed Strategy: Self-Balancing Basket Mean Reversion
- **Signal:** For each product $i$, calculate the deviation from the basket mean, then subtract a 200-tick EMA of that deviation.
- **Execution:** 
  - If Signal $S_i > 1.0 \sigma$, Sell $P_i$.
  - If Signal $S_i < -1.0 \sigma$, Buy $P_i$.
- **Position Management:** Scale orders proportionally to the signal strength, respecting the $\pm 10$ position limit.
- **Robustness:** The use of a rolling EMA allows the strategy to adapt to the wild mean shifts observed between days.

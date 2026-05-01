# GALAXY Trading Strategy Final Analysis

## Executive Summary
The GALAXY cluster in IMC Prosperity Round 5 consists of 5 products: `BLACK_HOLES`, `DARK_MATTER`, `PLANETARY_RINGS`, `SOLAR_FLAMES`, and `SOLAR_WINDS`. While these products trade together as a basket, their long-term mid-price linear relationships are unstable and prone to drift. The most reliable source of profitability is high-frequency market making on the wide bid-ask spread (~13 ticks) using the **Weighted Mid-Price** as a short-term fair value estimate.

## Key Findings
1.  **Basket Behavior:** All 5 GALAXY products trade simultaneously in discrete basket trades (detected in trade logs at specific timestamps with identical quantities). This suggests a strong structural link, but not a constant mid-price sum or fixed ratio.
2.  **Unstable Cointegration:** OLS and Johansen cointegration tests showed that while stationary combinations exist within a single day (e.g., `DARK_MATTER` vs `SOLAR` products), the coefficients are highly unstable across different days, making static intra-cluster arbitrage risky.
3.  **Cross-Cluster Oracles:** External oracles (e.g., `UV_VISOR_YELLOW` for `DARK_MATTER`) provide strong directional signals but can distract from the high-frequency edge available within the cluster's own orderbook microstructure.
4.  **High-Frequency Edge:** The primary edge is capturing the bid-ask spread. Given the spread width of ~13 ticks and high volume, a market making strategy that quotes around the inventory-neutral price (weighted mid) is highly effective.

## Winning Strategy: GALAXY_best.py
The final strategy uses a refined high-frequency market making approach:
-   **Fair Value Estimation:** Calculates the `Weighted Mid-Price` for each product independently. This price accounts for orderbook imbalance and serves as a proxy for the next immediate mid-price move.
-   **Asymmetric Quoting:** Places passive buy/sell orders around the weighted mid. If the fair value is significantly tilted, the bot becomes more aggressive (takers) to capture immediate liquidity.
-   **Inventory Management:** Uses position-based skewing to encourage trades that return the position to zero, mitigating the risk of accumulating large directional exposure in the basket.

## Backtest Performance (GALAXY_best.log)
-   **Total Profit:** **+41,676**
-   **Annualized Sharpe Ratio:** **13.33**
-   **Max Drawdown:** **~2.1%**
-   **Daily Consistency:** Profitable on all three test days (+460, +19,846, +21,370).

## Conclusion
The GALAXY cluster represents a classic high-frequency trading opportunity where microstructure (orderbook imbalance) outweighs macro-structure (basket cointegration). The `GALAXY_best` bot effectively exploits this by providing liquidity where needed and capturing the spread, resulting in a robust and highly profitable strategy.

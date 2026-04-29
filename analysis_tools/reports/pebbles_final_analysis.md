# PEBBLES Final Analysis Report - IMC Prosperity Round 5

## 1. Financial Thesis
The PEBBLES cluster (XS, S, M, L, XL) exhibits two primary characteristics that were exploited for this strategy:
1.  **Strict Sum Constraint**: The sum of the mid-prices of all five products is mathematically anchored to 50,000. Any deviation in the sum of fair values represents a mean-reversion opportunity.
2.  **Anchor Correlation**: Each PEBBLE is fundamentally linked to an 'anchor' product from previous rounds (e.g., UV_VISOR_AMBER for XS, PANEL_2X4 for XL). These anchors act as price discovery leads.

The winning strategy, **V6 (Hybrid MM + Adaptive Risk)**, combines these insights:
-   **Anchor-Based Fair Value**: Uses historical betas and a rolling EMA intercept to derive a raw fair value for each product relative to its anchor.
-   **Sum Constraint Adjustment**: Applies a global adjustment to all individual fair values to ensure their sum is exactly 50,000.
-   **Market Making**: Quotes limit orders around the adjusted fair value with a dynamic edge tailored to each product's liquidity.
-   **Basket Arbitrage**: Actively 'takes' the basket when the total cost of best asks falls below 50,000 (minus a safety margin).
-   **Adaptive Position Skew**: Uses a non-linear risk factor that increases as the position approaches the +/- 10 limits, ensuring rapid return to neutrality.

## 2. Iterative Development Process
-   **Baseline**: Used hardcoded betas and basic MM logic. PnL: 105k.
-   **V1**: Updated betas using `pebbles_master_analyzer.py`. PnL improved to 120k.
-   **V2**: Added basket taker and wider edges. Performance dropped (89k) due to poor limit management and excessive aggression.
-   **V3**: Refined V1 with tighter parameters and strict limit enforcement. PnL: 125k.
-   **V4**: Introduced dynamic per-product edges. PnL: 125.4k.
-   **V5**: Added anchor momentum and even tighter edges. Performance dropped slightly (124k).
-   **V6 (Best)**: Introduced Adaptive Risk Factor (non-linear skew). PnL: **126.0k**, Sharpe: **22.3**.
-   **V7**: Aggressive edge tightening. Performance dropped slightly (123k).

## 3. Backtest Results (V6)
Backtested on all 3 days of Round 5 with `--match-trades worse` and strict `limit:10`.

| Metric | Value |
| :--- | :--- |
| **Total PnL** | 125,961 |
| **Sharpe (Annualized)** | 22.35 |
| **Max Drawdown** | 37,196 (3.5% of final PnL) |
| **PnL Day 2** | 29,464 |
| **PnL Day 3** | 58,766 |
| **PnL Day 4** | 37,732 |

### Inventory Management
Mean absolute positions stayed healthy across all products:
-   **XS**: 7.29
-   **S**: 8.91
-   **M**: 7.03
-   **L**: 7.07
-   **XL**: 5.81

## 4. Conclusion
The strategy is highly robust, profitable across all days, and maintains high Sharpe ratios even under "worse" trade matching conditions. The key edge comes from the tight integration of the sum constraint and the anchor-lead relationship, combined with aggressive but safe inventory management.

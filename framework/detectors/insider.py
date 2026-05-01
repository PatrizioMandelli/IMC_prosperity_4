import numpy as np
import pandas as pd
from scipy import stats

from framework.detectors.base import BotDetector, DetectionResult

MIN_TRADES = 30
DEFAULT_HORIZONS = [1, 2, 5, 10, 20]


class InsiderSimulator(BotDetector):
    """
    Isolates agents with predictive alpha by computing forward markout after their trades.
    Refactors data_explorer.analyze_trade_signatures() with multi-horizon testing.
    """

    def __init__(self, agent_id: str | None = None, markout_horizons: list[int] | None = None):
        self.agent_id = agent_id
        self.markout_horizons = markout_horizons or DEFAULT_HORIZONS

    def detect(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> DetectionResult:
        product = prices_df["product"].iloc[0] if "product" in prices_df.columns else "UNKNOWN"

        if trades_df.empty or len(trades_df) < MIN_TRADES:
            return self._insufficient("InsiderSimulator", product, len(trades_df))

        agent_id = self.agent_id or self._top_agent(trades_df)
        if not agent_id:
            # All buyers/sellers anonymous — fall back to trade-size markout clustering
            return self._quantity_cluster_detect(prices_df, trades_df, product)

        # Filter trades by agent
        buyer_mask = trades_df.get("buyer", pd.Series(dtype=str)).fillna("").str.strip() == agent_id
        seller_mask = trades_df.get("seller", pd.Series(dtype=str)).fillna("").str.strip() == agent_id
        agent_trades = trades_df[buyer_mask | seller_mask].copy()

        if len(agent_trades) < MIN_TRADES:
            # Fallback: cluster by trade quantity
            return self._quantity_cluster_detect(prices_df, trades_df, product)

        agent_trades = agent_trades.sort_values("global_time").reset_index(drop=True)
        price_ref = prices_df[["global_time", "mid_price"]].sort_values("global_time")

        # Merge current mid into agent trades
        merged = pd.merge_asof(
            agent_trades,
            price_ref.rename(columns={"mid_price": "mid_at_trade"}),
            on="global_time",
            direction="backward",
        )

        best_horizon, best_markout, best_tstat = 0, 0.0, 0.0
        horizon_results = {}
        for h in self.markout_horizons:
            future_ref = price_ref.copy()
            future_ref["future_time"] = future_ref["global_time"] + h * 100  # ~h ticks ahead
            future_ref = future_ref.rename(columns={"mid_price": "future_price"})[
                ["future_time", "future_price"]
            ]

            tmp = merged.copy()
            tmp["future_time"] = tmp["global_time"] + h * 100
            tmp = tmp.sort_values("future_time")
            tmp = pd.merge_asof(tmp, future_ref, on="future_time", direction="backward")
            tmp["markout"] = tmp["future_price"] - tmp["mid_at_trade"]

            valid = tmp["markout"].dropna()
            if len(valid) < 5:
                continue
            mean_m = valid.mean()
            sem_m = valid.sem()
            tstat = abs(mean_m / sem_m) if sem_m > 0 else 0.0
            horizon_results[h] = {"mean_markout": mean_m, "tstat": tstat, "n": len(valid)}

            if tstat > best_tstat:
                best_tstat = tstat
                best_markout = mean_m
                best_horizon = h

        confidence = float(np.tanh(best_tstat / 3))

        return DetectionResult(
            detector_name="InsiderSimulator",
            product=product,
            confidence=confidence,
            pattern_description=(
                f"Agent '{agent_id}' shows predictive alpha: "
                f"mean markout={best_markout:+.2f} over {best_horizon} ticks "
                f"(t-stat={best_tstat:.2f})"
            ),
            counter_strategy=(
                f"Shadow '{agent_id}': buy when they buy, sell when they sell. "
                f"Expected edge ≈ {best_markout:+.2f} per trade over {best_horizon} ticks."
            ),
            evidence={
                "agent_id": agent_id,
                "best_horizon": best_horizon,
                "best_markout": best_markout,
                "best_tstat": best_tstat,
                "horizon_results": horizon_results,
                "agent_trade_count": len(agent_trades),
            },
            flagged_timestamps=agent_trades["global_time"].tolist()[:200],
        )

    def _top_agent(self, trades_df: pd.DataFrame) -> str:
        """Returns buyer or seller with highest total volume (excluding SUBMISSION and empty)."""
        ignore = {"", "SUBMISSION"}
        buyers = trades_df.get("buyer", pd.Series(dtype=str)).fillna("").str.strip()
        sellers = trades_df.get("seller", pd.Series(dtype=str)).fillna("").str.strip()

        vol = {}
        for name, qty in zip(buyers, trades_df.get("quantity", pd.Series(dtype=float)).fillna(0)):
            if name and name not in ignore:
                vol[name] = vol.get(name, 0) + qty
        for name, qty in zip(sellers, trades_df.get("quantity", pd.Series(dtype=float)).fillna(0)):
            if name and name not in ignore:
                vol[name] = vol.get(name, 0) + qty

        return max(vol, key=vol.get) if vol else ""

    def _quantity_cluster_detect(
        self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, product: str
    ) -> DetectionResult:
        """Fallback: markout analysis grouped by trade quantity bucket."""
        price_ref = prices_df[["global_time", "mid_price"]].sort_values("global_time")
        merged = pd.merge_asof(
            trades_df.sort_values("global_time"),
            price_ref.rename(columns={"mid_price": "_cur_mid"}),
            on="global_time",
            direction="backward",
        )
        future_ref = price_ref.copy()
        future_ref["future_time"] = future_ref["global_time"] + 1000
        future_ref = future_ref.rename(columns={"mid_price": "_future_mid"})[
            ["future_time", "_future_mid"]
        ]
        merged["future_time"] = merged["global_time"] + 1000
        merged = merged.sort_values("future_time")
        merged = pd.merge_asof(merged, future_ref, on="future_time", direction="backward")
        merged["markout"] = merged["_future_mid"] - merged["_cur_mid"]

        size_stats = merged.groupby("quantity").agg(
            count=("markout", "count"),
            mean_markout=("markout", "mean"),
        ).reset_index()
        significant = size_stats[size_stats["count"] > 20].sort_values(
            "mean_markout", key=abs, ascending=False
        )

        if significant.empty:
            return self._insufficient("InsiderSimulator", product, len(trades_df))

        top = significant.iloc[0]
        confidence = min(float(abs(top["mean_markout"]) / 5), 1.0)
        return DetectionResult(
            detector_name="InsiderSimulator",
            product=product,
            confidence=confidence,
            pattern_description=(
                f"Trade size {int(top['quantity'])} has mean markout={top['mean_markout']:+.2f} "
                f"over 10 ticks (N={int(top['count'])}). Possible info-based bot."
            ),
            counter_strategy=f"When size={int(top['quantity'])} trades appear, copy direction immediately.",
            evidence={"top_size": int(top["quantity"]), "mean_markout": top["mean_markout"]},
            flagged_timestamps=[],
        )

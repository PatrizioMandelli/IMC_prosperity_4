import numpy as np
import pandas as pd
from scipy import stats

from framework.detectors.base import BotDetector, DetectionResult

DAY_LENGTH = 1_000_000
MIN_TRADES = 30


class PseudoDirectionalDetector(BotDetector):
    """
    Detects bots operating with fixed price bands and regular timing intervals.
    Combines three tests:
      A. Groundhog Day — same (timestamp % DAY_LENGTH) repeated across days
      B. Metronome     — low inter-trade gap coefficient of variation
      C. Price Band    — trade price range << market mid price range
    Refactors data_explorer.analyze_simulation_determinism().
    """

    def detect(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> DetectionResult:
        product = prices_df["product"].iloc[0] if "product" in prices_df.columns else "UNKNOWN"

        if trades_df.empty or len(trades_df) < MIN_TRADES:
            return self._insufficient("PseudoDirectionalDetector", product, len(trades_df))

        t = trades_df.sort_values("global_time").reset_index(drop=True)

        groundhog = self._groundhog_score(t)
        metronome = self._metronome_score(t)
        band = self._band_score(prices_df, t)

        confidence = float(np.clip(
            0.4 * groundhog + 0.3 * metronome + 0.3 * band, 0.0, 1.0
        ))

        # Flagged timestamps: trades in the most-repeated time-of-day slot
        flagged_ts = []
        if "timestamp" in t.columns:
            t["tod"] = t["timestamp"] % DAY_LENGTH
            top_tod = t["tod"].value_counts().idxmax()
            flagged_ts = t[t["tod"] == top_tod]["global_time"].tolist()[:200]

        return DetectionResult(
            detector_name="PseudoDirectionalDetector",
            product=product,
            confidence=confidence,
            pattern_description=(
                f"Groundhog={groundhog:.2f}, Metronome={metronome:.2f}, Band={band:.2f} "
                f"(weighted confidence={confidence:.2f}). "
                "Bot likely operates on fixed time slots and/or narrow price range."
            ),
            counter_strategy=(
                "Front-run predicted entry times with limit orders 1 tick inside. "
                "If price-band detected, quote aggressively at band boundaries."
            ),
            evidence={
                "groundhog_score": groundhog,
                "metronome_score": metronome,
                "band_score": band,
                "n_trades": len(t),
            },
            flagged_timestamps=flagged_ts,
        )

    def _groundhog_score(self, trades_df: pd.DataFrame) -> float:
        """Fraction of timestamps repeated mod DAY_LENGTH across multiple days."""
        if "day" not in trades_df.columns or trades_df["day"].nunique() < 2:
            return 0.0
        if "timestamp" not in trades_df.columns:
            return 0.0
        tod = trades_df["timestamp"] % DAY_LENGTH
        value_counts = tod.value_counts()
        repeated = (value_counts > 1).sum()
        return float(np.clip(repeated / max(len(value_counts), 1), 0.0, 1.0))

    def _metronome_score(self, trades_df: pd.DataFrame) -> float:
        """1 - CV(inter-trade gaps). High score = regular intervals = bot-like."""
        if len(trades_df) < 3:
            return 0.0
        gaps = np.diff(trades_df["global_time"].values)
        gaps = gaps[gaps > 0]
        if len(gaps) == 0 or np.mean(gaps) == 0:
            return 0.0
        cv = np.std(gaps) / np.mean(gaps)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _band_score(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
        """
        (trade_price_range / mid_price_range). Low ratio = fixed band = bot-like.
        Returns 0 if mid range is too small to be meaningful.
        """
        if "price" not in trades_df.columns:
            return 0.0
        trade_prices = trades_df["price"].dropna()
        mid_prices = prices_df["mid_price"].dropna()
        if len(trade_prices) < 5 or len(mid_prices) < 5:
            return 0.0

        trade_range = trade_prices.max() - trade_prices.min()
        mid_range = mid_prices.max() - mid_prices.min()
        if mid_range < 1:
            return 0.0

        ratio = trade_range / mid_range
        # High band_score when ratio << 1 (trade prices confined to narrow band)
        return float(np.clip(1.0 - ratio, 0.0, 1.0))

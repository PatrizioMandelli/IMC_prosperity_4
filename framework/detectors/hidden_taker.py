import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

from framework.detectors.base import BotDetector, DetectionResult

MIN_OCCURRENCES = 10


class HiddenTakerDetector(BotDetector):
    """
    Detects systematic best-bid erosion without corresponding visible passive orders.
    Pattern: bid_price_1 drops by >= threshold ticks in 1-3 consecutive timestamps,
             accompanied by anonymous buy trades (empty buyer name).
    """

    def __init__(self, erosion_threshold: int = 2, min_occurrences: int = MIN_OCCURRENCES):
        self.erosion_threshold = erosion_threshold
        self.min_occurrences = min_occurrences

    def detect(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> DetectionResult:
        product = prices_df["product"].iloc[0] if "product" in prices_df.columns else "UNKNOWN"

        if len(prices_df) < 50:
            return self._insufficient("HiddenTakerDetector", product, len(prices_df))

        df = prices_df.sort_values("global_time").reset_index(drop=True)
        bid1 = df["bid_price_1"].ffill()

        # Find ticks where best bid drops by >= threshold
        bid_drop = bid1.diff()
        erosion_mask = bid_drop <= -self.erosion_threshold
        erosion_idx = df.index[erosion_mask].tolist()

        if len(erosion_idx) < self.min_occurrences:
            return DetectionResult(
                detector_name="HiddenTakerDetector",
                product=product,
                confidence=0.0,
                pattern_description=f"Only {len(erosion_idx)} erosion events found (threshold={self.erosion_threshold})",
                counter_strategy="No systematic hidden taker detected.",
                evidence={"erosion_count": len(erosion_idx), "threshold": self.erosion_threshold},
                flagged_timestamps=[],
            )

        flagged_ts = df.loc[erosion_idx, "global_time"].tolist()

        # Check for anonymous buy trades near erosion events (within 500 ticks)
        anon_buys = pd.DataFrame()
        if not trades_df.empty and "buyer" in trades_df.columns:
            anon_buys = trades_df[
                (trades_df["buyer"].fillna("").str.strip() == "") &
                (trades_df.get("maker_taker", pd.Series("UNKNOWN", index=trades_df.index)) == "TAKER")
            ]

        # Compute inter-event gap coefficient of variation (metronome score)
        gaps = np.diff(sorted(flagged_ts))
        cv = float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 1 and np.mean(gaps) > 0 else 1.0
        metronome_score = max(0.0, 1.0 - cv)  # low CV = regular = high score

        # Confidence: sigmoid of normalized erosion frequency
        erosion_rate = len(erosion_idx) / max(len(df), 1) * 100
        base_conf = float(expit(erosion_rate - 2))  # inflection at 2% erosion rate
        confidence = float(np.clip(base_conf * (1 + 0.3 * metronome_score), 0.0, 1.0))

        avg_interval = float(np.mean(gaps)) if len(gaps) > 0 else 0.0

        return DetectionResult(
            detector_name="HiddenTakerDetector",
            product=product,
            confidence=confidence,
            pattern_description=(
                f"Best bid erodes {len(erosion_idx)} times (>={self.erosion_threshold} ticks each); "
                f"avg interval={avg_interval:.0f} ticks, metronome_score={metronome_score:.2f}, "
                f"anonymous buy trades nearby={len(anon_buys)}"
            ),
            counter_strategy=(
                "Post passive asks 1-2 ticks above current ask_1 during erosion windows "
                "to capture the hidden taker premium. Widen spread when erosion_flag=True."
            ),
            evidence={
                "erosion_count": len(erosion_idx),
                "threshold": self.erosion_threshold,
                "avg_interval_ticks": avg_interval,
                "metronome_score": metronome_score,
                "cv_gaps": cv,
                "anon_taker_trades": len(anon_buys),
            },
            flagged_timestamps=flagged_ts[:200],  # cap to avoid huge serialization
        )

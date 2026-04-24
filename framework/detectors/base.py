from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DetectionResult:
    detector_name: str
    product: str
    confidence: float           # 0.0 – 1.0
    pattern_description: str
    counter_strategy: str
    evidence: dict = field(default_factory=dict)
    flagged_timestamps: list[int] = field(default_factory=list)


class BotDetector(ABC):
    """
    Each detector receives a single-product normalized prices_df and classified trades_df.
    Returns a DetectionResult with confidence score, description, and counter-strategy.
    """

    @abstractmethod
    def detect(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> DetectionResult:
        ...

    @staticmethod
    def _confidence_from_pvalue(p: float) -> float:
        """Maps a scipy p-value to [0, 1] confidence."""
        return float(max(0.0, min(1.0, 1.0 - p)))

    @staticmethod
    def _insufficient(detector_name: str, product: str, n: int) -> DetectionResult:
        return DetectionResult(
            detector_name=detector_name,
            product=product,
            confidence=0.0,
            pattern_description=f"Insufficient data (N={n} trades)",
            counter_strategy="N/A",
            evidence={"n_trades": n},
        )

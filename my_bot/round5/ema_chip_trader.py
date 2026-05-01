from collections import deque
from typing import Dict, List

import numpy as np

from datamodel import Order, OrderDepth, TradingState


class Trader:
    """Long-trend follower for the MICROCHIP family with optional short-term filter."""

    LIMIT = 10
    HARD_SHORT = {"MICROCHIP_OVAL"}

    # Each tuple is: (direction, trend_window, segment_start, segment_end,
    # threshold, short_filter_window). short_filter_window is None when no
    # short-term confirmation is required.
    CONFIGS = {
        "MICROCHIP_SQUARE":    (+1,  80, 15,  5, 4.5,  40),
        "MICROCHIP_CIRCLE":    (+1, 100, 10, 10, 8.0, None),
        "MICROCHIP_TRIANGLE":  (-1, 100, 10, 10, 5.0, None),
        "MICROCHIP_RECTANGLE": (-1, 100, 10, 10, 5.0,  50),
    }

    def __init__(self):
        all_products = self.HARD_SHORT | set(self.CONFIGS.keys())
        max_w = max(cfg[1] for cfg in self.CONFIGS.values())
        self.prices: Dict[str, deque] = {p: deque(maxlen=max_w) for p in all_products}

    @staticmethod
    def mid(depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    def _passive_order(
        self, symbol: str, depth: OrderDepth, pos: int, target: int
    ) -> List[Order]:
        """Place a single passive order one tick inside the spread."""
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        diff = target - pos
        if diff > 0:
            return [Order(symbol, best_bid + 1, diff)]
        if diff < 0:
            return [Order(symbol, best_ask - 1, diff)]
        return []

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # Update price history.
        for p in self.prices:
            depth = state.order_depths.get(p)
            if depth:
                m = self.mid(depth)
                if m:
                    self.prices[p].append(m)

        # OVAL: persistent short — the most reliable negative-slope asset.
        for p in self.HARD_SHORT:
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            pos = state.position.get(p, 0)
            orders = self._passive_order(p, depth, pos, -self.LIMIT)
            if orders:
                result[p] = orders

        # Other assets: trend logic with optional short-window protection.
        for p, (direction, trend_w, s_start, s_end, threshold, short_w) in self.CONFIGS.items():
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            pos = state.position.get(p, 0)
            hist = list(self.prices[p])
            if len(hist) < trend_w:
                continue

            # Long-trend signal: difference of the head and tail averages.
            start_val = np.mean(hist[:s_start])
            end_val = np.mean(hist[-s_end:])
            long_trend = end_val - start_val

            long_confirms = (
                (direction > 0 and long_trend > threshold)
                or (direction < 0 and long_trend < -threshold)
            )

            target = 0
            if long_confirms:
                if short_w is not None:
                    # Short window guard: only stay in if recent bars confirm.
                    short_hist = hist[-short_w:]
                    short_start = np.mean(short_hist[:5])
                    short_end = np.mean(short_hist[-5:])
                    short_trend = short_end - short_start
                    short_confirms = (
                        (direction > 0 and short_trend > 0)
                        or (direction < 0 and short_trend < 0)
                    )
                    target = self.LIMIT * direction if short_confirms else 0
                else:
                    target = self.LIMIT * direction

            if pos != target:
                orders = self._passive_order(p, depth, pos, target)
                if orders:
                    result[p] = orders

        return result, 0, ""

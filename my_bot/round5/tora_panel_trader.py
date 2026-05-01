import json
import math
from typing import Optional

from datamodel import Order, TradingState


class Trader:
    """Couple-vs-couple basket trader.

    Trades the spread (1X2 + 1X4) - (2X2 + 2X4) against an EMA mean and a
    floor-protected std. Entries on |z| >= ENTRY_Z, exits on |z| <= EXIT_Z.
    """

    def __init__(self):
        self.LIMIT = 10

        self.COUPLE_LEFT = ["PANEL_1X2", "PANEL_1X4"]
        self.COUPLE_RIGHT = ["PANEL_2X2", "PANEL_2X4"]

        self.START_MEAN = -2350

        self.ENTRY_Z = 1
        self.EXIT_Z = 0

        # Slow EMA on the structural mean; faster EMA on the realized variance.
        self.MEAN_ALPHA = 0.0001
        self.VOL_ALPHA = 0.01
        self.VOL_FLOOR = 30.0

    @staticmethod
    def get_mid(symbol: str, state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(symbol)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def run(self, state: TradingState):
        result = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}
        key = "BASKET_USER"

        # Need a complete book on both legs.
        mids = {}
        for sym in self.COUPLE_LEFT + self.COUPLE_RIGHT:
            mid = self.get_mid(sym, state)
            if mid is None:
                return result, 0, json.dumps(trader_data)
            mids[sym] = mid

        current_spread = (mids["PANEL_1X2"] + mids["PANEL_1X4"]) - (
            mids["PANEL_2X2"] + mids["PANEL_2X4"]
        )

        if key not in trader_data:
            trader_data[key] = {"mean": self.START_MEAN, "m2": self.VOL_FLOOR ** 2}

        mu = trader_data[key]["mean"]
        m2 = trader_data[key]["m2"]

        mu = (1 - self.MEAN_ALPHA) * mu + self.MEAN_ALPHA * current_spread
        m2 = (1 - self.VOL_ALPHA) * m2 + self.VOL_ALPHA * ((current_spread - mu) ** 2)

        trader_data[key] = {"mean": mu, "m2": m2}
        std = max(math.sqrt(m2), self.VOL_FLOOR)
        z_score = (current_spread - mu) / std

        # Pick the target direction.
        target_dir = 0
        current_lead_pos = state.position.get("PANEL_1X2", 0)
        if z_score > self.ENTRY_Z:
            target_dir = -1   # spread too high -> short left, long right
        elif z_score < -self.ENTRY_Z:
            target_dir = 1    # spread too low -> long left, short right
        elif abs(z_score) <= self.EXIT_Z:
            target_dir = 0    # reverted: flatten
        else:
            # In the dead zone: hold the existing direction.
            if current_lead_pos > 0:
                target_dir = 1
            elif current_lead_pos < 0:
                target_dir = -1

        # Left couple follows the target direction.
        for sym in self.COUPLE_LEFT:
            target_pos = target_dir * self.LIMIT
            current_pos = state.position.get(sym, 0)
            diff = target_pos - current_pos
            if diff != 0:
                depth = state.order_depths[sym]
                price = (
                    min(depth.sell_orders.keys())
                    if diff > 0
                    else max(depth.buy_orders.keys())
                )
                result.setdefault(sym, []).append(Order(sym, int(price), diff))

        # Right couple inverts the target direction.
        for sym in self.COUPLE_RIGHT:
            target_pos = -target_dir * self.LIMIT
            current_pos = state.position.get(sym, 0)
            diff = target_pos - current_pos
            if diff != 0:
                depth = state.order_depths[sym]
                price = (
                    min(depth.sell_orders.keys())
                    if diff > 0
                    else max(depth.buy_orders.keys())
                )
                result.setdefault(sym, []).append(Order(sym, int(price), diff))

        return result, 0, json.dumps(trader_data)

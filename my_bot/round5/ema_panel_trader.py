import json
import math
from datamodel import Order, TradingState
from typing import Optional


class Trader:
    def __init__(self):
        # Full limit allocated to the single master basket
        self.LIMIT = 10

        # ───────────────────────────────────────────────────────────
        # THE COUPLE VS COUPLE BASKET: (12 + 14) vs (22 + 24)
        # ───────────────────────────────────────────────────────────
        self.COUPLE_LEFT = ['PANEL_1X2', 'PANEL_1X4']
        zself.COUPLE_RIGHT = ['PANEL_2X2', 'PANEL_2X4']

        # THE STRUCTURAL ANCHOR
        self.START_MEAN = -2350

        self.ENTRY_Z = 1
        self.EXIT_Z = 0

        # Strategy Parameters
        self.MEAN_ALPHA = 0.0001  # Slow EMA to respect the structural mean
        self.VOL_ALPHA = 0.01
        self.VOL_FLOOR = 30.0

    def get_mid(self, symbol: str, state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(symbol)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def run(self, state: TradingState):
        result = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}
        key = "BASKET_USER"

        # 1. Fetch all Mid Prices
        mids = {}
        for sym in self.COUPLE_LEFT + self.COUPLE_RIGHT:
            mid = self.get_mid(sym, state)
            if mid is None:
                return result, 0, json.dumps(trader_data)
            mids[sym] = mid

        # 2. Calculate the "Couple vs Couple" Synthetic Spread
        current_spread = (mids['PANEL_1X2'] + mids['PANEL_1X4']) - \
            (mids['PANEL_2X2'] + mids['PANEL_2X4'])

        # 3. Apply the Hardcoded Structural Anchor
        if key not in trader_data:
            trader_data[key] = {
                "mean": self.START_MEAN, "m2": self.VOL_FLOOR**2}

        mu = trader_data[key]["mean"]
        m2 = trader_data[key]["m2"]

        # Update EMA stats
        mu = (1 - self.MEAN_ALPHA) * mu + self.MEAN_ALPHA * current_spread
        m2 = (1 - self.VOL_ALPHA) * m2 + \
            self.VOL_ALPHA * ((current_spread - mu) ** 2)

        trader_data[key] = {"mean": mu, "m2": m2}
        std = max(math.sqrt(m2), self.VOL_FLOOR)
        z_score = (current_spread - mu) / std

        # 4. Determine Target Direction
        target_dir = 0
        current_lead_pos = state.position.get('PANEL_1X2', 0)

        if z_score > self.ENTRY_Z:
            target_dir = -1  # Spread too high -> Short Left, Long Right
        elif z_score < -self.ENTRY_Z:
            target_dir = 1   # Spread too low -> Long Left, Short Right
        elif abs(z_score) <= self.EXIT_Z:
            target_dir = 0   # Reverted
        else:
            if current_lead_pos > 0:
                target_dir = 1
            elif current_lead_pos < 0:
                target_dir = -1

        # 5. Synchronized Execution
        # --- LEFT COUPLE (Follows target_dir) ---
        for sym in self.COUPLE_LEFT:
            target_pos = target_dir * self.LIMIT
            current_pos = state.position.get(sym, 0)
            diff = target_pos - current_pos

            if diff != 0:
                depth = state.order_depths[sym]
                # FIXED EXECUTION LOGIC
                if diff > 0:  # We want to BUY
                    price = min(depth.sell_orders.keys())  # Hit lowest ASK
                else:        # We want to SELL
                    price = max(depth.buy_orders.keys())  # Hit highest BID
                result.setdefault(sym, []).append(Order(sym, int(price), diff))

        # --- RIGHT COUPLE (Inverts target_dir) ---
        for sym in self.COUPLE_RIGHT:
            target_pos = -target_dir * self.LIMIT
            current_pos = state.position.get(sym, 0)
            diff = target_pos - current_pos

            if diff != 0:
                depth = state.order_depths[sym]
                # FIXED EXECUTION LOGIC
                if diff > 0:  # We want to BUY
                    price = min(depth.sell_orders.keys())  # Hit lowest ASK
                else:        # We want to SELL
                    price = max(depth.buy_orders.keys())  # Hit highest BID
                result.setdefault(sym, []).append(Order(sym, int(price), diff))

        return result, 0, json.dumps(trader_data)
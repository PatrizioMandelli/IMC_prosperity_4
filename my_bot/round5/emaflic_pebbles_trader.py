import json
import math
from datamodel import Order, TradingState, OrderDepth

class Trader:
    """
    Financial Thesis:
    1. Anchor-Based Pricing: Optimized betas and EMA intercept.
    2. Sum Constraint: Strong anchor at 50,000.
    3. Mid-Range Dynamic Edge: XS: 1.2, S: 1.6, M: 1.8, L: 1.8, XL: 2.2.
    4. Adaptive Risk Factor: Skew becomes more aggressive as position nears limits.
    5. Selective Basket Arb: Margin of 2.0.
    """

    CONFIG = {
        "PEBBLES_XS": { "anchor": "UV_VISOR_AMBER",             "beta":  1.4, "edge": 1.2 },
        "PEBBLES_S":  { "anchor": "GALAXY_SOUNDS_BLACK_HOLES",  "beta": -0.8, "edge": 1.6 },
        "PEBBLES_M":  { "anchor": "OXYGEN_SHAKE_MORNING_BREATH","beta": -0.9, "edge": 1.8 },
        "PEBBLES_L":  { "anchor": "TRANSLATOR_GRAPHITE_MIST",   "beta":  0.8, "edge": 1.8 },
        "PEBBLES_XL": { "anchor": "PANEL_2X4",                  "beta":  2.5, "edge": 2.2 },
    }

    LIMIT           = 10
    TARGET_SUM      = 50000.0
    N_TARGETS       = 5
    ALPHA           = 0.01
    RISK_FACTOR     = 1.0
    ARB_MARGIN      = 2.0

    def get_mid(self, depth: OrderDepth):
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def run(self, state: TradingState):
        result = {}

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        data.setdefault("intercepts", {})
        data.setdefault("last_raw_fairs", {})

        raw_fairs  = {}
        best_bids  = {}
        best_asks  = {}
        live_fairs = set()

        for target, cfg in self.CONFIG.items():
            anchor = cfg["anchor"]
            beta   = cfg["beta"]

            depth = state.order_depths.get(target)
            t_mid = self.get_mid(depth)

            anchor_depth = state.order_depths.get(anchor)
            a_mid = self.get_mid(anchor_depth)

            if depth and depth.buy_orders and depth.sell_orders:
                best_bids[target] = max(depth.buy_orders.keys())
                best_asks[target] = min(depth.sell_orders.keys())

            if t_mid is not None and a_mid is not None:
                intercept = t_mid - a_mid * beta
                if target not in data["intercepts"]:
                    data["intercepts"][target] = intercept
                else:
                    data["intercepts"][target] = (
                        (1 - self.ALPHA) * data["intercepts"][target]
                        + self.ALPHA * intercept
                    )
                fv = a_mid * beta + data["intercepts"][target]
                raw_fairs[target] = fv
                data["last_raw_fairs"][target] = fv
                live_fairs.add(target)
            elif target in data["last_raw_fairs"]:
                raw_fairs[target] = data["last_raw_fairs"][target]

        # 1. Selective Basket Arbitrage Check
        basket_buy_opportunity = False
        basket_sell_opportunity = False

        if len(best_asks) == self.N_TARGETS:
            sum_asks = sum(best_asks.values())
            if sum_asks < self.TARGET_SUM - self.ARB_MARGIN:
                basket_buy_opportunity = True

        if len(best_bids) == self.N_TARGETS:
            sum_bids = sum(best_bids.values())
            if sum_bids > self.TARGET_SUM + self.ARB_MARGIN:
                basket_sell_opportunity = True

        # 2. Sum Constraint Adjustment for Fair Values
        if len(raw_fairs) == self.N_TARGETS:
            adjustment = (self.TARGET_SUM - sum(raw_fairs.values())) / self.N_TARGETS
        else:
            adjustment = 0.0

        for target in live_fairs:
            depth = state.order_depths[target]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            fv        = raw_fairs[target] + adjustment
            pos       = state.position.get(target, 0)

            orders = []

            # --- Basket Taker Logic ---
            if basket_buy_opportunity:
                buy_qty = self.LIMIT - pos
                if buy_qty > 0:
                    orders.append(Order(target, best_asks[target], buy_qty))
                    pos += buy_qty

            elif basket_sell_opportunity:
                sell_qty = -self.LIMIT - pos
                if sell_qty < 0:
                    orders.append(Order(target, best_bids[target], sell_qty))
                    pos += sell_qty

            # --- Market Making Logic ---
            buy_limit = self.LIMIT - pos
            sell_limit = -self.LIMIT - pos

            if buy_limit > 0 or sell_limit < 0:
                # Adaptive Risk Factor
                current_risk = self.RISK_FACTOR
                if abs(pos) > 7:
                    current_risk *= 1.5

                skewed_fv = fv - pos * current_risk
                edge = self.CONFIG[target]["edge"]
                my_bid = math.floor(skewed_fv - edge)
                my_ask = math.ceil(skewed_fv + edge)

                my_bid = min(my_bid, best_asks[target] - 1)
                my_ask = max(my_ask, best_bids[target] + 1)

                if buy_limit > 0:
                    orders.append(Order(target, my_bid, buy_limit))
                if sell_limit < 0:
                    orders.append(Order(target, my_ask, sell_limit))

            if orders:
                result[target] = orders

        return result, 0, json.dumps(data)

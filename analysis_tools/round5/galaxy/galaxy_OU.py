import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List


class Trader:
    def __init__(self):
        self.products = [
            'GALAXY_SOUNDS_DARK_MATTER',
            'GALAXY_SOUNDS_BLACK_HOLES',
            'GALAXY_SOUNDS_SOLAR_WINDS',
            'GALAXY_SOUNDS_PLANETARY_RINGS',
        ]
        # OU half-lives extracted from fitted OU model; used to derive EMA alpha.
        self.windows = {
            'GALAXY_SOUNDS_DARK_MATTER': 1438,
            'GALAXY_SOUNDS_BLACK_HOLES': 16861,
            'GALAXY_SOUNDS_SOLAR_WINDS': 3479,
            'GALAXY_SOUNDS_PLANETARY_RINGS': 8455,
        }
        self.emas = {p: None for p in self.products}
        self.position_limit = 10
        # Sigma > 30 warrants a wider half-spread to avoid adverse selection.
        self.half_spreads = {p: 8.0 for p in self.products}

    def run(self, state: TradingState):
        result = {}

        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                for p in self.products:
                    if p in saved:
                        self.emas[p] = saved[p]
            except Exception:
                pass

        for product in self.products:
            if product not in state.order_depths:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if not order_depth.sell_orders or not order_depth.buy_orders:
                continue

            best_ask = min(order_depth.sell_orders)
            best_bid = max(order_depth.buy_orders)
            mid_price = (best_bid + best_ask) / 2.0

            alpha = 2.0 / (self.windows[product] + 1.0)
            if self.emas[product] is None:
                self.emas[product] = mid_price
            else:
                self.emas[product] = alpha * mid_price + (1.0 - alpha) * self.emas[product]

            ema = self.emas[product]
            half_spread = self.half_spreads[product]
            pos = state.position.get(product, 0)
            max_buy = self.position_limit - pos
            max_sell = -self.position_limit - pos

            if product != 'GALAXY_SOUNDS_PLANETARY_RINGS':
                # Market-making with inventory skew; hard cap prevents crossing the spread.
                fair = ema - pos * 2.5
                bid = min(math.floor(fair - half_spread), best_ask - 1)
                ask = max(math.ceil(fair + half_spread), best_bid + 1)

                # Aggressive unwind when approaching position limit.
                if pos >= 8:
                    ask = best_bid
                elif pos <= -8:
                    bid = best_ask

                if max_buy > 0:
                    orders.append(Order(product, bid, max_buy))
                if max_sell < 0:
                    orders.append(Order(product, ask, max_sell))

            else:
                # Trend-following: take liquidity in trend direction, rest passively.
                threshold = 3.0
                if mid_price > ema + threshold:
                    taker = min(max(0, -pos), max_buy)
                    passive = max_buy - taker
                    if taker > 0:
                        orders.append(Order(product, best_ask, taker))
                    if passive > 0:
                        orders.append(Order(product, best_bid, passive))
                elif mid_price < ema - threshold:
                    capacity = -max_sell
                    taker = min(max(0, pos), capacity)
                    passive = capacity - taker
                    if taker > 0:
                        orders.append(Order(product, best_bid, -taker))
                    if passive > 0:
                        orders.append(Order(product, best_ask, -passive))

            result[product] = orders

        return result, 0, json.dumps(self.emas)

from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            result[product] = self.osmium_strategy(state, state.order_depths[product])

        return result, conversions, ""

    def osmium_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        orders: List[Order] = []

        FAIR_VALUE = 10000
        LIMIT = 80

        # ── Parametri da sweepare ─────────────────────────────────────────
        TIGHT_IMPROVEMENT = 1   # tick dentro al best per il tight level
        DEEP_OFFSET = 2         # distanza simmetrica del deep dal tight

        current_pos = state.position.get(product, 0)
        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        # ── STEP 1: Arbitrage (invariato) ─────────────────────────────────
        if order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < FAIR_VALUE and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order(product, ask_price, take_vol))
                        buy_capacity -= take_vol
                else:
                    break

        if order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > FAIR_VALUE and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order(product, bid_price, take_vol))
                        sell_capacity -= take_vol
                else:
                    break

        # ── STEP 2: Market Making parametrizzato ──────────────────────────
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else FAIR_VALUE - 5

        my_bid = min(FAIR_VALUE - 1, best_bid + TIGHT_IMPROVEMENT)
        my_ask = max(FAIR_VALUE + 1, best_ask - TIGHT_IMPROVEMENT)

        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol
            if tight_buy_vol > 0:
                orders.append(Order(product, my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order(product, my_bid - DEEP_OFFSET, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = sell_capacity // 2
            deep_sell_vol = sell_capacity - tight_sell_vol
            if tight_sell_vol < 0:
                orders.append(Order(product, my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order(product, my_ask + DEEP_OFFSET, deep_sell_vol))

        return orders
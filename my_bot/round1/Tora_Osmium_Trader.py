from typing import List

from datamodel import OrderDepth, Order, TradingState


class Trader:
    """Two-step trader on ASH_COATED_OSMIUM: arb takes around FV, then quotes."""

    PRODUCT = "ASH_COATED_OSMIUM"
    FAIR_VALUE = 10000
    LIMIT = 80

    def run(self, state: TradingState):
        result = {}
        if self.PRODUCT in state.order_depths:
            result[self.PRODUCT] = self._osmium_strategy(
                state, state.order_depths[self.PRODUCT]
            )
        return result, 0, ""

    def _osmium_strategy(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        fv = self.FAIR_VALUE

        pos = state.position.get(self.PRODUCT, 0)
        buy_cap = self.LIMIT - pos
        sell_cap = -self.LIMIT - pos

        # Take any quote that crosses fair value.
        if depth.sell_orders:
            for ask_price, ask_vol in sorted(depth.sell_orders.items()):
                if ask_price < fv and buy_cap > 0:
                    take_vol = min(buy_cap, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order(self.PRODUCT, ask_price, take_vol))
                        buy_cap -= take_vol

        if depth.buy_orders:
            for bid_price, bid_vol in sorted(depth.buy_orders.items(), reverse=True):
                if bid_price > fv and sell_cap < 0:
                    take_vol = max(sell_cap, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order(self.PRODUCT, bid_price, take_vol))
                        sell_cap -= take_vol

        # Quote one tick inside the spread but never cross FV.
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fv + 5
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fv - 5

        my_bid = min(fv - 1, best_bid + 1)
        my_ask = max(fv + 1, best_ask - 1)

        if buy_cap > 0:
            tight_buy_vol = buy_cap // 2
            deep_buy_vol = buy_cap - tight_buy_vol
            if tight_buy_vol > 0:
                orders.append(Order(self.PRODUCT, my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order(self.PRODUCT, my_bid - 2, deep_buy_vol))

        if sell_cap < 0:
            tight_sell_vol = int(sell_cap / 2)
            deep_sell_vol = sell_cap - tight_sell_vol
            if tight_sell_vol < 0:
                orders.append(Order(self.PRODUCT, my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order(self.PRODUCT, my_ask + 1, deep_sell_vol))

        return orders

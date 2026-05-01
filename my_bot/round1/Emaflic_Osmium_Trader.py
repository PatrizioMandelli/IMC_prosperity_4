from typing import List

from datamodel import OrderDepth, Order, TradingState


class Trader:
    """FV-anchored market maker for ASH_COATED_OSMIUM.

    Quote size scales linearly with the deviation from the known fair value,
    so capacity is committed only when the price is actually away from FV.
    """

    PRODUCT = "ASH_COATED_OSMIUM"
    FAIR_VALUE = 10000
    LIMIT = 80
    DEV_FULL = 5.0  # ticks of deviation that justify full capacity

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

        best_ask = min(depth.sell_orders) if depth.sell_orders else fv + 8
        best_bid = max(depth.buy_orders) if depth.buy_orders else fv - 8
        fair = (best_ask + best_bid) / 2
        dev = fair - fv

        # Take any liquidity that crosses fair value.
        if depth.sell_orders:
            for ask_p, ask_v in sorted(depth.sell_orders.items()):
                if ask_p < fv and buy_cap > 0:
                    vol = min(buy_cap, abs(ask_v))
                    orders.append(Order(self.PRODUCT, ask_p, vol))
                    buy_cap -= vol

        if depth.buy_orders:
            for bid_p, bid_v in sorted(depth.buy_orders.items(), reverse=True):
                if bid_p > fv and sell_cap < 0:
                    vol = max(sell_cap, -bid_v)
                    orders.append(Order(self.PRODUCT, bid_p, vol))
                    sell_cap -= vol

        # FV-anchored quoting: never cross fair value.
        my_bid = min(fv - 1, best_bid + 1)
        my_ask = max(fv + 1, best_ask - 1)

        buy_scale = max(0.0, min(1.0, -dev / self.DEV_FULL))
        sell_scale = max(0.0, min(1.0, dev / self.DEV_FULL))

        # Tighter quote takes more size as we move further from FV.
        tight_frac = 0.5 + 0.3 * max(buy_scale, sell_scale)

        if buy_cap > 0 and buy_scale > 0:
            total_buy = min(buy_cap, round(self.LIMIT * buy_scale))
            tight_vol = max(1, round(total_buy * tight_frac))
            deep_vol = max(0, total_buy - tight_vol)
            orders.append(Order(self.PRODUCT, my_bid, tight_vol))
            if deep_vol > 0:
                orders.append(Order(self.PRODUCT, my_bid - 2, deep_vol))

        if sell_cap < 0 and sell_scale > 0:
            total_sell = max(sell_cap, -round(self.LIMIT * sell_scale))
            tight_vol = min(-1, -round(abs(total_sell) * tight_frac))
            deep_vol = max(total_sell - tight_vol, 0)
            orders.append(Order(self.PRODUCT, my_ask, tight_vol))
            if deep_vol > 0:
                orders.append(Order(self.PRODUCT, my_ask + 2, -deep_vol))

        # Stay quoting near FV so we keep collecting flow when dev is small.
        if abs(dev) < 1.0:
            neutral_size = 15
            if buy_cap > 0:
                orders.append(Order(self.PRODUCT, my_bid, min(neutral_size, buy_cap)))
            if sell_cap < 0:
                orders.append(Order(self.PRODUCT, my_ask, max(-neutral_size, sell_cap)))

        return orders

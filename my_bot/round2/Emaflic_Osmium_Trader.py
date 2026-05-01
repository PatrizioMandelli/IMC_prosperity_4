from typing import List

from datamodel import OrderDepth, Order, TradingState


class Trader:
    """ASH_COATED_OSMIUM trader: edge-filtered arb plus inventory-scaled MM."""

    PRODUCT = "ASH_COATED_OSMIUM"
    FAIR_VALUE = 10000
    LIMIT = 80

    # Min edge (in ticks) before we will cross to take liquidity.
    ARB_EDGE_MIN = 5
    # Fraction of capacity earmarked for arb takes.
    ARB_BUDGET_FRAC = 0.3
    # Distance of the deep maker quote from the tight quote.
    DEEP_OFFSET = 3

    def run(self, state: TradingState):
        result = {}
        if self.PRODUCT in state.order_depths:
            result[self.PRODUCT] = self._osmium_strategy(
                state, state.order_depths[self.PRODUCT]
            )
        return result, 0, ""

    def _osmium_strategy(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        product = self.PRODUCT
        fv = self.FAIR_VALUE

        pos = state.position.get(product, 0)
        inventory_ratio = pos / self.LIMIT  # in [-1, +1]

        buy_cap = self.LIMIT - pos
        sell_cap = -self.LIMIT - pos

        # Pressure is non-zero only on the side we are exposed on.
        long_pressure = max(0.0, inventory_ratio)
        short_pressure = max(0.0, -inventory_ratio)

        # Quadratic taper: full capacity in the safe zone, sharp cut near the limit.
        arb_buy_scale = max(0.0, 1.0 - long_pressure ** 2)
        arb_sell_scale = max(0.0, 1.0 - short_pressure ** 2)

        arb_buy_cap = int(buy_cap * self.ARB_BUDGET_FRAC * arb_buy_scale)
        arb_sell_cap = int(sell_cap * self.ARB_BUDGET_FRAC * arb_sell_scale)

        # Take liquidity only when edge >= ARB_EDGE_MIN.
        if depth.sell_orders:
            for ask_price, ask_vol in sorted(depth.sell_orders.items()):
                if ask_price <= fv - self.ARB_EDGE_MIN and arb_buy_cap > 0:
                    take = min(arb_buy_cap, abs(ask_vol))
                    if take > 0:
                        orders.append(Order(product, ask_price, take))
                        arb_buy_cap -= take
                        buy_cap -= take
                else:
                    break

        if depth.buy_orders:
            for bid_price, bid_vol in sorted(depth.buy_orders.items(), reverse=True):
                if bid_price >= fv + self.ARB_EDGE_MIN and arb_sell_cap < 0:
                    take = max(arb_sell_cap, -bid_vol)
                    if take < 0:
                        orders.append(Order(product, bid_price, take))
                        arb_sell_cap -= take
                        sell_cap -= take
                else:
                    break

        # Market making: standard FV±1 clamp, no price skew. Inventory only via volume.
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fv + 5
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fv - 5

        my_bid = min(fv - 1, best_bid + 1)
        my_ask = max(fv + 1, best_ask - 1)

        mm_buy_scale = max(0.0, 1.0 - long_pressure ** 2)
        mm_sell_scale = max(0.0, 1.0 - short_pressure ** 2)

        eff_buy_cap = int(buy_cap * mm_buy_scale)
        eff_sell_cap = int(sell_cap * mm_sell_scale)

        if eff_buy_cap > 0:
            tight = eff_buy_cap // 2
            deep = eff_buy_cap - tight
            if tight > 0:
                orders.append(Order(product, my_bid, tight))
            if deep > 0:
                orders.append(Order(product, my_bid - self.DEEP_OFFSET, deep))

        if eff_sell_cap < 0:
            tight = eff_sell_cap // 2
            deep = eff_sell_cap - tight
            if tight < 0:
                orders.append(Order(product, my_ask, tight))
            if deep < 0:
                orders.append(Order(product, my_ask + self.DEEP_OFFSET, deep))

        return orders

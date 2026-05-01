from typing import List

from datamodel import OrderDepth, Order, TradingState


class Trader:
    """Buy-and-hold baseline that keeps INTARIAN_PEPPER_ROOT pinned at +80."""

    PRODUCT = "INTARIAN_PEPPER_ROOT"
    POSITION_LIMIT = 80

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            if product != self.PRODUCT:
                continue

            depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            current_pos = state.position.get(product, 0)
            quantity_to_buy = self.POSITION_LIMIT - current_pos

            if quantity_to_buy > 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                orders.append(Order(product, best_ask, quantity_to_buy))

            result[product] = orders

        return result, 0, ""

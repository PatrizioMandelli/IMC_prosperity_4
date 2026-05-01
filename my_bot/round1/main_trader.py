import json
from typing import List

from datamodel import OrderDepth, Order, TradingState


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"
LIMIT = 80


class Trader:
    """Combined Osmium MM + Pepper buy-and-hold strategy."""

    def run(self, state: TradingState):
        result = {}

        if state.traderData == "":
            history = {}
        else:
            try:
                history = json.loads(state.traderData)
            except Exception:
                history = {}

        # Wipe trend memory on day rollover.
        current_virtual_day = state.timestamp // 100000
        last_virtual_day = history.get("system_last_day")
        if last_virtual_day is None or current_virtual_day > last_virtual_day:
            if PEPPER in history:
                history[PEPPER]["trend_ewma"] = None
        history["system_last_day"] = current_virtual_day

        for product in state.order_depths:
            depth: OrderDepth = state.order_depths[product]

            if product == OSMIUM:
                result[product] = self.osmium_strategy(state, depth)

            elif product == PEPPER:
                prod_history = history.get(product, {})
                orders, updated = self.compute_pepper_root_strategy_CORE_EXPLORE(
                    state, depth, prod_history
                )
                result[product] = orders
                history[product] = updated

            else:
                result[product] = []

        return result, 0, json.dumps(history)

    def osmium_strategy(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        fair_value = 10000

        pos = state.position.get(OSMIUM, 0)
        buy_cap = LIMIT - pos
        sell_cap = -LIMIT - pos

        if depth.sell_orders:
            for ask_p, ask_v in sorted(depth.sell_orders.items()):
                if ask_p < fair_value and buy_cap > 0:
                    take = min(buy_cap, abs(ask_v))
                    if take > 0:
                        orders.append(Order(OSMIUM, ask_p, take))
                        buy_cap -= take

        if depth.buy_orders:
            for bid_p, bid_v in sorted(depth.buy_orders.items(), reverse=True):
                if bid_p > fair_value and sell_cap < 0:
                    take = max(sell_cap, -bid_v)
                    if take < 0:
                        orders.append(Order(OSMIUM, bid_p, take))
                        sell_cap -= take

        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fair_value + 5
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fair_value - 5

        my_bid = min(fair_value - 1, best_bid + 1)
        my_ask = max(fair_value + 1, best_ask - 1)

        if buy_cap > 0:
            tight = buy_cap // 2
            deep = buy_cap - tight
            if tight > 0:
                orders.append(Order(OSMIUM, my_bid, tight))
            if deep > 0:
                orders.append(Order(OSMIUM, my_bid - 2, deep))

        if sell_cap < 0:
            tight = int(sell_cap / 2)
            deep = sell_cap - tight
            if tight < 0:
                orders.append(Order(OSMIUM, my_ask, tight))
            if deep < 0:
                orders.append(Order(OSMIUM, my_ask + 1, deep))

        return orders

    def compute_pepper_root_strategy_CORE_EXPLORE(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ):
        orders: List[Order] = []
        position = state.position.get(PEPPER, 0)
        buy_cap = LIMIT - position
        sell_cap = -LIMIT - position

        current_time = state.timestamp
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 20000

        macro_slope = 0.001
        prev_time = prod_history.get("prev_time", -1)
        is_new_day = (
            current_time == 0
            or current_time < prev_time
            or current_time % 1000000 == 0
        )

        if prod_history.get("day_open_price") is None or is_new_day:
            day_open_price = (best_bid + best_ask) / 2.0
            prod_history["day_open_price"] = day_open_price
            prod_history["day_start_time"] = current_time
        else:
            day_open_price = prod_history["day_open_price"]

        day_start_time = prod_history.get("day_start_time", 0)
        rel_time = current_time - day_start_time
        fair_value = day_open_price + (rel_time * macro_slope)
        prod_history["prev_time"] = current_time

        # Higher tolerance early in the day to lock the long core.
        is_startup = rel_time < 5000 and buy_cap > 0

        # Dump core into spikes above FV+7.
        if sell_cap < 0 and depth.buy_orders:
            for bid_p, bid_v in sorted(depth.buy_orders.items(), reverse=True):
                if bid_p >= fair_value + 7 and sell_cap < 0:
                    take = max(sell_cap, -bid_v)
                    if take < 0:
                        orders.append(Order(PEPPER, bid_p, take))
                        sell_cap -= take

        if buy_cap > 0 and depth.sell_orders:
            for ask_p, ask_v in sorted(depth.sell_orders.items()):
                max_acceptable_ask = fair_value + 15 if is_startup else fair_value + 6
                if ask_p <= max_acceptable_ask and buy_cap > 0:
                    take = min(buy_cap, abs(ask_v))
                    if take > 0:
                        orders.append(Order(PEPPER, ask_p, take))
                        buy_cap -= take

        if sell_cap < 0:
            t1 = int(sell_cap * 1 / 2)
            t2 = sell_cap - t1
            if t1 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 7)), t1))
            if t2 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 8)), t2))

        if buy_cap > 0:
            if is_startup:
                my_bid = min(int(round(fair_value + 10)), best_bid + 2)
            else:
                my_bid = min(int(round(fair_value + 4)), best_bid + 1)
            orders.append(Order(PEPPER, my_bid, buy_cap))

        prod_history["last_price"] = fair_value
        return orders, prod_history

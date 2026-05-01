import json
from typing import List

from datamodel import OrderDepth, Order, TradingState


class Trader:
    """Osmium edge-filter arb + scaled MM, plus a Pepper drift-anchored core.

    The Pepper strategy maintains a +80 long position against a linear
    drift-anchored fair value, with a circuit breaker that re-anchors and
    flushes inventory when the price decouples from the trend for too long.
    """

    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"
    LIMIT = 80

    OSMIUM_FAIR_VALUE = 10000
    ARB_EDGE_MIN = 5
    ARB_BUDGET_FRAC = 0.3

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
            if self.PEPPER in history:
                history[self.PEPPER]["trend_ewma"] = None
        history["system_last_day"] = current_virtual_day

        for product in state.order_depths:
            depth: OrderDepth = state.order_depths[product]

            if product == self.OSMIUM:
                result[product] = self.osmium_strategy(state, depth)

            elif product == self.PEPPER:
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
        product = self.OSMIUM
        fv = self.OSMIUM_FAIR_VALUE

        pos = state.position.get(product, 0)
        inventory_ratio = pos / self.LIMIT
        buy_cap = self.LIMIT - pos
        sell_cap = -self.LIMIT - pos

        long_pressure = max(0.0, inventory_ratio)
        short_pressure = max(0.0, -inventory_ratio)

        arb_buy_scale = max(0.0, 1.0 - long_pressure ** 2)
        arb_sell_scale = max(0.0, 1.0 - short_pressure ** 2)

        arb_buy_cap = int(buy_cap * self.ARB_BUDGET_FRAC * arb_buy_scale)
        arb_sell_cap = int(sell_cap * self.ARB_BUDGET_FRAC * arb_sell_scale)

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

        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fv + 10
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fv - 10

        my_bid = min(fv - 1, best_bid + 1)
        my_ask = max(fv + 1, best_ask - 1)

        if buy_cap > 0:
            tight = int(buy_cap // 3)
            deep = buy_cap - tight
            if tight > 0:
                orders.append(Order(product, my_bid, tight))
            if deep > 0:
                orders.append(Order(product, my_bid - 5, deep))

        if sell_cap < 0:
            tight = int(sell_cap / 3)
            deep = sell_cap - tight
            if tight < 0:
                orders.append(Order(product, my_ask, tight))
            if deep < 0:
                orders.append(Order(product, my_ask + 5, deep))

        return orders

    def compute_pepper_root_strategy_CORE_EXPLORE(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ):
        product = self.PEPPER
        orders: List[Order] = []
        position = state.position.get(product, 0)
        buy_cap = self.LIMIT - position
        sell_cap = -self.LIMIT - position

        current_time = state.timestamp

        # Live mid (fall back to last known fair value if the book is empty).
        fallback_mid = prod_history.get("last_price", 10000)
        if depth.buy_orders and depth.sell_orders:
            live_mid = (
                max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())
            ) / 2.0
        else:
            live_mid = fallback_mid

        # Sample mid every 5k ticks for slope re-estimation on a circuit-breaker trip.
        snapshots = prod_history.get("price_snapshots", [])
        if current_time % 5000 == 0:
            snapshots.append((current_time, live_mid))
            if len(snapshots) > 20:
                snapshots.pop(0)
            prod_history["price_snapshots"] = snapshots

        macro_slope = prod_history.get("ACTIVE_SLOPE", 0.001)

        prev_time = prod_history.get("prev_time", -1)
        is_new_day = current_time == 0 or current_time < prev_time

        if prod_history.get("day_open_price") is None or is_new_day:
            prod_history["day_open_price"] = live_mid
            prod_history["day_start_time"] = current_time

        day_open_price = prod_history["day_open_price"]
        day_start_time = prod_history.get("day_start_time", 0)
        rel_time = current_time - day_start_time
        fair_value = day_open_price + (rel_time * macro_slope)
        prod_history["prev_time"] = current_time

        best_bid = (
            max(depth.buy_orders.keys()) if depth.buy_orders
            else int(round(fair_value - 10))
        )
        best_ask = (
            min(depth.sell_orders.keys()) if depth.sell_orders
            else int(round(fair_value + 10))
        )

        # Circuit breaker: re-anchor only after the price stays decoupled
        # from the trend line for many consecutive ticks.
        broken_ticks = prod_history.get("broken_ticks", 0)
        if live_mid < fair_value - 100 or live_mid > fair_value + 100:
            broken_ticks += 1
        else:
            broken_ticks = 0
        prod_history["broken_ticks"] = broken_ticks

        if broken_ticks >= 50:
            # Refit slope from recent snapshots.
            new_slope = 0.001
            if len(snapshots) >= 5:
                first_ts, first_price = snapshots[0]
                last_ts, last_price = snapshots[-1]
                if last_ts > first_ts:
                    new_slope = (last_price - first_price) / (last_ts - first_ts)

            prod_history["ACTIVE_SLOPE"] = new_slope
            prod_history["day_open_price"] = live_mid
            prod_history["day_start_time"] = current_time
            prod_history["broken_ticks"] = 0
            fair_value = live_mid

            if position > 0:
                # Liquidate the long exposure into whatever bid is available.
                orders.append(Order(product, best_bid, -position))
            return orders, prod_history

        is_startup = rel_time < 500 and buy_cap > 0

        # Dump core into spikes >= FV+7.
        if sell_cap < 0 and depth.buy_orders:
            for bid_price, bid_vol in sorted(depth.buy_orders.items(), reverse=True):
                if bid_price >= fair_value + 7 and sell_cap < 0:
                    take = max(sell_cap, -bid_vol)
                    if take < 0:
                        orders.append(Order(product, bid_price, take))
                        sell_cap -= take

        # Force the long core; tolerate higher asks during startup.
        if buy_cap > 0 and depth.sell_orders:
            for ask_price, ask_vol in sorted(depth.sell_orders.items()):
                max_acceptable_ask = fair_value + 20 if is_startup else fair_value + 6
                if ask_price <= max_acceptable_ask and buy_cap > 0:
                    take = min(buy_cap, abs(ask_vol))
                    if take > 0:
                        orders.append(Order(product, ask_price, take))
                        buy_cap -= take

        my_ask1 = int(round(fair_value + 7))
        my_ask2 = int(round(fair_value + 15))

        my_bid_startup = min(int(round(fair_value + 8)), best_bid + 1)
        my_bid_normal = min(int(round(fair_value + 4)), best_bid + 1)
        my_bid = my_bid_startup if is_startup else my_bid_normal
        # Never let our bid cross our own first sell tier.
        my_bid = min(my_bid, my_ask1 - 1)

        if sell_cap < 0:
            t1 = int(sell_cap * 1 / 2)
            t2 = sell_cap - t1
            if t1 < 0:
                orders.append(Order(product, my_ask1, t1))
            if t2 < 0:
                orders.append(Order(product, my_ask2, t2))

        if buy_cap > 0:
            orders.append(Order(product, my_bid, buy_cap))

        prod_history["last_price"] = fair_value
        return orders, prod_history

import json
from typing import List, Tuple

from datamodel import OrderDepth, Order, TradingState


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"
LIMIT = 80


class Trader:
    """Container for several Pepper Root and Osmium strategy variants.

    The active strategy is selected in `run` (currently CORE_EXPLORE for Pepper
    and the no-op `osmium_strategy_0` for Osmium).
    """

    def __init__(self):
        self.short_ewma = None
        self.long_ewma = None

    @staticmethod
    def calculate_wall_mid(depth: OrderDepth) -> float:
        """Mid based on the price level with the largest volume on each side."""
        best_wall_bid = max(depth.buy_orders.items(), key=lambda x: x[1])[0]
        # Sell volumes are negative -> min picks the largest absolute size.
        best_wall_ask = min(depth.sell_orders.items(), key=lambda x: x[1])[0]
        return (best_wall_bid + best_wall_ask) / 2

    # --- Osmium variants ----------------------------------------------------

    def osmium_strategy_0(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        return []

    def osmium_strategy_MAIN(self, state: TradingState, depth: OrderDepth) -> List[Order]:
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
                orders.append(Order(OSMIUM, my_bid - 6, deep))

        if sell_cap < 0:
            tight = int(sell_cap / 2)
            deep = sell_cap - tight
            if tight < 0:
                orders.append(Order(OSMIUM, my_ask, tight))
            if deep < 0:
                orders.append(Order(OSMIUM, my_ask + 6, deep))

        return orders

    def osmium_strategy_wait(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        """Edge-tiered taker plus tight-quote inventory cap."""
        orders: List[Order] = []
        fair_value = 10000

        pos = state.position.get(OSMIUM, 0)
        virtual_pos = pos  # tracks fills accumulated within this tick

        buy_cap = LIMIT - pos
        sell_cap = -LIMIT - pos

        if depth.sell_orders:
            for ask_p, ask_v in sorted(depth.sell_orders.items()):
                if ask_p < fair_value and buy_cap > 0:
                    edge = fair_value - ask_p
                    if edge >= 5:
                        max_long = 80
                    elif edge >= 3:
                        max_long = 60
                    else:
                        max_long = 20
                    tier_budget = max(0, max_long - virtual_pos)
                    take = min(buy_cap, abs(ask_v), tier_budget)
                    if take > 0:
                        orders.append(Order(OSMIUM, ask_p, take))
                        buy_cap -= take
                        virtual_pos += take

        if depth.buy_orders:
            for bid_p, bid_v in sorted(depth.buy_orders.items(), reverse=True):
                if bid_p > fair_value and sell_cap < 0:
                    edge = bid_p - fair_value
                    if edge >= 5:
                        max_short = -80
                    elif edge >= 3:
                        max_short = -60
                    else:
                        max_short = -20
                    tier_budget = min(0, max_short - virtual_pos)
                    take = max(sell_cap, -bid_v, tier_budget)
                    if take < 0:
                        orders.append(Order(OSMIUM, bid_p, take))
                        sell_cap -= take
                        virtual_pos += take

        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else fair_value + 5
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else fair_value - 5

        my_bid = min(fair_value - 1, best_bid + 1)
        my_ask = max(fair_value + 1, best_ask - 1)

        # Cap the tight-quote inventory; remainder goes to the deep level.
        tight_max_pos = 15

        if buy_cap > 0:
            tight_room = max(0, tight_max_pos - virtual_pos)
            tight = min(buy_cap // 2, tight_room)
            deep = buy_cap - tight
            if tight > 0:
                orders.append(Order(OSMIUM, my_bid, tight))
            if deep > 0:
                orders.append(Order(OSMIUM, my_bid - 1, deep))

        if sell_cap < 0:
            tight_min_pos = -15
            tight_room = min(0, tight_min_pos - virtual_pos)
            tight = max(int(sell_cap / 2), tight_room)
            deep = sell_cap - tight
            if tight < 0:
                orders.append(Order(OSMIUM, my_ask, tight))
            if deep < 0:
                orders.append(Order(OSMIUM, my_ask + 1, deep))

        return orders

    def osmium_strategy_depth(self, state: TradingState, depth: OrderDepth) -> List[Order]:
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
                orders.append(Order(OSMIUM, my_bid - 3, deep))

        if sell_cap < 0:
            tight = int(sell_cap / 2)
            deep = sell_cap - tight
            if tight < 0:
                orders.append(Order(OSMIUM, my_ask, tight))
            if deep < 0:
                orders.append(Order(OSMIUM, my_ask + 4, deep))

        return orders

    # --- Pepper variants ----------------------------------------------------

    def compute_pepper_root_strategy_0(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ):
        return [], prod_history

    def compute_pepper_root_strategy_BH(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ) -> Tuple[List[Order], dict]:
        """Buy-and-hold against a fixed-slope FV anchor with markup sell traps."""
        orders: List[Order] = []
        position = state.position.get(PEPPER, 0)
        buy_cap = LIMIT - position
        sell_cap = -LIMIT - position

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 20000
        current_mid = (best_bid + best_ask) / 2.0
        current_time = state.timestamp

        macro_slope = 0.001

        # Anchor the intercept on the first tick of the day.
        day_open_price = prod_history.get("day_open_price")
        if day_open_price is None or current_time == 0:
            day_open_price = current_mid
            prod_history["day_open_price"] = day_open_price

        fair_value = day_open_price + (current_time * macro_slope)

        # Take asks below FV.
        if depth.sell_orders:
            for ask_p, ask_v in sorted(depth.sell_orders.items()):
                if ask_p < fair_value and buy_cap > 0:
                    take = min(buy_cap, abs(ask_v))
                    if take > 0:
                        orders.append(Order(PEPPER, ask_p, take))
                        buy_cap -= take

        # Take bids well above FV (the typical +20/+26 spike).
        if depth.buy_orders:
            for bid_p, bid_v in sorted(depth.buy_orders.items(), reverse=True):
                if bid_p >= fair_value + 15 and sell_cap < 0:
                    take = max(sell_cap, -bid_v)
                    if take < 0:
                        orders.append(Order(PEPPER, bid_p, take))
                        sell_cap -= take

        # Accumulate at FV.
        my_bid = min(int(round(fair_value)), best_bid + 1)
        if buy_cap > 0:
            orders.append(Order(PEPPER, my_bid, buy_cap))

        # Layered sell traps tuned to observed spike percentiles.
        if sell_cap < 0:
            t1 = int(sell_cap / 3)
            t2 = int(sell_cap / 3)
            t3 = sell_cap - t1 - t2
            if t1 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 16)), t1))
            if t2 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 20)), t2))
            if t3 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 25)), t3))

        prod_history["last_price"] = current_mid
        prod_history["last_fair_value"] = fair_value
        return orders, prod_history

    def compute_pepper_root_strategy_GRID(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ) -> Tuple[List[Order], dict]:
        """Static grid quotes around an OLS-derived linear FV."""
        orders: List[Order] = []
        position = state.position.get(PEPPER, 0)
        buy_cap = LIMIT - position
        sell_cap = -LIMIT - position

        current_time = state.timestamp
        macro_slope = 0.001272

        # Anchor on the day's first tick to absorb between-day price gaps.
        day_open_price = prod_history.get("day_open_price")
        if day_open_price is None or current_time == 0:
            if depth.buy_orders and depth.sell_orders:
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                day_open_price = (best_bid + best_ask) / 2.0
            else:
                day_open_price = 11974.32  # fallback to known Day -2 intercept
            prod_history["day_open_price"] = day_open_price

        fair_value = day_open_price + (current_time * macro_slope)

        # Bid grid: tight inside the typical spread, deep outside.
        if buy_cap > 0:
            tight = buy_cap // 2
            deep = buy_cap - tight
            if tight > 0:
                orders.append(Order(PEPPER, int(round(fair_value - 5)), tight))
            if deep > 0:
                orders.append(Order(PEPPER, int(round(fair_value - 10)), deep))

        # Sell traps at the 95th percentile and the historical max anomaly.
        if sell_cap < 0:
            t1 = int(sell_cap / 2)
            t2 = sell_cap - t1
            if t1 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 17)), t1))
            if t2 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 24)), t2))

        return orders, prod_history

    def compute_pepper_root_strategy_SNIPER(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ):
        """Day-aware FV with a 5-tick warmup and patient maker quotes."""
        orders: List[Order] = []
        position = state.position.get(PEPPER, 0)
        buy_cap = LIMIT - position
        sell_cap = -LIMIT - position

        current_time = state.timestamp
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 20000
        current_mid = (best_bid + best_ask) / 2.0

        # Log own fills against the previous fair value.
        last_fv = prod_history.get("last_price", 0)
        if last_fv > 0 and PEPPER in state.own_trades:
            for trade in state.own_trades[PEPPER]:
                if trade.timestamp == current_time - 100:
                    print(
                        f"[TS {current_time}] fill qty={trade.quantity} px={trade.price} "
                        f"prev_fv={last_fv:.2f} dist={trade.price - last_fv:.2f}"
                    )

        macro_slope = 0.001
        prev_time = prod_history.get("prev_time", -1)
        is_new_day = (
            current_time == 0
            or current_time < prev_time
            or current_time % 1000000 == 0
        )

        warmup_prices = prod_history.get("warmup_prices", [])
        if is_new_day:
            warmup_prices = []
            prod_history["day_open_price"] = None
            prod_history["day_start_time"] = current_time

        if prod_history.get("day_open_price") is None:
            warmup_prices.append(current_mid)
            prod_history["warmup_prices"] = warmup_prices
            day_open_price = current_mid  # use live mid while warming up
            if len(warmup_prices) >= 5:
                # Lock in the warmup average as the day's anchor.
                day_open_price = sum(warmup_prices) / len(warmup_prices)
                prod_history["day_open_price"] = day_open_price
        else:
            day_open_price = prod_history["day_open_price"]

        day_start_time = prod_history.get("day_start_time", 0)
        rel_time = current_time - day_start_time
        fair_value = day_open_price + (rel_time * macro_slope)
        prod_history["prev_time"] = current_time

        # Patient buyer: take asks at FV or below, then rest a maker bid.
        if buy_cap > 0:
            if depth.sell_orders:
                for ask_p, ask_v in sorted(depth.sell_orders.items()):
                    if ask_p <= fair_value and buy_cap > 0:
                        take = min(buy_cap, abs(ask_v))
                        if take > 0:
                            orders.append(Order(PEPPER, ask_p, take))
                            buy_cap -= take
            if buy_cap > 0:
                my_bid = min(int(round(fair_value - 1)), best_bid + 1)
                orders.append(Order(PEPPER, my_bid, buy_cap))

        # Two-tier sell traps at +12 and +20.
        if sell_cap < 0:
            t1 = int(sell_cap / 2)
            t2 = sell_cap - t1
            if t1 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 12)), t1))
            if t2 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 20)), t2))

        prod_history["last_price"] = fair_value
        return orders, prod_history

    def compute_pepper_root_strategy_BASICS(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ):
        """VWAP-anchored buy-and-hold with three-tier sell traps."""
        orders: List[Order] = []
        position = state.position.get(PEPPER, 0)
        buy_cap = LIMIT - position
        sell_cap = -LIMIT - position

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 20000

        if PEPPER in state.own_trades:
            for trade in state.own_trades[PEPPER]:
                if (
                    trade.timestamp == state.timestamp - 100
                    and trade.seller == "SUBMISSION"
                ):
                    print(
                        f"[TS {state.timestamp}] sold {trade.quantity} @ {trade.price}"
                    )

        # Volume-weighted mid: weight each side's price by the opposite side's size.
        bid_vol = sum(depth.buy_orders.values())
        ask_vol = abs(sum(depth.sell_orders.values()))
        if bid_vol > 0 and ask_vol > 0:
            vwap_mid = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        else:
            vwap_mid = (best_bid + best_ask) / 2.0
        fair_value = vwap_mid

        if buy_cap > 0:
            if depth.sell_orders:
                for ask_p, a_vol in sorted(depth.sell_orders.items()):
                    if ask_p <= fair_value and buy_cap > 0:
                        take = min(buy_cap, abs(a_vol))
                        if take > 0:
                            orders.append(Order(PEPPER, ask_p, take))
                            buy_cap -= take
            if buy_cap > 0:
                my_bid = min(int(round(fair_value - 1)), best_bid + 1)
                orders.append(Order(PEPPER, my_bid, buy_cap))

        # Three sell tiers at +12 / +18 / +24 with weights 40/40/20.
        if sell_cap < 0:
            t1 = int(sell_cap * 0.4)
            t2 = int(sell_cap * 0.4)
            t3 = sell_cap - t1 - t2
            if t1 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 12)), t1))
            if t2 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 18)), t2))
            if t3 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 24)), t3))

        return orders, prod_history

    def compute_pepper_root_strategy_CORE_EXPLORE(
        self, state: TradingState, depth: OrderDepth, prod_history: dict
    ):
        """Day-anchored core that aims to hold +80 inventory."""
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

        # Force fills early in the day to lock the long core.
        is_startup = rel_time < 5000 and buy_cap > 0

        # Dump core into spikes >= FV+7.
        if sell_cap < 0 and depth.buy_orders:
            for bid_p, bid_v in sorted(depth.buy_orders.items(), reverse=True):
                if bid_p >= fair_value + 7 and sell_cap < 0:
                    take = max(sell_cap, -bid_v)
                    if take < 0:
                        orders.append(Order(PEPPER, bid_p, take))
                        sell_cap -= take

        # Accumulate the core. Higher tolerance during startup to ensure fills.
        if buy_cap > 0 and depth.sell_orders:
            for ask_p, ask_v in sorted(depth.sell_orders.items()):
                max_acceptable_ask = fair_value + 15 if is_startup else fair_value + 6
                if ask_p <= max_acceptable_ask and buy_cap > 0:
                    take = min(buy_cap, abs(ask_v))
                    if take > 0:
                        orders.append(Order(PEPPER, ask_p, take))
                        buy_cap -= take

        # Two-tier sell traps at +7 and +8.
        if sell_cap < 0:
            t1 = int(sell_cap * 1 / 2)
            t2 = sell_cap - t1
            if t1 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 7)), t1))
            if t2 < 0:
                orders.append(Order(PEPPER, int(round(fair_value + 8)), t2))

        # Maker bid: aggressive overbid during startup, cap at FV+4 otherwise.
        if buy_cap > 0:
            if is_startup:
                my_bid = min(int(round(fair_value + 10)), best_bid + 2)
            else:
                my_bid = min(int(round(fair_value + 4)), best_bid + 1)
            orders.append(Order(PEPPER, my_bid, buy_cap))

        prod_history["last_price"] = fair_value
        return orders, prod_history

    # --- Entry point --------------------------------------------------------

    def run(self, state: TradingState):
        result = {}

        if state.traderData == "":
            history = {}
        else:
            try:
                history = json.loads(state.traderData)
            except Exception:
                history = {}

        # Wipe trend memory on day rollovers.
        current_virtual_day = state.timestamp // 100000
        last_virtual_day = history.get("system_last_day")
        if last_virtual_day is None or current_virtual_day > last_virtual_day:
            if PEPPER in history:
                history[PEPPER]["trend_ewma"] = None
        history["system_last_day"] = current_virtual_day

        for product in state.order_depths:
            depth: OrderDepth = state.order_depths[product]

            if product == OSMIUM:
                result[product] = self.osmium_strategy_0(state, depth)

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

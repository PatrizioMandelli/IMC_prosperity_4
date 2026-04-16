from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import json


class Trader:

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # 1. UNPACK MEMORY
        if state.traderData == "":
            history = {}
        else:
            try:
                history = json.loads(state.traderData)
            except Exception:
                history = {}

        # 2. OVERNIGHT GAP PROTECTION
        current_virtual_day = state.timestamp // 100000
        last_virtual_day = history.get("system_last_day")

        if last_virtual_day is None or current_virtual_day > last_virtual_day:
            if "INTARIAN_PEPPER_ROOT" in history:
                history["INTARIAN_PEPPER_ROOT"]["trend_ewma"] = None

        history["system_last_day"] = current_virtual_day

        # 3. ROUTE TO STRATEGIES
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            if product == "ASH_COATED_OSMIUM":
                result[product] = self.osmium_strategy(state, order_depth)

            elif product == "INTARIAN_PEPPER_ROOT":
                prod_history = history.get(product, {})
                orders, updated_prod_history = self.compute_pepper_root_strategy_CORE_EXPLORE(
                    state, order_depth, prod_history)
                result[product] = orders
                history[product] = updated_prod_history

            else:
                result[product] = []

        # 4. PACK MEMORY
        traderData = json.dumps(history)

        return result, conversions, traderData

    def osmium_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []

        FAIR_VALUE = 10000
        LIMIT = 80

        current_pos = state.position.get("ASH_COATED_OSMIUM", 0)

        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        # STEP 1: Take Liquidity - Arbitrage
        if len(order_depth.sell_orders) > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < FAIR_VALUE and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask_price, take_vol))
                        buy_capacity -= take_vol

        if len(order_depth.buy_orders) > 0:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > FAIR_VALUE and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid_price, take_vol))
                        sell_capacity -= take_vol

        # STEP 2: DYNAMIC MARKET MAKING
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else FAIR_VALUE - 5

        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_bid - 2, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = int(sell_capacity / 2)
            deep_sell_vol = sell_capacity - tight_sell_vol

            if tight_sell_vol < 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_ask + 1, deep_sell_vol))

        return orders

    def compute_pepper_root_strategy_CORE_EXPLORE(self, state: TradingState, order_depth: OrderDepth, prod_history: dict):
        orders: List[Order] = []
        position = state.position.get("INTARIAN_PEPPER_ROOT", 0)

        LIMIT = 80
        buy_capacity = LIMIT - position
        sell_capacity = -LIMIT - position

        current_time = state.timestamp
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 20000

        # --- 1. DAY-AWARE GOD-MODE ANCHOR ---
        MACRO_SLOPE = 0.001

        prev_time = prod_history.get("prev_time", -1)
        is_new_day = current_time == 0 or current_time < prev_time or (current_time % 1000000 == 0)

        if prod_history.get("day_open_price") is None or is_new_day:
            day_open_price = (best_bid + best_ask) / 2.0
            prod_history["day_open_price"] = day_open_price
            prod_history["day_start_time"] = current_time
        else:
            day_open_price = prod_history["day_open_price"]

        day_start_time = prod_history.get("day_start_time", 0)
        rel_time = current_time - day_start_time
        fair_value = day_open_price + (rel_time * MACRO_SLOPE)
        prod_history["prev_time"] = current_time

        is_startup = rel_time < 5000 and buy_capacity > 0

        # --- 2. AGGRESSIVE TAKER ACTIONS ---
        if sell_capacity < 0:
            if len(order_depth.buy_orders) > 0:
                for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price >= fair_value + 7 and sell_capacity < 0:
                        take_vol = max(sell_capacity, -bid_vol)
                        if take_vol < 0:
                            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, take_vol))
                            sell_capacity -= take_vol

        if buy_capacity > 0:
            if len(order_depth.sell_orders) > 0:
                for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                    max_acceptable_ask = fair_value + 15 if is_startup else fair_value + 6
                    if ask_price <= max_acceptable_ask and buy_capacity > 0:
                        take_vol = min(buy_capacity, abs(ask_vol))
                        if take_vol > 0:
                            orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, take_vol))
                            buy_capacity -= take_vol

        # --- 3. PASSIVE MAKER QUOTES ---
        if sell_capacity < 0:
            t1_vol = int(sell_capacity * 1 / 2)
            t2_vol = sell_capacity - t1_vol

            if t1_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(round(fair_value + 7)), t1_vol))
            if t2_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(round(fair_value + 8)), t2_vol))

        if buy_capacity > 0:
            if is_startup:
                my_bid = min(int(round(fair_value + 10)), best_bid + 2)
            else:
                my_bid = min(int(round(fair_value + 4)), best_bid + 1)

            orders.append(Order("INTARIAN_PEPPER_ROOT", my_bid, buy_capacity))

        prod_history["last_price"] = fair_value

        return orders, prod_history
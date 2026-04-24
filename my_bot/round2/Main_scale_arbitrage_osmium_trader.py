from typing import List
from datamodel import OrderDepth, TradingState, Order
import json


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

        # ── Parametri ─────────────────────────────────────────────────────────
        FAIR_VALUE = 10000
        LIMIT = 80

        ARB_EDGE_MIN = 5
        ARB_BUDGET_FRAC = 0.3

        # ── Stato ─────────────────────────────────────────────────────────────
        current_pos = state.position.get(product, 0)
        inventory_ratio = current_pos / LIMIT

        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        long_pressure = max(0.0,  inventory_ratio)
        short_pressure = max(0.0, -inventory_ratio)

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: ARBITRAGE
        # ──────────────────────────────────────────────────────────────────────
        arb_buy_scale = max(0.0, 1.0 - long_pressure ** 2)
        arb_sell_scale = max(0.0, 1.0 - short_pressure ** 2)

        arb_buy_cap = int(buy_capacity * ARB_BUDGET_FRAC * arb_buy_scale)
        arb_sell_cap = int(sell_capacity * ARB_BUDGET_FRAC * arb_sell_scale)

        if order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price <= FAIR_VALUE - ARB_EDGE_MIN and arb_buy_cap > 0:
                    take_vol = min(arb_buy_cap, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order(product, ask_price, take_vol))
                        arb_buy_cap -= take_vol
                        buy_capacity -= take_vol
                else:
                    break

        if order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price >= FAIR_VALUE + ARB_EDGE_MIN and arb_sell_cap < 0:
                    take_vol = max(arb_sell_cap, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order(product, bid_price, take_vol))
                        arb_sell_cap -= take_vol
                        sell_capacity -= take_vol
                else:
                    break

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: DYNAMIC MARKET MAKING
        # ──────────────────────────────────────────────────────────────────────
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else FAIR_VALUE + 10
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else FAIR_VALUE - 10

        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        if buy_capacity > 0:
            tight_buy_vol = int(buy_capacity // 3)
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(Order(product, my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order(product, my_bid - 5, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = int(sell_capacity / 3)
            deep_sell_vol = sell_capacity - tight_sell_vol

            if tight_sell_vol < 0:
                orders.append(Order(product, my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order(product, my_ask + 5, deep_sell_vol))

        return orders

    def compute_pepper_root_strategy_CORE_EXPLORE(self, state: TradingState, order_depth: OrderDepth,
                                                  prod_history: dict):
        orders: List[Order] = []
        position = state.position.get("INTARIAN_PEPPER_ROOT", 0)

        # --- THE CORE REQUIREMENT: MAINTAIN +80 INVENTORY ---
        # If position is 0, we aggressively buy 80.
        # If position is 80, buy_capacity is 0. We hold and ride the trend.
        # If a trap triggers and we drop to 0 or -80, we buy back to +80.
        LIMIT = 80
        buy_capacity = LIMIT - position
        sell_capacity = -LIMIT - position

        current_time = state.timestamp

        fallback_mid = prod_history.get("last_price", 10000)
        if order_depth.buy_orders and order_depth.sell_orders:
            live_mid = (max(order_depth.buy_orders.keys()) +
                        min(order_depth.sell_orders.keys())) / 2.0
        else:
            live_mid = fallback_mid

        snapshots = prod_history.get("price_snapshots", [])
        if current_time % 5000 == 0:
            snapshots.append((current_time, live_mid))
            if len(snapshots) > 20:  # Keep 20k ticks of history
                snapshots.pop(0)
            prod_history["price_snapshots"] = snapshots

        # --- 1. DAY-AWARE GOD-MODE ANCHOR ---
        MACRO_SLOPE = prod_history.get("ACTIVE_SLOPE", 0.001)

        prev_time = prod_history.get("prev_time", -1)
        is_new_day = current_time == 0 or current_time < prev_time

        # DA RIMUOVERE per i giorni successivi? potenzialmente no
        if prod_history.get("day_open_price") is None or is_new_day:
            prod_history["day_open_price"] = live_mid
            prod_history["day_start_time"] = current_time

        day_open_price = prod_history["day_open_price"]

        day_start_time = prod_history.get("day_start_time", 0)
        rel_time = current_time - day_start_time
        fair_value = day_open_price + (rel_time * MACRO_SLOPE)
        prod_history["prev_time"] = current_time

        # void order book?
        best_bid = max(order_depth.buy_orders.keys(
        )) if order_depth.buy_orders else int(round(fair_value - 10))
        best_ask = min(order_depth.sell_orders.keys(
        )) if order_depth.sell_orders else int(round(fair_value + 10))

        # --- THE STARTUP SWITCH ---
        # If we are within the first 5,000 timestamps of a new day and we are missing shares,
        # we enter "Panic Buy" mode to instantly lock in our 80 inventory for the macro trend.
        is_startup = rel_time < 10000 and buy_capacity > 0

        # =========================================================================
        # 🛡️ SECURITY LAYER 1: THE PERSISTENT CIRCUIT BREAKER
        # Trend must remain broken for 15 consecutive ticks before we panic.
        # =========================================================================
        broken_ticks = prod_history.get("broken_ticks", 0)

        if live_mid < fair_value - 100 or live_mid > fair_value + 100:
            broken_ticks += 1
        else:
            broken_ticks = 0  # Reset immediately if the market touches our line again

        prod_history["broken_ticks"] = broken_ticks

        if broken_ticks >= 50:

            # 1. Compute the new slope right here using recent history
            snapshots = prod_history.get("price_snapshots", [])
            new_slope = 0.001  # Fallback
            if len(snapshots) >= 5:
                first_ts, first_price = snapshots[0]
                last_ts, last_price = snapshots[-1]
                if last_ts > first_ts:
                    new_slope = (last_price - first_price) / \
                                (last_ts - first_ts)

            # 2. OVERRIDE the constants in memory "from now on"
            prod_history["ACTIVE_SLOPE"] = new_slope
            prod_history["day_open_price"] = live_mid
            prod_history["day_start_time"] = current_time

            # 3. Reset the broken ticks so we don't instantly trigger this again on the next tick
            prod_history["broken_ticks"] = 0

            # 4. Override fair_value locally (optional since you return below, but good practice)
            fair_value = live_mid

            if position > 0:
                # Liquidate into whatever bids exist
                orders.append(
                    Order("INTARIAN_PEPPER_ROOT", best_bid, -position))

            return orders, prod_history  # Evacuate and skip all other logic
        is_startup = rel_time < 500 and buy_capacity > 0

        # --- 2. AGGRESSIVE TAKER ACTIONS ---

        # OPEN SHORT (OR DUMP CORE): If an NPC bids >= FV + 7, smash it instantly.
        if sell_capacity < 0:
            if len(order_depth.buy_orders) > 0:
                for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price >= fair_value + 7 and sell_capacity < 0:
                        take_vol = max(sell_capacity, -bid_vol)
                        if take_vol < 0:
                            orders.append(
                                Order("INTARIAN_PEPPER_ROOT", bid_price, take_vol))
                            sell_capacity -= take_vol

        # ACCUMULATE CORE (THE SWITCH APPLIED)
        if buy_capacity > 0:
            if len(order_depth.sell_orders) > 0:
                for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):

                    # THE SWITCH: If startup, we accept ANY ask up to FV + 15 to force the fill.
                    # Otherwise, we use your highly tuned normal tolerance of FV + 6.
                    max_acceptable_ask = fair_value + 20 if is_startup else fair_value + 6

                    if ask_price <= max_acceptable_ask and buy_capacity > 0:
                        take_vol = min(buy_capacity, abs(ask_vol))
                        if take_vol > 0:
                            orders.append(
                                Order("INTARIAN_PEPPER_ROOT", ask_price, take_vol))
                            buy_capacity -= take_vol

        # --- 3. PASSIVE MAKER QUOTES ---

        my_ask1 = int(round(fair_value + 7))
        my_ask2 = int(round(fair_value + 15))

        my_bid_startup = min(int(round(fair_value + 8)), best_bid + 1)
        # TIGHT BID: For normal market conditions
        my_bid_normal = min(int(round(fair_value + 4)), best_bid + 1)
        my_bid = my_bid_startup if is_startup else my_bid_normal

        my_bid = min(my_bid, my_ask1 - 1)

        # QUOTE ASKS (The Traps): Tuned to your precise +7 and +8 layers.
        if sell_capacity < 0:
            t1_vol = int(sell_capacity * 1 / 2)
            t2_vol = sell_capacity - t1_vol

            if t1_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                                    my_ask1, t1_vol))
            if t2_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                                    my_ask2, t2_vol))

        # QUOTE BIDS (The Vacuum - SWITCH APPLIED)
        if buy_capacity > 0:
            if buy_capacity > 0:
                orders.append(
                    Order("INTARIAN_PEPPER_ROOT", my_bid, buy_capacity))

        prod_history["last_price"] = fair_value

        return orders, prod_history

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

        # 2. OVERNIGHT GAP PROTECTION (Wipe trend memory on new days)
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
                # Osmium remains completely stateless and unchanged
                result[product] = self.osmium_strategy(
                    state, order_depth)

            elif product == "INTARIAN_PEPPER_ROOT":
                # Fetch Pepper Root's specific memory
                prod_history = history.get(product, {})

                # Run the strategy (it returns orders AND updated memory)
                orders, updated_prod_history = self.compute_pepper_root_strategy_CORE_EXPLORE(
                    state, order_depth, prod_history)

                result[product] = orders
                history[product] = updated_prod_history  # Save it back

            else:
                result[product] = []

        # 4. PACK MEMORY
        traderData = json.dumps(history)

        return result, conversions, traderData
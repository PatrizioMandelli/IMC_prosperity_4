from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import json


class Trader:
    def __init__(self):
        # Initialize them on the very first tick
        self.short_ewma = None
        self.long_ewma = None

    def bid(self) -> int:
        # Placeholder for Round 2
        return 0

    def calculate_wall_mid(self, order_depth: OrderDepth) -> float:
        # Find the bid price with the largest absolute volume
        best_wall_bid = max(order_depth.buy_orders.items(),
                            key=lambda x: x[1])[0]

        # Sell volumes are negative, so we use min() to find the largest negative volume
        best_wall_ask = min(order_depth.sell_orders.items(),
                            key=lambda x: x[1])[0]

        return (best_wall_bid + best_wall_ask) / 2

    def osmium_strategy_0(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        return []

    def osmium_strategy_MAIN(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []

        FAIR_VALUE = 10000
        LIMIT = 80

        current_pos = state.position.get("ASH_COATED_OSMIUM", 0)

        # CAPACITY TRACKING
        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        # STEP 1: Take Liquidity - Arbitrage
        # Eat free money if it crosses Fair Value   % POTENTIAL: put size limit on this, to steal from deep in the book
        if len(order_depth.sell_orders) > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < FAIR_VALUE and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", ask_price, take_vol))
                        buy_capacity -= take_vol

        if len(order_depth.buy_orders) > 0:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > FAIR_VALUE and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", bid_price, take_vol))
                        sell_capacity -= take_vol

        # STEP 2: DYNAMIC MARKET MAKING
        # Pull the best market prices (Fallback to static if book is empty)
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else FAIR_VALUE - 5

        # Dynamic Pegging (else you don't trade)
        # Step 1 tick inside the spread for queue priority, but NEVER cross Fair Value safety net
        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                # Trail the deep quote 1 tick behind dynamic bid
                orders.append(Order("ASH_COATED_OSMIUM",
                              my_bid - 6, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = int(sell_capacity / 2)
            deep_sell_vol = sell_capacity - tight_sell_vol

            if tight_sell_vol < 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                # Trail the deep quote 1 tick behind dynamic ask
                orders.append(Order("ASH_COATED_OSMIUM",
                              my_ask + 6, deep_sell_vol))

        return orders

    def osmium_strategy_wait(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []

        FAIR_VALUE = 10000
        LIMIT = 80

        current_pos = state.position.get("ASH_COATED_OSMIUM", 0)
        virtual_pos = current_pos  # We must track our position as it changes DURING the tick!

        # CAPACITY TRACKING
        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        # STEP 1: Take Liquidity - Arbitrage
        if len(order_depth.sell_orders) > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < FAIR_VALUE and buy_capacity > 0:
                    edge = FAIR_VALUE - ask_price

                    # INVENTORY SCALING: Cap inventory based on the edge
                    if edge >= 5:
                        max_long = 80  # Massive swing: Allow full limit
                    elif edge >= 3:
                        max_long = 60  # Medium swing: Allow half limit
                    else:
                        max_long = 20  # 1-tick edge: Keep powder dry!

                    # Calculate how much more we can buy in this specific tier
                    tier_budget = max(0, max_long - virtual_pos)

                    # Take the minimum of our total capacity, the available volume, and our tier budget
                    take_vol = min(buy_capacity, abs(ask_vol), tier_budget)

                    if take_vol > 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", ask_price, take_vol))
                        buy_capacity -= take_vol
                        virtual_pos += take_vol  # Update our tracker!

        if len(order_depth.buy_orders) > 0:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > FAIR_VALUE and sell_capacity < 0:
                    edge = bid_price - FAIR_VALUE

                    # INVENTORY SCALING (Short side is negative!)
                    if edge >= 5:
                        max_short = -80
                    elif edge >= 3:
                        max_short = -60
                    else:
                        max_short = -20

                    tier_budget = min(0, max_short - virtual_pos)

                    # For negative numbers, max() brings us closer to 0
                    take_vol = max(sell_capacity, -bid_vol, tier_budget)

                    if take_vol < 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", bid_price, take_vol))
                        sell_capacity -= take_vol
                        virtual_pos += take_vol

        # STEP 2: DYNAMIC MARKET MAKING
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else FAIR_VALUE - 5

        # Dynamic Pegging
        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        # --- THE FIX: MAKER INVENTORY SCALING ---
        # We refuse to hold more than 15 units for a tiny 1-tick passive spread.
        TIGHT_MAX_POS = 15

        if buy_capacity > 0:
            # How much room is left under our strict 15-unit limit?
            tight_buy_room = max(0, TIGHT_MAX_POS - virtual_pos)

            # We take the minimum of half our budget OR whatever room is left in the tight limit
            tight_buy_vol = min(buy_capacity // 2, tight_buy_room)

            # ALL remaining budget is aggressively shoved into the deep safety net!
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM",
                              my_bid - 1, deep_buy_vol))

        if sell_capacity < 0:
            TIGHT_MIN_POS = -15  # Short side is negative

            # min() brings us closer to 0 for negative capacities
            tight_sell_room = min(0, TIGHT_MIN_POS - virtual_pos)

            # max() is used because we are dealing with negative volumes (-40 vs -10 -> we want -10)
            tight_sell_vol = max(int(sell_capacity / 2), tight_sell_room)
            deep_sell_vol = sell_capacity - tight_sell_vol

            if tight_sell_vol < 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order("ASH_COATED_OSMIUM",
                              my_ask + 1, deep_sell_vol))

        return orders

    def osmium_strategy_depth(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []

        FAIR_VALUE = 10000
        LIMIT = 80

        current_pos = state.position.get("ASH_COATED_OSMIUM", 0)

        # CAPACITY TRACKING
        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        # STEP 1: Take Liquidity - Arbitrage
        # Eat free money if it crosses Fair Value   % POTENTIAL: put size limit on this, to steal from deep in the book
        if len(order_depth.sell_orders) > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < FAIR_VALUE and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", ask_price, take_vol))
                        buy_capacity -= take_vol

        if len(order_depth.buy_orders) > 0:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > FAIR_VALUE and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", bid_price, take_vol))
                        sell_capacity -= take_vol

        # STEP 2: DYNAMIC MARKET MAKING
        # Pull the best market prices (Fallback to static if book is empty)
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else FAIR_VALUE - 5

        # Dynamic Pegging (else you don't trade)
        # Step 1 tick inside the spread for queue priority, but NEVER cross Fair Value safety net
        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                # Trail the deep quote 4 ticks behind our dynamic bid
                orders.append(Order("ASH_COATED_OSMIUM",
                              my_bid - 3, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = int(sell_capacity / 2)
            deep_sell_vol = sell_capacity - tight_sell_vol

            if tight_sell_vol < 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                # Trail the deep quote 4 ticks behind our dynamic ask
                orders.append(Order("ASH_COATED_OSMIUM",
                              my_ask + 4, deep_sell_vol))

        return orders

    def compute_pepper_root_strategy_0(self, state: TradingState, order_depth: OrderDepth, prod_history: dict):
        return [], prod_history

    def compute_pepper_root_strategy_BH(self, state: TradingState, order_depth: OrderDepth, prod_history: dict) -> Tuple[List[Order], dict]:
        orders: List[Order] = []
        position = state.position.get("INTARIAN_PEPPER_ROOT", 0)

        LIMIT = 80
        buy_capacity = LIMIT - position
        sell_capacity = -LIMIT - position

        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else 20000
        current_mid = (best_bid + best_ask) / 2.0
        current_time = state.timestamp

        # --- 1. THE GOD MODE DRIFT ANCHOR ---
        # We know the absolute mathematical truth: slope is 0.001
        MACRO_SLOPE = 0.001

        # We find our daily intercept
        day_open_price = prod_history.get("day_open_price")

        # If it's the very first tick of the day, lock in the intercept
        if day_open_price is None or current_time == 0:
            day_open_price = current_mid
            prod_history["day_open_price"] = day_open_price

        # The indestructible Fair Value line
        fair_value = day_open_price + (current_time * MACRO_SLOPE)

        # --- 2. OPPORTUNISTIC TAKER (The Anomaly Sniper) ---

        # Look for panic sellers dumping below our True Line
        if len(order_depth.sell_orders) > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < fair_value and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(
                            Order("INTARIAN_PEPPER_ROOT", ask_price, take_vol))
                        buy_capacity -= take_vol

        # Look for the massive +20/+26 tick spikes you found in the logs!
        if len(order_depth.buy_orders) > 0:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                # We only sell if the premium is massive (e.g., +15 ticks)
                if bid_price >= fair_value + 15 and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(
                            Order("INTARIAN_PEPPER_ROOT", bid_price, take_vol))
                        sell_capacity -= take_vol

        # --- 3. HYBRID MAKER (Accumulate & Trap) ---

        # ACCUMULATE: Magnetize to the True Line to safely build +80 inventory
        my_bid = min(int(round(fair_value)), best_bid + 1)
        if buy_capacity > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", my_bid, buy_capacity))

        # THE EXTREME TRAPS: Cast nets perfectly calibrated to your data
        if sell_capacity < 0:
            tier1_vol = int(sell_capacity / 3)
            tier2_vol = int(sell_capacity / 3)
            tier3_vol = sell_capacity - tier1_vol - tier2_vol

            # Based on your top 5 spikes (+20 to +26), we place traps precisely where
            # they are statistically guaranteed to catch the whales.
            if tier1_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value + 16)), tier1_vol))
            if tier2_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value + 20)), tier2_vol))
            if tier3_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value + 25)), tier3_vol))

        # Housekeeping
        prod_history["last_price"] = current_mid
        prod_history["last_fair_value"] = fair_value

        return orders, prod_history

    def compute_pepper_root_strategy_GRID(self, state: TradingState, order_depth: OrderDepth, prod_history: dict) -> Tuple[List[Order], dict]:
        orders: List[Order] = []
        position = state.position.get("INTARIAN_PEPPER_ROOT", 0)

        LIMIT = 80
        buy_capacity = LIMIT - position
        sell_capacity = -LIMIT - position

        current_time = state.timestamp

        # --- 1. THE MATHEMATICAL TRUTH ---
        # Using the exact OLS regression from your analysis
        MACRO_SLOPE = 0.001272

        # We auto-anchor the intercept on the very first tick of the day to protect
        # against different starting prices across Day -2, -1, and 0.
        day_open_price = prod_history.get("day_open_price")

        if day_open_price is None or current_time == 0:
            # Safely grab the mid, or fallback to the known Day -2 intercept
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                day_open_price = (best_bid + best_ask) / 2.0
            else:
                day_open_price = 11974.32  # Fallback to known math

            prod_history["day_open_price"] = day_open_price

        # The Indestructible Moving Centerline
        fair_value = day_open_price + (current_time * MACRO_SLOPE)

        # --- 2. THE ASCENDING GRID (Market Making the Canyon) ---

        # BUY SIDE (The Accumulation Net)
        # We know the spread is ~13 (FV +/- 6.5). We place bids just inside and deep outside.
        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value - 5)), tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value - 10)), deep_buy_vol))

        # SELL SIDE (The Whale Harpoons)
        # We target the exact percentiles from your distribution analysis
        if sell_capacity < 0:
            # Remember, this is a negative number
            tier1_vol = int(sell_capacity / 2)
            tier2_vol = sell_capacity - tier1_vol

            if tier1_vol < 0:
                # 95th Percentile Trap (+17.80)
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value + 17)), tier1_vol))
            if tier2_vol < 0:
                # Maximum Anomaly Trap (+24.50 to +26.90)
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value + 24)), tier2_vol))

        # We keep the JSON tiny.
        return orders, prod_history

    def compute_pepper_root_strategy_SNIPER(self, state: TradingState, order_depth: OrderDepth, prod_history: dict):
        orders: List[Order] = []
        position = state.position.get("INTARIAN_PEPPER_ROOT", 0)

        LIMIT = 80
        buy_capacity = LIMIT - position
        sell_capacity = -LIMIT - position

        current_time = state.timestamp
        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else 20000
        current_mid = (best_bid + best_ask) / 2.0

        # --- 0. TELEMETRY: ALL OWN TRADES ---
        # We removed the "SUBMISSION" check. If it's in own_trades, WE did it.
        last_fv = prod_history.get("last_price", 0)

        if last_fv > 0 and "INTARIAN_PEPPER_ROOT" in state.own_trades:
            for trade in state.own_trades["INTARIAN_PEPPER_ROOT"]:
                if trade.timestamp == current_time - 100:
                    print(
                        f"\n🚨 [TS {current_time}] EXECUTED TRADE! Qty: {trade.quantity} @ Price: {trade.price}")
                    print(
                        f"   -> Previous FV: {last_fv:.2f} | Distance from FV: {trade.price - last_fv:.2f} ticks")

        # --- 1. THE DAY-AWARE GOD-MODE ANCHOR (With Warm-Up) ---
        MACRO_SLOPE = 0.001

        prev_time = prod_history.get("prev_time", -1)
        is_new_day = current_time == 0 or current_time < prev_time or (
            current_time % 1000000 == 0)

        # WARM-UP: Instead of trusting Tick 0, we average the first 5 ticks to filter out morning spikes
        warmup_prices = prod_history.get("warmup_prices", [])

        if is_new_day:
            warmup_prices = []
            prod_history["day_open_price"] = None  # Reset
            prod_history["day_start_time"] = current_time

        if prod_history.get("day_open_price") is None:
            warmup_prices.append(current_mid)
            prod_history["warmup_prices"] = warmup_prices

            # Use live mid while warming up
            day_open_price = current_mid

            if len(warmup_prices) >= 5:
                # Lock in the robust average as our anchor
                day_open_price = sum(warmup_prices) / len(warmup_prices)
                prod_history["day_open_price"] = day_open_price
                print(
                    f"\n⚓ [TS {current_time}] ANCHOR LOCKED: {day_open_price:.2f}")
        else:
            day_open_price = prod_history["day_open_price"]

        day_start_time = prod_history.get("day_start_time", 0)

        # Calculate robust FV
        rel_time = current_time - day_start_time
        fair_value = day_open_price + (rel_time * MACRO_SLOPE)
        prod_history["prev_time"] = current_time

        # --- 2. X-RAY HEARTBEAT ---
        # Every 10,000 ticks, print our internal state so we can see if we are misaligned
        if current_time % 10000 == 0 and current_time > 0:
            print(f"\n💓 [TS {current_time}] HEARTBEAT")
            print(f"   -> Bot Fair Value: {fair_value:.2f}")
            print(
                f"   -> Live Market Mid: {current_mid:.2f} (Spread: {best_bid} - {best_ask})")
            print(
                f"   -> Traps Deployed At: +12 ({fair_value + 12:.0f}) | +20 ({fair_value + 20:.0f})")

        # --- 3. PATIENT SNAP-BACK ---
        if buy_capacity > 0:
            if len(order_depth.sell_orders) > 0:
                for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                    if ask_price <= fair_value and buy_capacity > 0:
                        take_vol = min(buy_capacity, abs(ask_vol))
                        if take_vol > 0:
                            orders.append(
                                Order("INTARIAN_PEPPER_ROOT", ask_price, take_vol))
                            buy_capacity -= take_vol

            if buy_capacity > 0:
                my_bid = min(int(round(fair_value - 1)), best_bid + 1)
                orders.append(
                    Order("INTARIAN_PEPPER_ROOT", my_bid, buy_capacity))

        # --- 4. THE WHALE HARPOON ---
        if sell_capacity < 0:
            tier1_vol = int(sell_capacity / 2)
            tier2_vol = sell_capacity - tier1_vol

            if tier1_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value + 12)), tier1_vol))

            if tier2_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", int(
                    round(fair_value + 20)), tier2_vol))

        prod_history["last_price"] = fair_value

        return orders, prod_history

    def compute_pepper_root_strategy_BASICS(self, state: TradingState, order_depth: OrderDepth, prod_history: dict):
        orders: List[Order] = []
        position = state.position.get("INTARIAN_PEPPER_ROOT", 0)

        LIMIT = 80
        buy_capacity = LIMIT - position
        sell_capacity = -LIMIT - position

        # Safely get Top of Book
        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else 20000

        # --- 0. TELEMETRY: PRINT WHEN A SELL TRAP IS FILLED ---
        if "INTARIAN_PEPPER_ROOT" in state.own_trades:
            for trade in state.own_trades["INTARIAN_PEPPER_ROOT"]:
                # If we were the seller on the previous tick, our marked-up trap was hit!
                if trade.timestamp == state.timestamp - 100 and trade.seller == "SUBMISSION":
                    print(
                        f"[TS {state.timestamp}] 🎯 TRAP SPRUNG! Sold {trade.quantity} shares @ {trade.price}")

        # --- 1. INSTANTANEOUS VWAP (MICROPRICE) ---
        bid_vol = sum(order_depth.buy_orders.values())
        # Prosperity ask volumes are negative
        ask_vol = abs(sum(order_depth.sell_orders.values()))

        if bid_vol > 0 and ask_vol > 0:
            # Weight the bid price by the ask volume, and ask price by bid volume.
            # This perfectly balances the "True Mid" based on where the heavy liquidity is sitting.
            vwap_mid = (best_bid * ask_vol + best_ask *
                        bid_vol) / (bid_vol + ask_vol)
        else:
            vwap_mid = (best_bid + best_ask) / 2.0

        fair_value = vwap_mid

        # --- 2. BUY AND HOLD (Maintain +80) ---
        if buy_capacity > 0:
            # Aggressive Take: If someone panics and sells below our VWAP, snatch it instantly
            if len(order_depth.sell_orders) > 0:
                for ask_price, a_vol in sorted(order_depth.sell_orders.items()):
                    if ask_price <= fair_value and buy_capacity > 0:
                        take_vol = min(buy_capacity, abs(a_vol))
                        if take_vol > 0:
                            orders.append(
                                Order("INTARIAN_PEPPER_ROOT", ask_price, take_vol))
                            buy_capacity -= take_vol

            # Passive Accumulation: Place bids slightly below VWAP to soak up the rest cheaply
            if buy_capacity > 0:
                my_bid = min(int(round(fair_value - 1)), best_bid + 1)
                orders.append(
                    Order("INTARIAN_PEPPER_ROOT", my_bid, buy_capacity))

        # --- 3. LAYERED SELL TRAPS (The Markup) ---
        if sell_capacity < 0:
            # We quote different asks with different sizes depending on the markup.
            # We allocate more volume to the "likely" spikes, and save a little for the massive whales.

            t1_vol = int(sell_capacity * 0.4)  # 40% of inventory at +12 markup
            t2_vol = int(sell_capacity * 0.4)  # 40% of inventory at +18 markup
            t3_vol = sell_capacity - t1_vol - t2_vol  # Remaining 20% at +24 markup

            if t1_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                              int(round(fair_value + 12)), t1_vol))

            if t2_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                              int(round(fair_value + 18)), t2_vol))

            if t3_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                              int(round(fair_value + 24)), t3_vol))

        # We don't even need to use prod_history anymore. The bot is perfectly stateless!
        return orders, prod_history

    def compute_pepper_root_strategy_CORE_EXPLORE(self, state: TradingState, order_depth: OrderDepth, prod_history: dict):
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
        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else 20000

        # --- 1. DAY-AWARE GOD-MODE ANCHOR ---
        MACRO_SLOPE = 0.001

        prev_time = prod_history.get("prev_time", -1)
        is_new_day = current_time == 0 or current_time < prev_time or (
            current_time % 1000000 == 0)

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

        # --- THE STARTUP SWITCH ---
        # If we are within the first 5,000 timestamps of a new day and we are missing shares,
        # we enter "Panic Buy" mode to instantly lock in our 80 inventory for the macro trend.
        is_startup = rel_time < 5000 and buy_capacity > 0

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
                    max_acceptable_ask = fair_value + 15 if is_startup else fair_value + 6

                    if ask_price <= max_acceptable_ask and buy_capacity > 0:
                        take_vol = min(buy_capacity, abs(ask_vol))
                        if take_vol > 0:
                            orders.append(
                                Order("INTARIAN_PEPPER_ROOT", ask_price, take_vol))
                            buy_capacity -= take_vol

        # --- 3. PASSIVE MAKER QUOTES ---

        # QUOTE ASKS (The Traps): Tuned to your precise +7 and +8 layers.
        if sell_capacity < 0:
            t1_vol = int(sell_capacity * 1 / 2)
            t2_vol = sell_capacity - t1_vol

            if t1_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                              int(round(fair_value + 7)), t1_vol))
            if t2_vol < 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT",
                              int(round(fair_value + 8)), t2_vol))

        # QUOTE BIDS (The Vacuum - SWITCH APPLIED)
        if buy_capacity > 0:
            if is_startup:
                # PANIC QUOTE: If the asks were too thin, aggressively outbid the entire
                # market by 2 ticks to guarantee we are filled instantly on the next tick.
                my_bid = min(int(round(fair_value + 10)), best_bid + 2)
            else:
                # NORMAL CORE BUILDER: Cap at FV + 4 to ensure we don't overpay during normal trading.
                my_bid = min(int(round(fair_value + 4)), best_bid + 1)

            orders.append(Order("INTARIAN_PEPPER_ROOT", my_bid, buy_capacity))

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
                result[product] = self.osmium_strategy_0(state, order_depth)

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
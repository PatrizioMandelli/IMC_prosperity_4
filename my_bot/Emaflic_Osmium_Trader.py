from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            result[product] = self.osmium_strategy(state, state.order_depths[product])

        return result, conversions, ""

    def osmium_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        orders: List[Order] = []

        PRODUCT = "ASH_COATED_OSMIUM"
        LIMIT = 80
        FALLBACK_FV = 10000

        # --- TUNABLE PARAMS ---
        TIGHT_SPREAD = 3       # half-spread on tight quote (was 1)
        DEEP_SPREAD = 5        # half-spread on deep quote (was 3)
        SKEW_FACTOR = 0.05     # FV shift per unit of inventory
        TIGHT_FRAC = 0.5       # fraction of capacity on tight quote

        current_pos = state.position.get(PRODUCT, 0)

        # ==============================
        # STEP 0: DYNAMIC FAIR VALUE
        # ==============================
        # Microprice: volume-weighted mid, shifts FV toward the thinner side of the book
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            bid_vol = abs(order_depth.buy_orders[best_bid])
            ask_vol = abs(order_depth.sell_orders[best_ask])
            fair_value = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        else:
            fair_value = FALLBACK_FV

        # Inventory skew: if long, lower FV to sell more aggressively (and vice versa)
        fair_value -= current_pos * SKEW_FACTOR

        # Integer boundaries for quoting
        fv_int = int(round(fair_value))

        # CAPACITY TRACKING
        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos  # negative number

        # ==============================
        # STEP 1: TAKE LIQUIDITY (Arbitrage + Inventory Unwind)
        # ==============================
        # Take any ask below FV (buy cheap)
        if order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < fv_int and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order(PRODUCT, ask_price, take_vol))
                        buy_capacity -= take_vol
                # Inventory unwind: if short, also take asks AT fair value to flatten
                elif ask_price == fv_int and current_pos < -10 and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol), abs(current_pos))
                    if take_vol > 0:
                        orders.append(Order(PRODUCT, ask_price, take_vol))
                        buy_capacity -= take_vol

        # Take any bid above FV (sell expensive)
        if order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > fv_int and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order(PRODUCT, bid_price, take_vol))
                        sell_capacity -= take_vol
                # Inventory unwind: if long, also hit bids AT fair value to flatten
                elif bid_price == fv_int and current_pos > 10 and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol, -current_pos)
                    if take_vol < 0:
                        orders.append(Order(PRODUCT, bid_price, take_vol))
                        sell_capacity -= take_vol

        # ==============================
        # STEP 2: MARKET MAKING (wider spread, two-tier)
        # ==============================
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else fv_int + TIGHT_SPREAD + 2
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else fv_int - TIGHT_SPREAD - 2

        # Tight quote: peg inside the spread but never cross FV
        my_bid = min(fv_int - TIGHT_SPREAD, best_bid + 1)
        my_ask = max(fv_int + TIGHT_SPREAD, best_ask - 1)

        # Deep quote: further out for capturing wider moves
        my_deep_bid = fv_int - DEEP_SPREAD
        my_deep_ask = fv_int + DEEP_SPREAD

        # --- BUY SIDE ---
        if buy_capacity > 0:
            tight_buy_vol = int(buy_capacity * TIGHT_FRAC)
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(Order(PRODUCT, my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order(PRODUCT, my_deep_bid, deep_buy_vol))

        # --- SELL SIDE ---
        if sell_capacity < 0:
            tight_sell_vol = int(sell_capacity * TIGHT_FRAC)
            deep_sell_vol = sell_capacity - tight_sell_vol

            if tight_sell_vol < 0:
                orders.append(Order(PRODUCT, my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order(PRODUCT, my_deep_ask, deep_sell_vol))

        return orders

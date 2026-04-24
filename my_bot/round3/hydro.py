"""Inventory-skewed PRICING — push asks down when long, bids up when short."""
import json
from typing import List
from datamodel import OrderDepth, TradingState, Order
import math


class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        try:
            memory = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            memory = {}

        product = "HYDROGEL_PACK"
        if product in state.order_depths:
            result[product] = self.logic(state, state.order_depths[product], memory)

        return result, conversions, json.dumps(memory)

    def logic(self, state: TradingState, depth: OrderDepth, memory: dict) -> List[Order]:
        orders: List[Order] = []
        product = "HYDROGEL_PACK"

        if not depth.sell_orders or not depth.buy_orders:
            return orders

        LIMIT = 200
        current_pos = state.position.get(product, 0)

        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())

        def deep_w(om, rev):
            t = 0; ws = 0
            for p in sorted(om.keys(), reverse=rev)[:3]:
                v = abs(om[p])
                ws += p * v
                t += v
            return ws / t if t else None

        w_bid = deep_w(depth.buy_orders, True)
        w_ask = deep_w(depth.sell_orders, False)
        deep_mid = (w_bid + w_ask) / 2.0 if w_bid and w_ask else (best_bid + best_ask) / 2.0

        a_fast = 0.35
        a_slow = 0.01
        fast_fair = a_fast * deep_mid + (1 - a_fast) * memory.get("fast_fair", deep_mid)
        slow_fair = a_slow * deep_mid + (1 - a_slow) * memory.get("slow_fair", deep_mid)
        memory["fast_fair"] = fast_fair
        memory["slow_fair"] = slow_fair
        fair_value = 0.7 * fast_fair + 0.3 * slow_fair

        # Inv skew (size only)
        if abs(current_pos) > 160:
            sm = ((abs(current_pos) - 160) / 40.0) ** 2
            sg = 1 if current_pos > 0 else -1
            inv_skew = sg * sm * 10 + (current_pos / 75.0)
        else:
            inv_skew = current_pos / 75.0
        adjusted_fair = fair_value - inv_skew

        buy_cap = LIMIT - current_pos
        sell_cap = -LIMIT - current_pos
        MAX_TAKER = 25

        if best_ask < adjusted_fair - 2.0 and buy_cap > 0:
            qty = min(buy_cap, abs(depth.sell_orders[best_ask]), MAX_TAKER)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_cap -= qty
        if best_bid > adjusted_fair + 2.0 and sell_cap < 0:
            qty = max(sell_cap, -depth.buy_orders[best_bid], -MAX_TAKER)
            if qty < 0:
                orders.append(Order(product, best_bid, qty))
                sell_cap -= abs(qty)

        # ASYMMETRIC sizing
        pos_ratio = current_pos / float(LIMIT)
        BASE = 190
        bid_mult = math.exp(-2.0 * pos_ratio) if pos_ratio > 0 else (1 - pos_ratio)
        ask_mult = math.exp(2.0 * pos_ratio) if pos_ratio < 0 else (1 + pos_ratio)
        bid_size = max(5, min(195, int(BASE * bid_mult)))
        ask_size = max(5, min(195, int(BASE * ask_mult)))

        # ASYMMETRIC PRICING — when long, push ask price down (more aggressive sell)
        # When short, push bid price up (more aggressive buy)
        bid_offset = 1  # default penny-jump
        ask_offset = 1
        if pos_ratio > 0.4:
            ask_offset = 2  # more aggressive sell
        if pos_ratio > 0.7:
            ask_offset = 3
        if pos_ratio < -0.4:
            bid_offset = 2
        if pos_ratio < -0.7:
            bid_offset = 3

        my_bid = best_bid + bid_offset
        my_ask = best_ask - ask_offset

        # Sanity: never cross
        if my_bid >= my_ask:
            my_bid = int(adjusted_fair - 1)
            my_ask = int(adjusted_fair + 1)

        # Also clamp via fair value (don't quote way above fair)
        my_bid = min(my_bid, int(math.floor(fair_value - 0.5)))
        my_ask = max(my_ask, int(math.ceil(fair_value + 0.5)))

        if buy_cap > 0:
            orders.append(Order(product, my_bid, min(buy_cap, bid_size)))
        if sell_cap < 0:
            orders.append(Order(product, my_ask, max(sell_cap, -ask_size)))

        return orders

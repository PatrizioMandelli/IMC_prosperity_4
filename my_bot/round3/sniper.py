import json
from typing import List, Tuple, Dict, Any
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
            result[product] = self.hydro_beast_v25_logic(state, state.order_depths[product], memory)

        return result, conversions, json.dumps(memory)

    def hydro_beast_v25_logic(self, state: TradingState, depth: OrderDepth, memory: dict) -> List[Order]:
        orders: List[Order] = []
        product = "HYDROGEL_PACK"

        if not depth.sell_orders or not depth.buy_orders:
            return orders

        # --- CONFIGURATION (v10 BASE + SNIPER TAKER) ---
        LIMIT = 200
        MAX_CHUNK_TAKER = 100 # SNIPER: Massive increase from 14
        MAX_CHUNK_MAKER = 90 # V10 safe spot
        current_pos = state.position.get(product, 0)

        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        
        # 1. ENHANCED MID PRICE
        def get_weighted_mid(orders_map, is_buy):
            total_vol = 0
            weighted_sum = 0
            sorted_prices = sorted(orders_map.keys(), reverse=is_buy)[:3]
            for p in sorted_prices:
                vol = abs(orders_map[p])
                weighted_sum += p * vol
                total_vol += vol
            return (weighted_sum / total_vol) if total_vol > 0 else None

        w_bid = get_weighted_mid(depth.buy_orders, True)
        w_ask = get_weighted_mid(depth.sell_orders, False)
        deep_mid = (w_bid + w_ask) / 2.0 if w_bid and w_ask else (best_bid + best_ask) / 2.0

        # 2. ADAPTIVE FAIR (v10 parameters)
        alpha_fast = 0.25 
        alpha_slow = 0.01
        
        fast_fair = memory.get("fast_fair", deep_mid)
        slow_fair = memory.get("slow_fair", deep_mid)
        
        fast_fair = alpha_fast * deep_mid + (1 - alpha_fast) * fast_fair
        slow_fair = alpha_slow * deep_mid + (1 - alpha_slow) * slow_fair
        
        memory["fast_fair"] = fast_fair
        memory["slow_fair"] = slow_fair
        
        fair_value = 0.7 * fast_fair + 0.3 * slow_fair

        # 3. RELAXED INVENTORY SKEW (v10 parameters)
        if abs(current_pos) > 160:
            skew_multiplier = ((abs(current_pos) - 160) / 40.0) ** 2
            skew_sign = 1 if current_pos > 0 else -1
            inv_skew = skew_sign * skew_multiplier * 10 + (current_pos / 60.0)
        else:
            inv_skew = (current_pos / 60.0) 
        
        adjusted_fair = fair_value - inv_skew

        # 4. MASSIVE TAKER LOGIC (Sniper)
        buy_cap = LIMIT - current_pos
        sell_cap = -LIMIT - current_pos

        if best_ask < adjusted_fair - 2.0 and buy_cap > 0:
            # Take exactly what's offered up to our massive limit
            qty_to_take = min(buy_cap, abs(depth.sell_orders[best_ask]), MAX_CHUNK_TAKER)
            orders.append(Order(product, best_ask, qty_to_take))
            buy_cap -= qty_to_take
            
        if best_bid > adjusted_fair + 2.0 and sell_cap < 0:
            # Sell exactly what's bid up to our massive limit
            qty_to_take = max(sell_cap, -depth.buy_orders[best_bid], -MAX_CHUNK_TAKER)
            orders.append(Order(product, best_bid, qty_to_take))
            sell_cap -= abs(qty_to_take)

        # 5. MAKER LOGIC (v10 parameters)
        my_bid = min(int(math.floor(adjusted_fair - 1)), best_bid + 1)
        my_ask = max(int(math.ceil(adjusted_fair + 1)), best_ask - 1)

        if my_bid >= my_ask:
            my_bid = int(adjusted_fair - 1)
            my_ask = int(adjusted_fair + 1)

        if buy_cap > 0:
            orders.append(Order(product, my_bid, min(buy_cap, MAX_CHUNK_MAKER)))
        if sell_cap < 0:
            orders.append(Order(product, my_ask, max(sell_cap, -MAX_CHUNK_MAKER)))

        return orders

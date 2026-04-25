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

        # Gestione di entrambi i prodotti
        for product in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
            if product in state.order_depths:
                result[product] = self.logic(state, product, state.order_depths[product], memory)

        return result, conversions, json.dumps(memory)

    def logic(self, state: TradingState, product: str, depth: OrderDepth, memory: dict) -> List[Order]:
        orders: List[Order] = []
        if not depth.sell_orders or not depth.buy_orders:
            return orders

        # --- COSTANTI E POSIZIONE ---
        LIMIT = 200
        current_pos = state.position.get(product, 0)
        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        pkey = product + "_"

        # --- CALCOLO DEEP MID (W-MID) ---
        def deep_w(om, rev):
            t = 0;
            ws = 0
            for p in sorted(om.keys(), reverse=rev)[:3]:
                v = abs(om[p])
                ws += p * v
                t += v
            return ws / t if t else None

        w_bid = deep_w(depth.buy_orders, True)
        w_ask = deep_w(depth.sell_orders, False)
        deep_mid = (w_bid + w_ask) / 2.0 if w_bid and w_ask else (best_bid + best_ask) / 2.0

        # --- AGGIORNAMENTO MEMORIA E VOLATILITÀ ---
        if pkey + "fast_fair" not in memory:
            memory[pkey + "fast_fair"] = deep_mid
            memory[pkey + "slow_fair"] = deep_mid
            memory[pkey + "var_est"] = 0.0
            memory[pkey + "prev_mid"] = deep_mid

        prev_mid = memory.get(pkey + "prev_mid", deep_mid)
        price_change = deep_mid - prev_mid
        a_var = 0.05
        memory[pkey + "var_est"] = a_var * (price_change ** 2) + (1 - a_var) * memory.get(pkey + "var_est", 0.0)
        memory[pkey + "prev_mid"] = deep_mid
        vol_est = math.sqrt(memory[pkey + "var_est"]) if memory[pkey + "var_est"] > 0 else 0.0

        # --- LOGICA EMA (Merge delle strategie) ---
        # Per Hydrogel usiamo i pesi fissi del primo script, per Velvet quelli dinamici
        diff = abs(memory[pkey + "fast_fair"] - memory[pkey + "slow_fair"])

        if product == "HYDROGEL_PACK":
            a_f, a_s = 0.35, 0.01
        else:
            a_f = 0.55 if diff > 1.5 else 0.35
            a_s = 0.01

        fast_fair = a_f * deep_mid + (1 - a_f) * memory[pkey + "fast_fair"]
        slow_fair = a_s * deep_mid + (1 - a_s) * memory[pkey + "slow_fair"]
        memory[pkey + "fast_fair"] = fast_fair
        memory[pkey + "slow_fair"] = slow_fair

        trend = fast_fair - slow_fair
        fair_value = 0.7 * fast_fair + 0.3 * slow_fair

        # --- REGIME DI TRADING E SKEW DIVISOR ---
        # Default (Hydrogel style)
        skew_div = 50.0
        max_maker = 195
        taker_threshold = 2.0
        max_taker = 50

        # Sovrascrittura regime basata sulla volatilità (da Velvet script)
        if vol_est > 3.5:
            skew_div = 18.0
            max_maker = 25
            taker_threshold = 1.5
            max_taker = 25
        elif vol_est > 1.5 and abs(trend) > 2.0:
            skew_div = 65.0
            max_maker = 190
            taker_threshold = 1.0
            max_taker = 75

        # --- SKEW AGGRESSIVO (Dal primo script) ---
        if abs(current_pos) > 130:
            sm = ((abs(current_pos) - 130) / 40.0) ** 3 / 2
            sg = 1 if current_pos > 0 else -1
            inv_skew = sg * sm * 10 + (current_pos / skew_div)
        else:
            inv_skew = current_pos / skew_div

        # Smorzamento skew in trend favorevole
        if vol_est <= 3.5 and abs(trend) > 2.0:
            if (trend > 0 and current_pos > 0) or (trend < 0 and current_pos < 0):
                inv_skew *= 0.5

        adjusted_fair = fair_value - inv_skew

        # --- TAKER LOGIC ---
        buy_cap = LIMIT - current_pos
        sell_cap = -LIMIT - current_pos

        for ask_price in sorted(depth.sell_orders.keys()):
            if ask_price < adjusted_fair - taker_threshold and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[ask_price]), max_taker)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_cap -= qty

        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_price > adjusted_fair + taker_threshold and sell_cap < 0:
                qty = max(sell_cap, -abs(depth.buy_orders[bid_price]), -max_taker)
                if qty < 0:
                    orders.append(Order(product, bid_price, qty))
                    sell_cap -= abs(qty)

        # --- MAKER LOGIC (Market Making) ---
        pos_ratio = current_pos / float(LIMIT)
        BASE = 190
        bid_mult = math.exp(-2.0 * pos_ratio) if pos_ratio > 0 else (1 - pos_ratio)
        ask_mult = math.exp(2.0 * pos_ratio) if pos_ratio < 0 else (1 + pos_ratio)
        bid_size = max(5, min(max_maker, int(BASE * bid_mult)))
        ask_size = max(5, min(max_maker, int(BASE * ask_mult)))

        bid_offset = 1
        ask_offset = 1
        if pos_ratio > 0.4: ask_offset = 2
        if pos_ratio > 0.7: ask_offset = 3
        if pos_ratio < -0.4: bid_offset = 2
        if pos_ratio < -0.7: bid_offset = 3

        my_bid = best_bid + bid_offset
        my_ask = best_ask - ask_offset

        if my_bid >= my_ask:
            my_bid = int(adjusted_fair - 1)
            my_ask = int(adjusted_fair + 1)

        # --- EMERGENCY EXIT (Symmetrical Exit Logic) ---
        bid_clamp_offset = -0.5
        ask_clamp_offset = 0.5
        if current_pos > 150:
            ask_clamp_offset = -2.0  # Svendi più facilmente
        elif current_pos < -150:
            bid_clamp_offset = 2.0  # Ricompra più facilmente

        my_bid = min(my_bid, int(math.floor(fair_value + bid_clamp_offset)))
        my_ask = max(my_ask, int(math.ceil(fair_value + ask_clamp_offset)))

        if buy_cap > 0:
            orders.append(Order(product, my_bid, min(buy_cap, bid_size)))
        if sell_cap < 0:
            orders.append(Order(product, my_ask, max(sell_cap, -ask_size)))

        return orders
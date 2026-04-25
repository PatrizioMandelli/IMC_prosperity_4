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

        # --- 1. PREPARAZIONE DATI BOOK (TOP 3 LIVELLI) ---
        def get_top_n(orders, n, reverse):
            prices = sorted(orders.keys(), reverse=reverse)[:n]
            return [[p, abs(orders[p])] for p in prices]

        curr_bids = get_top_n(depth.buy_orders, 3, True)
        curr_asks = get_top_n(depth.sell_orders, 3, False)
        prev_book = memory.get("prev_book", None)

        # --- 2. CALCOLO MULTI-LEVEL MICRO-PRICE (STATICA) ---
        v_bid_tot = sum(b[1] for b in curr_bids)
        v_ask_tot = sum(a[1] for a in curr_asks)

        # Prezzi medi pesati per lato
        w_bid_p = sum(b[0] * b[1] for b in curr_bids) / v_bid_tot if v_bid_tot > 0 else best_bid
        w_ask_p = sum(a[0] * a[1] for a in curr_asks) / v_ask_tot if v_ask_tot > 0 else best_ask

        # Fair Value statico (spinto verso il lato con meno volume)
        static_fair = (w_bid_p * v_ask_tot + w_ask_p * v_bid_tot) / (v_bid_tot + v_ask_tot)

        # --- 3. CALCOLO ORDER FLOW IMBALANCE (DINAMICA) ---
        ofi_signal = 0
        if prev_book:
            for i in range(3):
                # OFI Bids
                if i < len(curr_bids) and i < len(prev_book['bids']):
                    cp, cv = curr_bids[i];
                    pp, pv = prev_book['bids'][i]
                    if cp > pp:
                        ofi_signal += cv
                    elif cp < pp:
                        ofi_signal -= pv
                    else:
                        ofi_signal += (cv - pv)
                # OFI Asks
                if i < len(curr_asks) and i < len(prev_book['asks']):
                    cp, cv = curr_asks[i];
                    pp, pv = prev_book['asks'][i]
                    if cp < pp:
                        ofi_signal -= cv
                    elif cp > pp:
                        ofi_signal += pv
                    else:
                        ofi_signal -= (cv - pv)

        # Integrazione OFI nel Fair Value
        sensitivity = 0.002  # Calibrato per non sovrareagire
        fair_value = static_fair + (ofi_signal * sensitivity)

        # Salvataggio memoria
        memory["prev_book"] = {"bids": curr_bids, "asks": curr_asks}
        memory["fair_value"] = fair_value

        # --- 4. INVENTORY SKEWING ---
        # Più siamo sbilanciati, più spostiamo il FV per scoraggiare l'accumulo
        inv_skew = current_pos / 40.0
        if abs(current_pos) > 130:
            extra_skew = ((abs(current_pos) - 130) / 40.0) ** 1.5
            inv_skew += (1 if current_pos > 0 else -1) * extra_skew * 5

        adjusted_fair = fair_value - inv_skew

        # --- 5. LOGICA TAKER (AGGRESSIVA) ---
        buy_cap = LIMIT - current_pos
        sell_cap = -LIMIT - current_pos
        max_taker = 80 if abs(current_pos) < 100 else 30

        if best_ask < adjusted_fair - 1.1 and buy_cap > 0:
            qty = min(buy_cap, abs(depth.sell_orders[best_ask]), max_taker)
            orders.append(Order(product, best_ask, qty))
            buy_cap -= qty

        if best_bid > adjusted_fair + 1.1 and sell_cap < 0:
            qty = max(sell_cap, -depth.buy_orders[best_bid], -max_taker)
            orders.append(Order(product, best_bid, qty))
            sell_cap -= abs(qty)

        # --- 6. LOGICA MAKER (PASSIVA) ---
        pos_ratio = current_pos / float(LIMIT)
        base_size = 150

        # Dimensioni asimmetriche basate sulla posizione
        bid_sz = max(2, min(LIMIT, int(base_size * (math.exp(-2.0 * pos_ratio) if pos_ratio > 0 else (1 - pos_ratio)))))
        ask_sz = max(2, min(LIMIT, int(base_size * (math.exp(2.0 * pos_ratio) if pos_ratio < 0 else (1 + pos_ratio)))))

        # Offset lineare per stare larghi quando rischiosi
        slope = 2.5
        bid_off = 1 + int(max(0, -pos_ratio * slope))
        ask_off = 1 + int(max(0, pos_ratio * slope))

        my_bid = min(best_bid + bid_off, best_ask - 1)  # Evitiamo di crossare il book
        my_ask = max(best_ask - ask_off, best_bid + 1)

        # --- 7. SAFETY CLAMPS (EMERGENCY EXIT) ---
        # Se pos > 150, permettiamo di vendere anche sotto il FV per scaricare
        # Se pos < -150, permettiamo di comprare anche sopra il FV per coprire
        exit_margin = 1.5 if abs(current_pos) > 150 else 0.5

        # Prezzi limite "etici" basati sul Fair Value
        # Se sono short (pos < 0), il mio bid massimo può salire oltre il fair
        abs_max_bid = int(math.floor(fair_value + exit_margin)) if current_pos < 0 else int(math.floor(fair_value))
        # Se sono long (pos > 0), il mio ask minimo può scendere sotto il fair
        abs_min_ask = int(math.ceil(fair_value - exit_margin)) if current_pos > 0 else int(math.ceil(fair_value))

        my_bid = min(my_bid, abs_max_bid)
        my_ask = max(my_ask, abs_min_ask)

        if buy_cap > 0:
            orders.append(Order(product, my_bid, min(buy_cap, bid_sz)))
        if sell_cap < 0:
            orders.append(Order(product, my_ask, max(sell_cap, -ask_sz)))

        return orders
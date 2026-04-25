import json
import math
from typing import List, Dict
from datamodel import OrderDepth, TradingState, Order


class Trader:
    def run(self, state: TradingState):
        """
        Punto di ingresso richiesto dal backtester di Prosperity.
        """
        result = {}
        conversions = 0

        # 1. GESTIONE MEMORIA (Trader Data)
        try:
            # Carica i dati dai turni precedenti (fondamentale per medie e volatilità)
            memory = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            memory = {}

        # 2. SELEZIONE PRODOTTO
        # Applichiamo la logica Velvet "schiacciata" solo su Hydrogel
        product = "HYDROGEL_PACK"

        if product in state.order_depths:
            # Eseguiamo la logica e salviamo gli ordini nel dizionario dei risultati
            result[product] = self.compute_orders(state, product, memory)

        # 3. RITORNO DEI DATI
        # Prosperity vuole: dizionario ordini, conversioni, e la memoria aggiornata in JSON
        return result, conversions, json.dumps(memory)

    def compute_orders(self, state: TradingState, product: str, memory: dict) -> List[Order]:
        """
        Logica core: Parametri Velvet applicati a Hydrogel.
        """
        orders: List[Order] = []
        depth = state.order_depths[product]

        if not depth.sell_orders or not depth.buy_orders:
            return orders

        # --- COSTANTI E POSIZIONE ---
        LIMIT = 200
        current_pos = state.position.get(product, 0)
        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        pkey = product + "_"

        # --- CALCOLO INDICATORI AVANZATI ---
        # W-Mid: Media pesata dei volumi per identificare il prezzo reale
        w_bid = self.get_weighted_price(depth.buy_orders, True)
        w_ask = self.get_weighted_price(depth.sell_orders, False)
        deep_mid = (w_bid + w_ask) / 2.0 if (w_bid and w_ask) else (best_bid + best_ask) / 2.0

        # Volatilità: Reagisce ai salti di prezzo
        vol_est = self.get_volatility_estimate(pkey, deep_mid, memory)

        # --- LOGICA EMA ADATTIVA (IL CUORE DEL VELVET) ---
        prev_fast = memory.get(pkey + "fast_fair", deep_mid)
        prev_slow = memory.get(pkey + "slow_fair", deep_mid)

        # Se la differenza tra le medie è alta (>1.5), l'algoritmo accelera (alpha 0.55)
        alpha_fast = 0.55 if abs(prev_fast - prev_slow) > 1.5 else 0.35
        alpha_slow = 0.01

        fast_fair = alpha_fast * deep_mid + (1 - alpha_fast) * prev_fast
        slow_fair = alpha_slow * deep_mid + (1 - alpha_slow) * prev_slow

        memory[pkey + "fast_fair"] = fast_fair
        memory[pkey + "slow_fair"] = slow_fair

        fair_value = 0.7 * fast_fair + 0.3 * slow_fair
        trend = fast_fair - slow_fair

        # --- REGIME DI TRADING (REATTIVO ALLA VOLATILITÀ) ---
        if vol_est > 3.5:
            # Emergenza: Mercato troppo nervoso, riduciamo esposizione
            skew_div, max_maker, taker_thr, max_taker = 18.0, 25, 1.5, 25
        elif vol_est > 1.5 and abs(trend) > 2.0:
            # Trend: Mercato direzionale, inseguiamo il prezzo
            skew_div, max_maker, taker_thr, max_taker = 65.0, 190, 1.0, 75
        else:
            # Calmo: Market Making standard (Hydrogel classico)
            skew_div, max_maker, taker_thr, max_taker = 50.0, 195, 2.0, 50

        # --- SKEW E FAIR VALUE AGGIUSTATO ---
        if abs(current_pos) > 130:
            sm = ((abs(current_pos) - 130) / 40.0) ** 3 / 2
            sg = 1 if current_pos > 0 else -1
            inv_skew = sg * sm * 10 + (current_pos / skew_div)
        else:
            inv_skew = current_pos / skew_div

        # Smorzamento se la posizione è "giusta" rispetto al trend
        if vol_est <= 3.5 and abs(trend) > 2.0:
            if (trend > 0 and current_pos > 0) or (trend < 0 and current_pos < 0):
                inv_skew *= 0.5

        adjusted_fair = fair_value - inv_skew

        # --- ESECUZIONE ORDINI TAKER ---
        buy_cap = LIMIT - current_pos
        sell_cap = -LIMIT - current_pos

        for prc, qty in sorted(depth.sell_orders.items()):
            if prc < adjusted_fair - taker_thr and buy_cap > 0:
                exec_qty = min(buy_cap, abs(qty), max_taker)
                orders.append(Order(product, prc, exec_qty))
                buy_cap -= exec_qty

        for prc, qty in sorted(depth.buy_orders.items(), reverse=True):
            if prc > adjusted_fair + taker_thr and sell_cap < 0:
                exec_qty = max(sell_cap, -abs(qty), -max_taker)
                orders.append(Order(product, prc, exec_qty))
                sell_cap -= abs(exec_qty)

        # --- ESECUZIONE ORDINI MAKER ---
        pos_ratio = current_pos / float(LIMIT)
        bid_sz = max(5, min(max_maker, int(190 * (math.exp(-2.0 * pos_ratio) if pos_ratio > 0 else (1 - pos_ratio)))))
        ask_sz = max(5, min(max_maker, int(190 * (math.exp(2.0 * pos_ratio) if pos_ratio < 0 else (1 + pos_ratio)))))

        # Spread dinamico
        off = 1
        if abs(pos_ratio) > 0.4: off = 2
        if abs(pos_ratio) > 0.7: off = 3

        # Uscita emergenza se posizione > 150
        bid_cl, ask_cl = -0.5, 0.5
        if current_pos > 150:
            ask_cl = -2.0
        elif current_pos < -150:
            bid_cl = 2.0

        my_bid = min(best_bid + off, int(math.floor(fair_value + bid_cl)))
        my_ask = max(best_ask - off, int(math.ceil(fair_value + ask_cl)))

        if buy_cap > 0: orders.append(Order(product, my_bid, min(buy_cap, bid_sz)))
        if sell_cap < 0: orders.append(Order(product, my_ask, max(sell_cap, -ask_sz)))

        return orders

    # --- METODI HELPER ---
    def get_weighted_price(self, oms, reverse):
        v_t, w_s = 0, 0
        for p in sorted(oms.keys(), reverse=reverse)[:3]:
            v = abs(oms[p]);
            w_s += p * v;
            v_t += v
        return w_s / v_t if v_t else None

    def get_volatility_estimate(self, pkey, mid, mem):
        prev = mem.get(pkey + "prev_mid", mid)
        diff = mid - prev
        var = 0.05 * (diff ** 2) + 0.95 * mem.get(pkey + "var_est", 0.0)
        mem[pkey + "var_est"], mem[pkey + "prev_mid"] = var, mid
        return math.sqrt(var) if var > 0 else 0.0ge
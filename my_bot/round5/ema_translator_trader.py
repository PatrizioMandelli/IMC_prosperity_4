import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    def __init__(self):
        self.limits = {
            "TRANSLATOR_ECLIPSE_CHARCOAL": 10,
            "TRANSLATOR_VOID_BLUE": 10,
            "TRANSLATOR_ASTRO_BLACK": 10
        }
        
        # Pair 1: ECLIPSE vs VOID
        self.p1_alpha = 6700
        self.p1_beta = 0.3
        self.p1_std = 315
        
        # Pair 2: ASTRO vs VOID (Note: ASTRO = alpha - beta * VOID)
        self.p2_alpha = 15500
        self.p2_beta = -0.55
        self.p2_std = 365

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        m_eclipse = self.get_mid_price(state.order_depths.get("TRANSLATOR_ECLIPSE_CHARCOAL"))
        m_void = self.get_mid_price(state.order_depths.get("TRANSLATOR_VOID_BLUE"))
        m_astro = self.get_mid_price(state.order_depths.get("TRANSLATOR_ASTRO_BLACK"))
        
        if m_void is None:
            return {}, 0, state.traderData

        # 1. Pair 1: ECLIPSE vs VOID
        if m_eclipse is not None:
            spread1 = m_eclipse - (self.p1_alpha + self.p1_beta * m_void)
            z1 = spread1 / self.p1_std
            self.apply_pair_logic(state, "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_VOID_BLUE", z1, result)

        # 2. Pair 2: ASTRO vs VOID
        if m_astro is not None:
            spread2 = m_astro - (self.p2_alpha + self.p2_beta * m_void)
            z2 = spread2 / self.p2_std
            self.apply_pair_logic(state, "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_VOID_BLUE", z2, result)

        return result, conversions, state.traderData

    def apply_pair_logic(self, state, p1, p2, z, result):
        pos1 = state.position.get(p1, 0)
        lim1 = self.limits[p1]

        depth1 = state.order_depths[p1]
        best_ask1, best_bid1 = min(depth1.sell_orders.keys()), max(depth1.buy_orders.keys())

        orders1 = result.get(p1, [])

        # OIB Skew
        oib1 = 0
        if depth1.buy_orders and depth1.sell_orders:
            b_vol = sum(depth1.buy_orders.values())
            s_vol = sum(abs(v) for v in depth1.sell_orders.values())
            oib1 = (b_vol - s_vol) / (b_vol + s_vol)

        entry_z = 1.0 - (oib1 * 0.2)

        # --- NUOVA LOGICA DI PRICING (MAKER) ---
        spread = best_ask1 - best_bid1

        # Prezzo a cui vogliamo COMPRARE (qty > 0)
        if spread > 1:
            my_bid_price = best_bid1 + 1  # Miglioriamo il bid per essere primi
        else:
            my_bid_price = best_bid1  # Spread al minimo, ci accodiamo

        # Prezzo a cui vogliamo VENDERE (qty < 0)
        if spread > 1:
            my_ask_price = best_ask1 - 1  # Miglioriamo l'ask per essere primi
        else:
            my_ask_price = best_ask1  # Spread al minimo, ci accodiamo
        # ---------------------------------------

        if z > entry_z:
            # Dobbiamo VENDERE (qty negativa)
            qty = -lim1 - pos1
            if qty < 0:
                # Usiamo my_ask_price INVECE di best_bid1
                orders1.append(Order(p1, my_ask_price, qty))

        elif z < -entry_z:
            # Dobbiamo COMPRARE (qty positiva)
            qty = lim1 - pos1
            if qty > 0:
                # Usiamo my_bid_price INVECE di best_ask1
                orders1.append(Order(p1, my_bid_price, qty))

        elif abs(z) < 0.2:
            # Chiusura posizione
            if pos1 > 0:
                # Abbiamo comprato, ora dobbiamo vendere per chiudere
                orders1.append(Order(p1, my_ask_price, -pos1))
            if pos1 < 0:
                # Abbiamo venduto, ora dobbiamo comprare per chiudere
                orders1.append(Order(p1, my_bid_price, -pos1))

        if orders1:
            result[p1] = orders1
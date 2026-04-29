from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
from collections import deque
import numpy as np


class Trader:
    LIMIT = 10

    HARD_SHORT = {"MICROCHIP_OVAL"}

    # CONFIGURAZIONE (direction, trend_w, s_start, s_end, threshold, short_w)
    # Aggiunto short_w=40 a SQUARE per proteggerlo dalle inversioni del Day 4
    CONFIGS = {
        "MICROCHIP_SQUARE": (+1, 80, 15, 5, 4.5, 40),
        "MICROCHIP_CIRCLE": (+1, 100, 10, 10, 8.0, None),
        "MICROCHIP_TRIANGLE": (-1, 100, 10, 10, 5.0, None),
        "MICROCHIP_RECTANGLE": (-1, 100, 10, 10, 5.0, 50),
    }

    def __init__(self):
        all_products = self.HARD_SHORT | set(self.CONFIGS.keys())
        # Calcoliamo la finestra massima necessaria per il deque
        max_w = max(cfg[1] for cfg in self.CONFIGS.values())
        self.prices: Dict[str, deque] = {
            p: deque(maxlen=max_w) for p in all_products
        }

    def mid(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    def _passive_order(self, symbol: str, depth: OrderDepth,
                       pos: int, target: int) -> List[Order]:
        """
        Esegue ordini passivi per gestire lo spread elevato (8-12 punti).
        SQUARE ha lo spread medio più alto (11.72)[cite: 1].
        """
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        orders: List[Order] = []
        diff = target - pos

        if diff > 0:
            # Voglio comprare — offro al bid+1 (passivo)
            orders.append(Order(symbol, best_bid + 1, diff))
        elif diff < 0:
            # Voglio vendere — offro all'ask-1 (passivo)
            orders.append(Order(symbol, best_ask - 1, diff))
        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # 1. Aggiorna prezzi[cite: 1]
        for p in self.prices:
            depth = state.order_depths.get(p)
            if depth:
                m = self.mid(depth)
                if m:
                    self.prices[p].append(m)

        # 2. OVAL: Short fisso sempre (l'asset più affidabile con slope negativa costante)[cite: 1]
        for p in self.HARD_SHORT:
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            pos = state.position.get(p, 0)
            orders = self._passive_order(p, depth, pos, -self.LIMIT)
            if orders:
                result[p] = orders

        # 3. Altri Asset: Logica Trend con protezione short-term[cite: 1]
        for p, (direction, trend_w, s_start, s_end, threshold, short_w) in self.CONFIGS.items():
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            pos = state.position.get(p, 0)
            hist = list(self.prices[p])

            if len(hist) < trend_w:
                continue

            # Calcolo Trend Lungo (Media Mobile su due segmenti)[cite: 1]
            start_val = np.mean(hist[:s_start])
            end_val = np.mean(hist[-s_end:])
            long_trend = end_val - start_val

            long_confirms = (direction > 0 and long_trend > threshold) or \
                            (direction < 0 and long_trend < -threshold)

            target = 0
            if long_confirms:
                # Se è presente un filtro short_w, verifichiamo il trend immediato per uscire prima[cite: 1]
                if short_w is not None:
                    short_hist = hist[-short_w:]
                    short_start = np.mean(short_hist[:5])
                    short_end = np.mean(short_hist[-5:])
                    short_trend = short_end - short_start

                    # Se il trend breve è coerente con la nostra direzione, manteniamo il target[cite: 1]
                    short_confirms = (direction > 0 and short_trend > 0) or \
                                     (direction < 0 and short_trend < 0)
                    target = self.LIMIT * direction if short_confirms else 0
                else:
                    target = self.LIMIT * direction

            # Esegui l'ordine solo se dobbiamo cambiare posizione[cite: 1]
            if pos != target:
                orders = self._passive_order(p, depth, pos, target)
                if orders:
                    result[p] = orders

        return result, 0, ""
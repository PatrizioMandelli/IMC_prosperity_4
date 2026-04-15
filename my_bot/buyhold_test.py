import json
from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # In questa strategia non serve memoria (traderData)
        traderData = ""

        for product in state.order_depths:
            if product != "INTARIAN_PEPPER_ROOT":
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # 1. POSIZIONE ATTUALE E LIMITI
            POSITION_LIMIT = 80
            current_pos = state.position.get(product, 0)

            # 2. CALCOLO QUANTO MANCA AL MASSIMO
            # Vogliamo stare sempre a +80
            quantity_to_buy = POSITION_LIMIT - current_pos

            # 3. ESECUZIONE (Se non siamo già a 80)
            if quantity_to_buy > 0:
                if len(order_depth.sell_orders) > 0:
                    # Troviamo il miglior prezzo di vendita (ask)
                    best_ask = min(order_depth.sell_orders.keys())

                    # Inviamo l'ordine per comprare tutto quello che serve
                    # al prezzo corrente per assicurarci l'esecuzione immediata
                    orders.append(Order(product, best_ask, quantity_to_buy))

            result[product] = orders

        return result, conversions, traderData
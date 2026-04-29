import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict


class Trader:
    def __init__(self):
        self.limit = 10

        # StatArb Parameters
        self.beta_sr = 2
        self.mean_sr = 32350
        self.std_sr = 860

        self.beta_to = 0.5
        self.mean_to = 5860
        self.std_to = 410

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask+ best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = ""

        # 1. Update Signals
        mid_sq = self.get_mid_price(state.order_depths.get("MICROCHIP_SQUARE"))
        mid_rect = self.get_mid_price(state.order_depths.get("MICROCHIP_RECTANGLE"))
        z_sr = 0
        if mid_sq is not None and mid_rect is not None:
            spread_sr = mid_sq + self.beta_sr * mid_rect
            z_sr = (spread_sr - self.mean_sr) / self.std_sr

        mid_tri = self.get_mid_price(state.order_depths.get("MICROCHIP_TRIANGLE"))
        mid_oval = self.get_mid_price(state.order_depths.get("MICROCHIP_OVAL"))
        z_to = 0
        if mid_tri is not None and mid_oval is not None:
            spread_to = mid_tri - self.beta_to * mid_oval
            z_to = (spread_to - self.mean_to) / self.std_to

        # 2. Trade Microchips
        products = ["MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE", "MICROCHIP_OVAL",
                    "MICROCHIP_CIRCLE"]

        for product in products:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid = (best_ask + best_bid) / 2.0

            # Fair Value calculation basato sui segnali StatArb
            fair_val = mid
            if product == "MICROCHIP_SQUARE":
                fair_val = mid - (z_sr * 12)
            elif product == "MICROCHIP_RECTANGLE":
                fair_val = mid - (z_sr * 6)
            elif product == "MICROCHIP_TRIANGLE":
                fair_val = mid - (z_to * 6)
            elif product == "MICROCHIP_OVAL":
                fair_val = mid + (z_to * 6)

            # --- A. TAKER LOGIC (Aggressiva) ---
            # Se il fair value è superiore al prezzo di vendita attuale, compriamo subito
            if fair_val > best_ask:
                buy_qty = min(order_depth.sell_orders[best_ask], self.limit - curr_pos)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    curr_pos += buy_qty

            # Se il fair value è inferiore al prezzo di acquisto attuale, vendiamo subito
            if fair_val < best_bid:
                sell_qty = max(-order_depth.buy_orders[best_bid], -self.limit - curr_pos)
                if sell_qty < 0:
                    orders.append(Order(product, best_bid, sell_qty))
                    curr_pos += sell_qty

            # --- B. MAKER LOGIC (Price Improvement) ---
            # Cerchiamo di essere i primi della coda migliorando il best bid/ask di 1 tick

            if curr_pos < self.limit:
                # Proviamo a metterci sopra il miglior compratore per essere eseguiti per primi
                bid_pr = best_bid + 1
                # Non superiamo mai il nostro fair value (per non comprare in perdita)
                # Sottraiamo un piccolo margine (1) per sicurezza
                bid_pr = min(bid_pr, int(math.floor(fair_val - 1)))

                # Se il prezzo calcolato è valido e non incrocia il book in modo errato
                if bid_pr < best_ask:
                    orders.append(Order(product, bid_pr, self.limit - curr_pos))

            if curr_pos > -self.limit:
                # Proviamo a metterci sotto il miglior venditore
                ask_pr = best_ask - 1
                # Non scendiamo mai sotto il fair value
                ask_pr = max(ask_pr, int(math.ceil(fair_val + 1)))

                if ask_pr > best_bid:
                    orders.append(Order(product, ask_pr, -self.limit - curr_pos))

            result[product] = orders

        return result, conversions, traderData
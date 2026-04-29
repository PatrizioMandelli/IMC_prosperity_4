from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
from collections import deque


class Trader:
    LIMIT = 10
    SKEW_THRESH = 9
    TREND_WINDOW = 50  # step per stimare il trend

    PRODUCTS = [
        "SLEEP_POD_COTTON",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_SUEDE",
    ]

    # Prodotti con trend forte → applichiamo trend filter

    TREND_FILTER = {"SLEEP_POD_LAMB_WOOL", "SLEEP_POD_COTTON", "SLEEP_POD_SUEDE"}

    def __init__(self):
        self.price_history: Dict[str, deque] = {
            p: deque(maxlen=self.TREND_WINDOW) for p in self.PRODUCTS
        }

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        for symbol in self.PRODUCTS:
            depth: OrderDepth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            mid = (best_bid + best_ask) / 2
            pos = state.position.get(symbol, 0)

            self.price_history[symbol].append(mid)

            buy_price = best_bid + 1
            sell_price = best_ask - 1

            if pos >= self.SKEW_THRESH:
                sell_price -= 1
            elif pos <= -self.SKEW_THRESH:
                buy_price += 1

            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

            buy_qty = max(0, self.LIMIT - pos)
            sell_qty = max(0, self.LIMIT + pos)

            # ── Trend filter solo su Lamb Wool ────────────────────────────
            if symbol in self.TREND_FILTER:
                hist = self.price_history[symbol]
                if len(hist) >= self.TREND_WINDOW:
                    trend = hist[-1] - hist[0]  # positivo = salita, negativo = discesa
                    if trend > 0:
                        # Prezzo in salita → non comprare, solo vendi
                        buy_qty = 0
                        # Smaltisci inventario long se presente
                        if pos > 0:
                            sell_qty = pos  # vendi solo quello che hai
                    elif trend < 0:
                        # Prezzo in discesa → non vendere, solo compra
                        sell_qty = 0
                        # Smaltisci inventario short se presente
                        if pos < 0:
                            buy_qty = -pos  # ricompra solo quello che hai

            orders: List[Order] = []
            if buy_qty > 0: orders.append(Order(symbol, buy_price, int(buy_qty)))
            if sell_qty > 0: orders.append(Order(symbol, sell_price, -int(sell_qty)))

            if orders:
                result[symbol] = orders

        return result, 0, ""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
from collections import deque
import numpy as np


class Trader:


    LIMIT    = 10
    WINDOW   = 200
    Z_ENTRY  = 1.5
    Z_EXIT   = 0
    TREND_W  = 50


    LEADER  = "PEBBLES_XL"
    BASKET  = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
    ALL     = BASKET + [LEADER]


    # Market making puro — nessun filtro
    MM_ONLY      = {"PEBBLES_S", "PEBBLES_L"}
    # Market making con trend filter
    TREND_FILTER = {"PEBBLES_M", "PEBBLES_XS"}


    def __init__(self):
        self.prices:      Dict[str, deque] = {p: deque(maxlen=max(self.WINDOW, self.TREND_W)) for p in self.ALL}
        self.spread_hist: deque            = deque(maxlen=self.WINDOW)


    def mid(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2


    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}


        # ── Aggiorna prezzi ───────────────────────────────────────────────
        mids = {}
        for p in self.ALL:
            depth = state.order_depths.get(p)
            if depth:
                m = self.mid(depth)
                if m:
                    self.prices[p].append(m)
                    mids[p] = m


        # ── Z-score XL vs basket ──────────────────────────────────────────
        z = 0.0
        if len(mids) == len(self.ALL):
            basket_avg = np.mean([mids[p] for p in self.BASKET])
            spread = mids[self.LEADER] - basket_avg
            self.spread_hist.append(spread)
            if len(self.spread_hist) >= self.WINDOW:
                arr = np.array(self.spread_hist)
                std = arr.std()
                if std > 0:
                    z = (spread - arr.mean()) / std


        # ── Trading ───────────────────────────────────────────────────────
        for p in self.ALL:
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue


            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            pos      = state.position.get(p, 0)
            orders: List[Order] = []


            if p == self.LEADER:
                # ── XL: z-score ───────────────────────────────────────────
                if z > self.Z_ENTRY and pos > -self.LIMIT:
                    qty = self.LIMIT + pos
                    if qty > 0: orders.append(Order(p, best_bid, -qty))
                elif z < -self.Z_ENTRY and pos < self.LIMIT:
                    qty = self.LIMIT - pos
                    if qty > 0: orders.append(Order(p, best_ask, qty))
                elif abs(z) < self.Z_EXIT and pos != 0:
                    if pos > 0: orders.append(Order(p, best_bid, -pos))
                    else:       orders.append(Order(p, best_ask, -pos))
                else:
                    buy_qty  = max(0, self.LIMIT - pos)
                    sell_qty = max(0, self.LIMIT + pos)
                    if buy_qty  > 0: orders.append(Order(p, best_bid + 1,  int(buy_qty)))
                    if sell_qty > 0: orders.append(Order(p, best_ask - 1, -int(sell_qty)))


            elif p in self.MM_ONLY:
                # ── S e L: market making puro ─────────────────────────────
                buy_qty  = max(0, self.LIMIT - pos)
                sell_qty = max(0, self.LIMIT + pos)
                if buy_qty  > 0: orders.append(Order(p, best_bid + 1,  int(buy_qty)))
                if sell_qty > 0: orders.append(Order(p, best_ask - 1, -int(sell_qty)))


            elif p in self.TREND_FILTER:
                # ── M e XS: market making + trend filter ──────────────────
                hist = self.prices[p]
                buy_qty  = max(0, self.LIMIT - pos)
                sell_qty = max(0, self.LIMIT + pos)


                if len(hist) >= self.TREND_W:
                    trend = hist[-1] - hist[-self.TREND_W]
                    if trend > 0:
                        buy_qty = 0
                        sell_qty = min(self.LIMIT, pos + 5) if pos > 0 else self.LIMIT + pos
                    elif trend < 0:
                        sell_qty = 0
                        if pos < 0: buy_qty = -pos


                if buy_qty  > 0: orders.append(Order(p, best_bid + 1,  int(buy_qty)))
                if sell_qty > 0: orders.append(Order(p, best_ask - 1, -int(sell_qty)))


            if orders:
                result[p] = orders


        return result, 0, ""
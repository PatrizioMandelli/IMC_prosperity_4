import json
from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # ── Load persistent state ─────────────────────────────────────────
        td = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}

        avg_cost = td.get("avg_cost", 0.0)
        last_pos = td.get("last_pos", 0)
        last_ts = td.get("last_ts", -1)

        product = "ASH_COATED_OSMIUM"

        # ── Update avg_cost from own_trades ───────────────────────────────
        own_trades = state.own_trades.get(product, []) if state.own_trades else []
        new_last_ts = last_ts
        for t in sorted(own_trades, key=lambda x: x.timestamp):
            if t.timestamp <= last_ts:
                continue
            if t.buyer == "SUBMISSION" or t.buyer == "":
                q = t.quantity
            elif t.seller == "SUBMISSION" or t.buyer == "":
                q = -t.quantity
            else:
                continue

            new_pos = last_pos + q

            if new_pos == 0:
                avg_cost = 0.0
            elif last_pos == 0:
                avg_cost = float(t.price)
            elif (last_pos > 0) == (new_pos > 0):
                # stesso segno
                if abs(new_pos) > abs(last_pos):
                    # aumento posizione → weighted avg
                    avg_cost = (avg_cost * abs(last_pos) + t.price * abs(q)) / abs(new_pos)
                # riduzione → avg_cost invariato
            else:
                # flip di segno
                avg_cost = float(t.price)

            last_pos = new_pos
            if t.timestamp > new_last_ts:
                new_last_ts = t.timestamp

        # ── Strategy ──────────────────────────────────────────────────────
        if product in state.order_depths:
            result[product] = self.osmium_strategy(
                state, state.order_depths[product], avg_cost
            )

        # ── Persist state (sync last_pos con ground truth) ────────────────
        trader_data = json.dumps({
            "avg_cost": avg_cost,
            "last_pos": state.position.get(product, 0),
            "last_ts": new_last_ts,
        })
        return result, conversions, trader_data

    def osmium_strategy(
        self, state: TradingState, order_depth: OrderDepth, avg_cost: float
    ) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        orders: List[Order] = []

        FAIR_VALUE = 10000
        LIMIT = 80
        MARGIN = 6 # tick minimi di profit richiesti sopra/sotto avg_cost per chiudere

        current_pos = state.position.get(product, 0)
        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        # ── STEP 1: Arbitrage (invariato) ─────────────────────────────────
        if order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < FAIR_VALUE and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order(product, ask_price, take_vol))
                        buy_capacity -= take_vol
                else:
                    break

        if order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > FAIR_VALUE and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order(product, bid_price, take_vol))
                        sell_capacity -= take_vol
                else:
                    break

        # ── STEP 2: Market Making (invariato + clamp su avg_cost) ─────────
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else FAIR_VALUE - 5

        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        # Clamp asimmetrico: non vendo sotto avg_cost+MARGIN (long),
        # non compro sopra avg_cost-MARGIN (short).
        # Se il mercato non mi paga abbastanza, aspetto. FV garantito → pazienza paga.
        if current_pos > 0 and avg_cost > 0:
            floor_ask = int(round(avg_cost)) + MARGIN
            my_ask = max(my_ask, floor_ask)
        elif current_pos < 0 and avg_cost > 0:
            ceil_bid = int(round(avg_cost)) - MARGIN
            my_bid = min(my_bid, ceil_bid)

        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol
            if tight_buy_vol > 0:
                orders.append(Order(product, my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order(product, my_bid - 2, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = int(sell_capacity / 2)
            deep_sell_vol = sell_capacity - tight_sell_vol
            if tight_sell_vol < 0:
                orders.append(Order(product, my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order(product, my_ask + 1, deep_sell_vol))

        return orders
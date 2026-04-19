from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            result[product] = self.osmium_strategy(state, state.order_depths[product])

        return result, conversions, ""

    def osmium_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        orders: List[Order] = []

        # ── Parametri ─────────────────────────────────────────────────────────
        FAIR_VALUE = 10000
        LIMIT = 80

        # Arb: filtro adverse selection + quota capacity
        ARB_EDGE_MIN = 5          # edge minimo in tick per entrare in arb
        ARB_BUDGET_FRAC = 0.3     # quota capacity riservata all'arb

        # Market making
        DEEP_OFFSET = 3           # distanza del deep level dal tight

        # ── Stato ─────────────────────────────────────────────────────────────
        current_pos = state.position.get(product, 0)
        inventory_ratio = current_pos / LIMIT            # ∈ [-1, +1]

        buy_capacity = LIMIT - current_pos               # >= 0
        sell_capacity = -LIMIT - current_pos             # <= 0

        # Pressure ∈ [0, 1] — positivo solo dal lato dove sei esposto
        long_pressure = max(0.0,  inventory_ratio)
        short_pressure = max(0.0, -inventory_ratio)

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: ARBITRAGE con soglia di edge
        # Filtro: entra in arb solo se edge >= ARB_EDGE_MIN tick.
        # Cap inventario quadratico: (1 - pressure^2).
        # ──────────────────────────────────────────────────────────────────────
        arb_buy_scale = max(0.0, 1.0 - long_pressure ** 2)
        arb_sell_scale = max(0.0, 1.0 - short_pressure ** 2)

        arb_buy_cap = int(buy_capacity * ARB_BUDGET_FRAC * arb_buy_scale)
        arb_sell_cap = int(sell_capacity * ARB_BUDGET_FRAC * arb_sell_scale)

        if order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price <= FAIR_VALUE - ARB_EDGE_MIN and arb_buy_cap > 0:
                    take_vol = min(arb_buy_cap, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order(product, ask_price, take_vol))
                        arb_buy_cap -= take_vol
                        buy_capacity -= take_vol
                else:
                    break

        if order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price >= FAIR_VALUE + ARB_EDGE_MIN and arb_sell_cap < 0:
                    take_vol = max(arb_sell_cap, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order(product, bid_price, take_vol))
                        arb_sell_cap -= take_vol
                        sell_capacity -= take_vol
                else:
                    break

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: MARKET MAKING senza skew prezzi
        # Prezzi: clamp standard FV±1, no skew.
        # Gestione inventario: solo via volume scaling quadratico.
        # ──────────────────────────────────────────────────────────────────────
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else FAIR_VALUE - 5

        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        # Scaling volume quadratico: tieni size pieno in zona safe, taglio ripido al limite
        mm_buy_scale = max(0.0, 1.0 - long_pressure ** 2)
        mm_sell_scale = max(0.0, 1.0 - short_pressure ** 2)

        eff_buy_cap = int(buy_capacity * mm_buy_scale)
        eff_sell_cap = int(sell_capacity * mm_sell_scale)

        # Split tight / deep con floor division (coerente su negativi)
        if eff_buy_cap > 0:
            tight_buy_vol = eff_buy_cap // 2
            deep_buy_vol = eff_buy_cap - tight_buy_vol
            if tight_buy_vol > 0:
                orders.append(Order(product, my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order(product, my_bid - DEEP_OFFSET, deep_buy_vol))

        if eff_sell_cap < 0:
            tight_sell_vol = eff_sell_cap // 2
            deep_sell_vol = eff_sell_cap - tight_sell_vol
            if tight_sell_vol < 0:
                orders.append(Order(product, my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order(product, my_ask + DEEP_OFFSET, deep_sell_vol))

        return orders
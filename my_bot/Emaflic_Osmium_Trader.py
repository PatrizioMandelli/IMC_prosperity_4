from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:
    """
    ASH_COATED_OSMIUM v4 — FV-anchored sizing
    ==========================================
    Core insight (user's): FV=10000 is fixed and known.
    Price always reverts. Every unit bought below 10000 will be sold above.
    Every unit sold above 10000 will be bought back below.

    Therefore: size should scale with distance from FV, not be fixed at half-capacity.

    Improvements over original:
    1. buy_size  = f(FV - mid): full capacity when far below FV, zero when above FV
       sell_size = f(mid - FV): full capacity when far above FV, zero when below FV
    2. Tight quote gets 80% of size when deviating (we want fills at best price)
       Deep quote gets 20% (keeps some liquidity deeper as backstop)
    3. Deep sell fixed to my_ask+2 (was asymmetric +1 in original)
    4. my_bid capped at FV-1, my_ask floored at FV+1 (same safety net)
    5. Step 1 arbitrage unchanged (still eats anything that crosses FV)
    """

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            result[product] = self.osmium_strategy(
                state, state.order_depths[product]
            )

        return result, conversions, ""

    def osmium_strategy(
        self, state: TradingState, order_depth: OrderDepth
    ) -> List[Order]:

        orders: List[Order] = []
        FV    = 10000
        LIMIT = 80
        # How many ticks of deviation = full capacity commitment
        # At DEV_FULL=5: mid=9995 or below -> buy full 80 units
        DEV_FULL = 5.0

        pos       = state.position.get("ASH_COATED_OSMIUM", 0)
        buy_cap   =  LIMIT - pos
        sell_cap  = -LIMIT - pos   # negative

        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else FV + 8
        best_bid = max(order_depth.buy_orders)  if order_depth.buy_orders  else FV - 8
        fair     = (best_ask + best_bid) / 2
        dev      = fair - FV   # negative = cheap, positive = expensive

        # ── Step 1: Take mispriced liquidity ──────────────────────────────────
        if order_depth.sell_orders:
            for ask_p, ask_v in sorted(order_depth.sell_orders.items()):
                if ask_p < FV and buy_cap > 0:
                    vol = min(buy_cap, abs(ask_v))
                    orders.append(Order("ASH_COATED_OSMIUM", ask_p, vol))
                    buy_cap -= vol

        if order_depth.buy_orders:
            for bid_p, bid_v in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_p > FV and sell_cap < 0:
                    vol = max(sell_cap, -bid_v)
                    orders.append(Order("ASH_COATED_OSMIUM", bid_p, vol))
                    sell_cap -= vol

        # ── Step 2: FV-anchored market making ─────────────────────────────────
        my_bid = min(FV - 1, best_bid + 1)
        my_ask = max(FV + 1, best_ask - 1)

        # Scale size with deviation from FV
        # buy_scale:  1.0 when dev <= -DEV_FULL, 0.0 when dev >= 0
        # sell_scale: 1.0 when dev >= +DEV_FULL, 0.0 when dev <= 0
        buy_scale  = max(0.0, min(1.0, -dev / DEV_FULL))
        sell_scale = max(0.0, min(1.0,  dev / DEV_FULL))

        # When scale is high (far from FV): 80% tight, 20% deep
        # When scale is low (near FV): 50% tight, 50% deep (standard symmetric MM)
        tight_frac = 0.5 + 0.3 * max(buy_scale, sell_scale)  # 0.5 to 0.8

        # Buy side
        if buy_cap > 0 and buy_scale > 0:
            total_buy = min(buy_cap, round(LIMIT * buy_scale))
            tight_vol = max(1, round(total_buy * tight_frac))
            deep_vol  = max(0, total_buy - tight_vol)

            orders.append(Order("ASH_COATED_OSMIUM", my_bid, tight_vol))
            if deep_vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_bid - 2, deep_vol))

        # Sell side
        if sell_cap < 0 and sell_scale > 0:
            total_sell = max(sell_cap, -round(LIMIT * sell_scale))
            tight_vol  = min(-1, -round(abs(total_sell) * tight_frac))
            deep_vol   = max(total_sell - tight_vol, 0)

            orders.append(Order("ASH_COATED_OSMIUM", my_ask, tight_vol))
            if deep_vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_ask + 2, -deep_vol))

        # Near FV (|dev| < 1): symmetric small MM to stay active
        if abs(dev) < 1.0:
            neutral_size = 15
            if buy_cap > 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_bid, min(neutral_size, buy_cap)))
            if sell_cap < 0:
                orders.append(Order("ASH_COATED_OSMIUM", my_ask, max(-neutral_size, sell_cap)))

        return orders
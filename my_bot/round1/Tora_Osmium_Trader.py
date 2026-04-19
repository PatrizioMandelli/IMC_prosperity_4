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
        orders: List[Order] = []

        FAIR_VALUE = 10000
        LIMIT = 80

        current_pos = state.position.get("ASH_COATED_OSMIUM", 0)

        # CAPACITY TRACKING
        buy_capacity = LIMIT - current_pos
        sell_capacity = -LIMIT - current_pos

        # STEP 1: Take Liquidity - Arbitrage
        # Eat free money if it crosses Fair Value   % POTENTIAL: put size limit on this, to steal from deep in the book
        if len(order_depth.sell_orders) > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < FAIR_VALUE and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", ask_price, take_vol))
                        buy_capacity -= take_vol

        if len(order_depth.buy_orders) > 0:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > FAIR_VALUE and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(
                            Order("ASH_COATED_OSMIUM", bid_price, take_vol))
                        sell_capacity -= take_vol

        # STEP 2: DYNAMIC MARKET MAKING
        # Pull the best market prices (Fallback to static if book is empty)
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else FAIR_VALUE - 5

        # Dynamic Pegging (else you don't trade)
        # Step 1 tick inside the spread for queue priority, but NEVER cross Fair Value safety net
        my_bid = min(FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(FAIR_VALUE + 1, best_ask - 1)

        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol

            if tight_buy_vol > 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                # Trail the deep quote 1 tick behind dynamic bid
                orders.append(Order("ASH_COATED_OSMIUM",
                                    my_bid - 2, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = int(sell_capacity / 2)
            deep_sell_vol = sell_capacity - tight_sell_vol

            if tight_sell_vol < 0:
                orders.append(
                    Order("ASH_COATED_OSMIUM", my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                # Trail the deep quote 1 tick behind dynamic ask
                orders.append(Order("ASH_COATED_OSMIUM",
                                    my_ask + 1, deep_sell_vol))

        return orders

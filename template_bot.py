# from datamodel import OrderDepth, TradingState, Order
# when uploading bot, change to above line and comment out the line below
from prosperity4bt.datamodel import OrderDepth, TradingState, Order

from typing import List, Dict


class Trader:
    def run(self, state: TradingState):
        # Dictionary to hold the orders we want to send to the exchange
        result: Dict[str, List[Order]] = {}

        # Iterate over all the products in the order depths
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # --- YOUR STRATEGY GOES HERE ---
            # Example:
            # if product == "EMERALDS":
            #     orders.append(Order(product, 10000, 1))

            result[product] = orders

        # Variables required by the engine
        conversions = 0
        traderData = state.traderData

        return result, conversions, traderData

import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    def __init__(self):
        self.limit = 10
        self.products = ['SLEEP_POD_COTTON', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_NYLON', 'SLEEP_POD_POLYESTER', 'SLEEP_POD_SUEDE']
        
        self.oracles = {
            'SLEEP_POD_POLYESTER': 'UV_VISOR_AMBER',
            'SLEEP_POD_SUEDE': 'MICROCHIP_SQUARE',
            'SLEEP_POD_NYLON': 'MICROCHIP_TRIANGLE',
            'SLEEP_POD_LAMB_WOOL': 'TRANSLATOR_ECLIPSE_CHARCOAL',
            'SLEEP_POD_COTTON': 'SLEEP_POD_POLYESTER'
        }
        
        self.params = {
            'SLEEP_POD_POLYESTER': {'alpha': 19140.15, 'beta': -0.9226},
            'SLEEP_POD_SUEDE': {'alpha': 5257.67, 'beta': 0.4516},
            'SLEEP_POD_NYLON': {'alpha': 14314.54, 'beta': -0.4830},
            'SLEEP_POD_LAMB_WOOL': {'alpha': 17729.92, 'beta': -0.7162},
            'SLEEP_POD_COTTON': {'alpha': -729.98 / 0.9638, 'beta': 1 / 0.9638}
        }

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        for product in self.products:
            if product not in state.order_depths: continue
            
            oracle_name = self.oracles[product]
            if oracle_name not in state.order_depths: continue
            
            oracle_mid = self.get_mid_price(state.order_depths[oracle_name])
            if oracle_mid is None: continue
            
            order_depth = state.order_depths[product]
            if not order_depth.sell_orders or not order_depth.buy_orders:
                continue

            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)
            
            p = self.params[product]
            fair_val = p['alpha'] + p['beta'] * oracle_mid
            
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Use pennying strategy to stay at the extremes and ensure fills in 'worse' mode.
            # This follows the directive to use best_bid + 1 and best_ask - 1.
            bid_price = best_bid + 1
            ask_price = best_ask - 1
            
            if curr_pos < self.limit:
                orders.append(Order(product, bid_price, self.limit - curr_pos))
            if curr_pos > -self.limit:
                orders.append(Order(product, ask_price, -self.limit - curr_pos))
                
            result[product] = orders
            
        return result, conversions, state.traderData

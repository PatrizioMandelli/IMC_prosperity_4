import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    def __init__(self):
        self.limit = 10
        self.products = [
            'SNACKPACK_CHOCOLATE',
            'SNACKPACK_PISTACHIO',
            'SNACKPACK_RASPBERRY',
            'SNACKPACK_STRAWBERRY',
            'SNACKPACK_VANILLA'
        ]
        
        # Historical average prices for each flavor (approximate)
        self.avg_prices = {
            'SNACKPACK_CHOCOLATE': 10000.0,
            'SNACKPACK_PISTACHIO': 9500.0,
            'SNACKPACK_RASPBERRY': 10000.0,
            'SNACKPACK_STRAWBERRY': 10500.0,
            'SNACKPACK_VANILLA': 10000.0
        }
        
        # Relative weights (how much each flavor is of the total cluster sum)
        total_avg = sum(self.avg_prices.values())
        self.weights = {p: self.avg_prices[p] / total_avg for p in self.products}
        
        self.alpha = 0.05
        self.edge = 1.0
        self.risk_factor = 3.0

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except:
                data = {"ema_weights": self.weights}
        else:
            data = {"ema_weights": self.weights}

        current_prices = {}
        for p in self.products:
            mid = self.get_mid_price(state.order_depths.get(p))
            if mid is not None:
                current_prices[p] = mid
        
        if not current_prices:
            return result, conversions, json.dumps(data)

        cluster_sum = sum(current_prices.values())
        current_rel_weights = {p: current_prices[p] / cluster_sum for p in current_prices}
        
        # Target position for each product based on its deviation from historical relative weight
        for product in current_prices:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)
            
            # Update EMA weight
            if product in data["ema_weights"]:
                data["ema_weights"][product] = self.alpha * current_rel_weights[product] + (1 - self.alpha) * data["ema_weights"][product]
            else:
                data["ema_weights"][product] = current_rel_weights[product]

            # Deviation from EMA weight
            # If current weight > EMA weight, flavor is relatively expensive
            # We want a negative position
            # Scaling: a 0.001 (0.1%) weight deviation might justify a full position
            weight_dev = current_rel_weights[product] - data["ema_weights"][product]
            target_pos = -weight_dev * 5000.0 # Scaling factor to tune
            target_pos = max(min(target_pos, self.limit), -self.limit)
            
            fair_val = current_prices[product] + (target_pos - curr_pos) * self.risk_factor
            
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            bid_price = min(int(math.floor(fair_val - self.edge)), best_bid + 1)
            ask_price = max(int(math.ceil(fair_val + self.edge)), best_ask - 1)
            
            if curr_pos < self.limit:
                orders.append(Order(product, bid_price, self.limit - curr_pos))
            if curr_pos > -self.limit:
                orders.append(Order(product, ask_price, -self.limit - curr_pos))
                
            result[product] = orders
            
        return result, conversions, json.dumps(data)

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
        
        # Johansen weights
        self.weights = {
            'SNACKPACK_CHOCOLATE': 1.0000,
            'SNACKPACK_PISTACHIO': -0.2891,
            'SNACKPACK_RASPBERRY': -0.1172,
            'SNACKPACK_STRAWBERRY': 0.1744,
            'SNACKPACK_VANILLA': 0.7224
        }
        
        self.spread_mean = 15078.52
        self.spread_std = 62.33
        
        # Strawberry Oracle
        self.strawberry_oracle = 'UV_VISOR_AMBER'
        self.strawberry_beta = -0.3259
        
        self.alpha = 0.01
        self.risk_factor = 2.0
        self.edge = 1.0

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
                data = {"ema_mean": self.spread_mean, "strawberry_intercept": 13310.0}
        else:
            data = {"ema_mean": self.spread_mean, "strawberry_intercept": 13310.0}

        current_prices = {}
        for p in self.products:
            mid = self.get_mid_price(state.order_depths.get(p))
            if mid is not None:
                current_prices[p] = mid
        
        if len(current_prices) < len(self.products):
            return result, conversions, json.dumps(data)

        # 1. Calculate Weighted Spread
        current_spread = sum(current_prices[p] * self.weights[p] for p in self.products)
        data["ema_mean"] = self.alpha * current_spread + (1 - self.alpha) * data["ema_mean"]
        z_spread = (current_spread - data["ema_mean"]) / self.spread_std
        z_capped = max(min(z_spread, 2.0), -2.0)
        
        # 2. Strawberry Oracle Signal
        visor_mid = self.get_mid_price(state.order_depths.get(self.strawberry_oracle))
        strawberry_z = 0.0
        if visor_mid is not None:
            # Intercept adjustment
            strawberry_intercept = current_prices['SNACKPACK_STRAWBERRY'] - self.strawberry_beta * visor_mid
            data["strawberry_intercept"] = 0.05 * strawberry_intercept + (1 - 0.05) * data["strawberry_intercept"]
            strawberry_fair = data["strawberry_intercept"] + self.strawberry_beta * visor_mid
            strawberry_z = (current_prices['SNACKPACK_STRAWBERRY'] - strawberry_fair) / 5.0 # Local dev

        for product in self.products:
            mid = current_prices[product]
            curr_pos = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            
            # Weighted Target based on Johansen spread
            # Note: if weight is negative, a high spread means product is cheap relative to others
            # Target position sign should be -z * weight
            target_pos = -z_capped * self.weights[product] * 5.0 # Scaling
            
            # Special overlay for Strawberry
            if product == 'SNACKPACK_STRAWBERRY':
                target_pos -= strawberry_z * 3.0
            
            target_pos = max(min(target_pos, self.limit), -self.limit)
            
            fair_val = mid + (target_pos - curr_pos) * self.risk_factor
            
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

import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    def __init__(self):
        self.limit = 10
        self.alpha = 0.05
        self.products = ["UV_VISOR_MAGENTA", "UV_VISOR_ORANGE", "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_RED"]

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        data = json.loads(state.traderData) if state.traderData else {
            "ema_diff": {},
            "var_diff": {}
        }
        
        current_prices = {}
        for p in self.products:
            mid = self.get_mid_price(state.order_depths.get(p))
            if mid is not None:
                current_prices[p] = mid
                
        if len(current_prices) < len(self.products):
            return {}, 0, state.traderData

        signals = {p: 0.0 for p in self.products}

        # Calculate pairwise differences and Z-scores
        for i in range(len(self.products)):
            for j in range(i + 1, len(self.products)):
                p1 = self.products[i]
                p2 = self.products[j]
                
                pair_key = f"{p1}_{p2}"
                diff = current_prices[p1] - current_prices[p2]
                
                if pair_key not in data["ema_diff"]:
                    data["ema_diff"][pair_key] = diff
                    data["var_diff"][pair_key] = 0.0
                else:
                    data["ema_diff"][pair_key] = self.alpha * diff + (1 - self.alpha) * data["ema_diff"][pair_key]
                
                ema_diff = data["ema_diff"][pair_key]
                dev = diff - ema_diff
                
                data["var_diff"][pair_key] = self.alpha * (dev ** 2) + (1 - self.alpha) * data["var_diff"][pair_key]
                std_dev = math.sqrt(data["var_diff"][pair_key]) if data["var_diff"][pair_key] > 0 else 1.0
                
                z = dev / std_dev
                
                # If z > 0, P1 is relatively overvalued compared to P2
                # So we want to sell P1 (negative signal) and buy P2 (positive signal)
                # To prevent extreme trends, we cap the z signal at 3
                z_capped = max(min(z, 3.0), -3.0)
                
                signals[p1] -= z_capped
                signals[p2] += z_capped

        for product in self.products:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)
            
            mid = current_prices[product]
            
            # The signal tells us how much we want to hold
            # There are 4 pairs for each product. Max signal sum is 4 * 3 = 12.
            # We scale it to be a target position.
            target_pos = signals[product] * (self.limit / 6.0) # tune scaling
            target_pos = max(min(target_pos, self.limit), -self.limit)
            
            # Fair value adjustment to attract executions towards target_pos
            # If target_pos > curr_pos, we want to buy, so we increase fair_val
            fair_val = mid + (target_pos - curr_pos) * 1.5
            
            edge = 1.5
            
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Improvement: place orders at best_bid + 1 and best_ask - 1 to ensure execution in 'worse' match mode
            # while still being constrained by our fair value calculation.
            bid_price = min(int(math.floor(fair_val - edge)), best_bid + 1)
            ask_price = max(int(math.ceil(fair_val + edge)), best_ask - 1)
            
            if curr_pos < self.limit:
                orders.append(Order(product, bid_price, self.limit - curr_pos))
            if curr_pos > -self.limit:
                orders.append(Order(product, ask_price, -self.limit - curr_pos))
                
            result[product] = orders
            
        return result, conversions, json.dumps(data)

from datamodel import Order, TradingState, OrderDepth
import collections
from typing import List, Dict
import json
import math

class Trader:
    def __init__(self):
        self.products = [
            'GALAXY_SOUNDS_BLACK_HOLES',
            'GALAXY_SOUNDS_DARK_MATTER',
            'GALAXY_SOUNDS_PLANETARY_RINGS',
            'GALAXY_SOUNDS_SOLAR_FLAMES',
            'GALAXY_SOUNDS_SOLAR_WINDS'
        ]
        
        # Best Oracles (target: (oracle, alpha, beta, std))
        self.configs = {
            'GALAXY_SOUNDS_BLACK_HOLES': ('PEBBLES_S', 20559.43, -1.0179, 446.23),
            'GALAXY_SOUNDS_DARK_MATTER': ('UV_VISOR_YELLOW', 6144.54, 0.3725, 211.76),
            'GALAXY_SOUNDS_PLANETARY_RINGS': ('GALAXY_SOUNDS_DARK_MATTER', 8025.67, 0.2608, 297.89),
            'GALAXY_SOUNDS_SOLAR_WINDS': ('PANEL_1X4', 15490.14, -0.5376, 302.85),
            'GALAXY_SOUNDS_SOLAR_FLAMES': ('GALAXY_SOUNDS_SOLAR_WINDS', 14003.29, -0.2789, 424.10)
        }
        
        self.pos_limit = 10

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = state.traderData or ""

        mids = {}
        for p in state.order_depths:
            mids[p] = self.get_mid_price(state.order_depths[p])

        for target in self.products:
            if target not in self.configs or target not in state.order_depths:
                continue
            
            oracle, alpha, beta, std = self.configs[target]
            oracle_mid = mids.get(oracle)
            if oracle_mid is None: continue
            
            fair_target = alpha + beta * oracle_mid
            current_pos = state.position.get(target, 0)
            
            target_depth = state.order_depths[target]
            buy_orders = collections.OrderedDict(sorted(target_depth.buy_orders.items(), reverse=True))
            sell_orders = collections.OrderedDict(sorted(target_depth.sell_orders.items()))
            
            spread = mids[target] - fair_target
            z_score = spread / std
            
            orders = []
            
            # Entry / Exit logic
            target_pos = 0
            if z_score < -1.0:
                target_pos = self.pos_limit
            elif z_score > 1.0:
                target_pos = -self.pos_limit
            elif abs(z_score) < 0.2:
                target_pos = 0
            else:
                # Hold current position if between 0.2 and 1.0 (hysteresis)
                target_pos = current_pos
            
            diff = target_pos - current_pos
            
            if diff > 0: # Buy
                # Take aggressive
                for price, vol in sell_orders.items():
                    if price <= fair_target:
                        buy_qty = min(diff, -vol)
                        orders.append(Order(target, price, buy_qty))
                        diff -= buy_qty
                    if diff <= 0: break
                # Place passive
                if diff > 0:
                    best_bid = max(target_depth.buy_orders.keys())
                    orders.append(Order(target, best_bid + 1, diff))
                    
            elif diff < 0: # Sell
                diff = -diff
                for price, vol in buy_orders.items():
                    if price >= fair_target:
                        sell_qty = min(diff, vol)
                        orders.append(Order(target, price, -sell_qty))
                        diff -= sell_qty
                    if diff <= 0: break
                if diff > 0:
                    best_ask = min(target_depth.sell_orders.keys())
                    orders.append(Order(target, best_ask - 1, -diff))

            result[target] = orders

        return result, conversions, traderData

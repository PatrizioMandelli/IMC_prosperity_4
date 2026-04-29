import json
import math
import numpy as np
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    def __init__(self):
        self.products = [
            "TRANSLATOR_ASTRO_BLACK",
            "TRANSLATOR_ECLIPSE_CHARCOAL",
            "TRANSLATOR_GRAPHITE_MIST",
            "TRANSLATOR_SPACE_GRAY",
            "TRANSLATOR_VOID_BLUE"
        ]
        self.limits = {p: 10 for p in self.products}
        self.alpha = 0.01  # Roughly 200-tick window
        self.entry_z = 1.5
        self.exit_z = 0.2

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        # Load state from traderData
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except:
                data = {}

        ema_state = data.get("ema", {})
        m2_state = data.get("m2", {})  # For rolling variance
        count = data.get("count", 0)

        # Get current mid prices
        mids = {}
        valid_mids = []
        for p in self.products:
            m = self.get_mid_price(state.order_depths.get(p))
            if m is not None:
                mids[p] = m
                valid_mids.append(m)
        
        if len(valid_mids) < 3:
            return result, conversions, state.traderData

        basket_mean = sum(valid_mids) / len(valid_mids)
        count += 1

        # Update EMAs and Variance
        signals = {}
        stds = {}
        for p in self.products:
            if p in mids:
                rel_price = mids[p] - basket_mean
                
                # Update EMA (Mean)
                curr_ema = ema_state.get(p, rel_price)
                new_ema = self.alpha * rel_price + (1 - self.alpha) * curr_ema
                ema_state[p] = new_ema
                
                # Update EMA (Variance using Welford-like EMA)
                diff = rel_price - new_ema
                curr_m2 = m2_state.get(p, 70.0**2)
                new_m2 = self.alpha * (diff**2) + (1 - self.alpha) * curr_m2
                m2_state[p] = new_m2
                
                stds[p] = math.sqrt(new_m2)
                signals[p] = diff

        # Execute trades
        for p in self.products:
            if p not in signals or p not in stds:
                continue
            
            sig = signals[p]
            std = stds[p]
            z = sig / std if std > 0 else 0
            
            pos = state.position.get(p, 0)
            limit = self.limits[p]
            depth = state.order_depths[p]
            
            best_ask = min(depth.sell_orders.keys())
            best_bid = max(depth.buy_orders.keys())
            
            orders = []
            
            # Entry Logic
            if z < -self.entry_z:
                # Buy
                qty = limit - pos
                if qty > 0:
                    # Place bid at best bid + 1 (passive-aggressive)
                    orders.append(Order(p, min(best_bid + 1, best_ask - 1), qty))
            
            elif z > self.entry_z:
                # Sell
                qty = -limit - pos
                if qty < 0:
                    # Place ask at best ask - 1 (passive-aggressive)
                    orders.append(Order(p, max(best_ask - 1, best_bid + 1), qty))
            
            # Exit Logic
            elif abs(z) < self.exit_z:
                if pos > 0:
                    # Close buy position (Sell)
                    orders.append(Order(p, max(best_ask - 1, best_bid + 1), -pos))
                elif pos < 0:
                    # Close sell position (Buy)
                    orders.append(Order(p, min(best_bid + 1, best_ask - 1), -pos))
            
            if orders:
                result[p] = orders

        # Save state
        data_out = json.dumps({"ema": ema_state, "m2": m2_state, "count": count})
        
        return result, conversions, data_out

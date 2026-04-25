import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import math
import itertools
from copy import deepcopy
from typing import List, Dict

from prosperity4bt.data import has_day_data
from prosperity4bt.file_reader import PackageResourcesReader, FileSystemReader
from prosperity4bt.runner import run_backtest
from prosperity4bt.models import TradeMatchingMode
from prosperity4bt.metrics import risk_metrics_full_period
from prosperity4bt import datamodel
import sys
sys.modules["datamodel"] = datamodel
from datamodel import Order

# We'll dynamically create the Trader class in the loop
def get_trader_class(params):
    class Trader:
        def run(self, state):
            result = {}
            conversions = 0
            try:
                memory = json.loads(state.traderData) if state.traderData else {}
            except Exception:
                memory = {}

            # Detect day change
            prev_ts = memory.get("prev_ts", -1)
            day = memory.get("current_day", 0)
            if state.timestamp < prev_ts:
                day += 1
            memory["prev_ts"] = state.timestamp
            memory["current_day"] = day

            # Calculate TTE in ticks
            # Total days = 5. Day 0 means 5 days left.
            # 1 day = 10,000 ticks.
            ticks_left = (5 - day) * 10000 - (state.timestamp // 100)
            ticks_left = max(1, ticks_left)
            
            # --- 1. Compute Velvetfruit Deep Mid & Volatility ---
            v_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
            v_mid = 0.0
            vol_est = 0.0
            
            if v_depth and v_depth.buy_orders and v_depth.sell_orders:
                best_bid = max(v_depth.buy_orders.keys())
                best_ask = min(v_depth.sell_orders.keys())
                v_mid = (best_bid + best_ask) / 2.0
                
                prev_mid = memory.get("VELV_prev_mid", v_mid)
                var_est = memory.get("VELV_var_est", params['initial_var'])
                
                diff = v_mid - prev_mid
                var_est = params['var_alpha'] * (diff ** 2) + (1 - params['var_alpha']) * var_est
                
                memory["VELV_prev_mid"] = v_mid
                memory["VELV_var_est"] = var_est
                vol_est = math.sqrt(var_est)
                if vol_est < 0.001: vol_est = 0.001

            # Normal CDF and PDF
            def norm_cdf(x):
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
            def norm_pdf(x):
                return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

            # Option Pricing (Bachelier Model)
            # C = (S-K)N(d) + sigma*sqrt(T)*n(d)
            # d = (S-K) / (sigma*sqrt(T))
            total_delta = 0.0
            vouchers = [f"VEV_{k}" for k in range(4000, 6501, 100)] if "VEV_4000" in state.order_depths else []
            
            voucher_fairs = {}
            for v in vouchers:
                strike = float(v.split("_")[1])
                sigma_t = vol_est * params['vol_multiplier'] * math.sqrt(ticks_left)
                d = (v_mid - strike) / sigma_t if sigma_t > 0 else 0
                
                fair_price = (v_mid - strike) * norm_cdf(d) + sigma_t * norm_pdf(d)
                delta = norm_cdf(d)
                
                voucher_fairs[v] = fair_price
                pos = state.position.get(v, 0)
                total_delta += pos * delta

            # --- 2. Trade Velvetfruit (with Hedging) ---
            if "VELVETFRUIT_EXTRACT" in state.order_depths:
                depth = state.order_depths["VELVETFRUIT_EXTRACT"]
                orders = []
                limit = 200
                pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
                
                # Hedged position
                eff_pos = pos + total_delta * params['hedge_ratio']
                
                best_ask = min(depth.sell_orders.keys())
                best_bid = max(depth.buy_orders.keys())
                
                fast_fair = memory.get("VELV_fast_fair", v_mid)
                slow_fair = memory.get("VELV_slow_fair", v_mid)
                
                a_f = params['v_alpha_fast']
                a_s = params['v_alpha_slow']
                
                fast_fair = a_f * v_mid + (1 - a_f) * fast_fair
                slow_fair = a_s * v_mid + (1 - a_s) * slow_fair
                memory["VELV_fast_fair"] = fast_fair
                memory["VELV_slow_fair"] = slow_fair
                
                fair_value = params['v_mix'] * fast_fair + (1 - params['v_mix']) * slow_fair
                
                inv_skew = eff_pos / params['v_skew_div']
                adj_fair = fair_value - inv_skew
                
                buy_cap = limit - pos
                sell_cap = -limit - pos
                
                taker_thr = params['v_taker_thr']
                max_taker = params['v_max_taker']
                
                # Taker
                for prc in sorted(depth.sell_orders.keys()):
                    if prc < adj_fair - taker_thr and buy_cap > 0:
                        qty = min(buy_cap, abs(depth.sell_orders[prc]), max_taker)
                        if qty > 0:
                            orders.append(Order("VELVETFRUIT_EXTRACT", prc, qty))
                            buy_cap -= qty
                            
                for prc in sorted(depth.buy_orders.keys(), reverse=True):
                    if prc > adj_fair + taker_thr and sell_cap < 0:
                        qty = max(sell_cap, -abs(depth.buy_orders[prc]), -max_taker)
                        if qty < 0:
                            orders.append(Order("VELVETFRUIT_EXTRACT", prc, qty))
                            sell_cap -= abs(qty)
                            
                # Maker
                my_bid = int(math.floor(adj_fair - params['v_maker_spread']))
                my_ask = int(math.ceil(adj_fair + params['v_maker_spread']))
                
                if buy_cap > 0: orders.append(Order("VELVETFRUIT_EXTRACT", my_bid, min(buy_cap, params['v_maker_qty'])))
                if sell_cap < 0: orders.append(Order("VELVETFRUIT_EXTRACT", my_ask, max(sell_cap, -params['v_maker_qty'])))
                
                result["VELVETFRUIT_EXTRACT"] = orders

            # --- 3. Trade Vouchers ---
            for v in vouchers:
                if v in state.order_depths:
                    depth = state.order_depths[v]
                    orders = []
                    limit = 300
                    pos = state.position.get(v, 0)
                    
                    fair_value = voucher_fairs[v]
                    inv_skew = pos / params['opt_skew_div']
                    adj_fair = fair_value - inv_skew
                    
                    buy_cap = limit - pos
                    sell_cap = -limit - pos
                    
                    taker_thr = params['opt_taker_thr']
                    max_taker = params['opt_max_taker']
                    
                    for prc in sorted(depth.sell_orders.keys()):
                        if prc < adj_fair - taker_thr and buy_cap > 0:
                            qty = min(buy_cap, abs(depth.sell_orders[prc]), max_taker)
                            if qty > 0:
                                orders.append(Order(v, prc, qty))
                                buy_cap -= qty
                                
                    for prc in sorted(depth.buy_orders.keys(), reverse=True):
                        if prc > adj_fair + taker_thr and sell_cap < 0:
                            qty = max(sell_cap, -abs(depth.buy_orders[prc]), -max_taker)
                            if qty < 0:
                                orders.append(Order(v, prc, qty))
                                sell_cap -= abs(qty)
                                
                    my_bid = int(math.floor(adj_fair - params['opt_maker_spread']))
                    my_ask = int(math.ceil(adj_fair + params['opt_maker_spread']))
                    
                    my_bid = min(my_bid, int(math.floor(fair_value - 0.5)))
                    my_ask = max(my_ask, int(math.ceil(fair_value + 0.5)))

                    # Avoid placing options bids below 0
                    if my_bid < 0:
                        my_bid = 0
                    
                    if buy_cap > 0: orders.append(Order(v, my_bid, min(buy_cap, params['opt_maker_qty'])))
                    if sell_cap < 0: orders.append(Order(v, my_ask, max(sell_cap, -params['opt_maker_qty'])))
                    
                    result[v] = orders

            # --- 4. Trade Hydrogel (Vanilla) ---
            if "HYDROGEL_PACK" in state.order_depths:
                depth = state.order_depths["HYDROGEL_PACK"]
                orders = []
                best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
                best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
                mid = (best_bid + best_ask) / 2.0
                
                fast_fair = memory.get("HYD_fast_fair", mid)
                slow_fair = memory.get("HYD_slow_fair", mid)
                
                a_f = 0.35
                a_s = 0.01
                fast_fair = a_f * mid + (1 - a_f) * fast_fair
                slow_fair = a_s * mid + (1 - a_s) * slow_fair
                memory["HYD_fast_fair"] = fast_fair
                memory["HYD_slow_fair"] = slow_fair
                
                fair = 0.7 * fast_fair + 0.3 * slow_fair
                
                pos = state.position.get("HYDROGEL_PACK", 0)
                inv_skew = pos / 50.0
                adj_fair = fair - inv_skew
                
                buy_cap = 200 - pos
                sell_cap = -200 - pos
                
                for prc in sorted(depth.sell_orders.keys()):
                    if prc < adj_fair - 2.0 and buy_cap > 0:
                        qty = min(buy_cap, abs(depth.sell_orders[prc]), 50)
                        if qty > 0:
                            orders.append(Order("HYDROGEL_PACK", prc, qty))
                            buy_cap -= qty
                
                for prc in sorted(depth.buy_orders.keys(), reverse=True):
                    if prc > adj_fair + 2.0 and sell_cap < 0:
                        qty = max(sell_cap, -abs(depth.buy_orders[prc]), -50)
                        if qty < 0:
                            orders.append(Order("HYDROGEL_PACK", prc, qty))
                            sell_cap -= abs(qty)
                            
                my_bid = int(math.floor(adj_fair - 1))
                my_ask = int(math.ceil(adj_fair + 1))
                if buy_cap > 0: orders.append(Order("HYDROGEL_PACK", my_bid, min(buy_cap, 50)))
                if sell_cap < 0: orders.append(Order("HYDROGEL_PACK", my_ask, max(sell_cap, -50)))
                
                result["HYDROGEL_PACK"] = orders

            return result, conversions, json.dumps(memory)
    return Trader

def main():
    print("Starting optimization...")
    file_reader = FileSystemReader(Path("data"))
    
    # Define search space
    search_space = {
        'initial_var': [0.0, 0.5, 2.0],
        'var_alpha': [0.01, 0.05, 0.1],
        'vol_multiplier': [1.0, 1.2, 1.5, 0.8],
        'hedge_ratio': [0.0, 0.5, 1.0],
        'v_alpha_fast': [0.35, 0.55],
        'v_alpha_slow': [0.01, 0.05],
        'v_mix': [0.7],
        'v_skew_div': [50.0, 25.0],
        'v_taker_thr': [1.5, 2.0],
        'v_max_taker': [25, 50],
        'v_maker_spread': [1.0, 1.5],
        'v_maker_qty': [50, 100],
        'opt_skew_div': [100.0, 50.0],
        'opt_taker_thr': [1.0, 2.0],
        'opt_max_taker': [50, 100],
        'opt_maker_spread': [0.5, 1.0, 1.5],
        'opt_maker_qty': [100]
    }
    
    keys = list(search_space.keys())
    combinations = list(itertools.product(*(search_space[k] for k in keys)))
    
    import random
    random.seed(42)
    random.shuffle(combinations) # Shuffle to randomly sample
    
    best_pnl = -float('inf')
    best_params = None
    
    # Run 100 iterations max
    iterations = min(100, len(combinations))
    print(f"Total iterations to run: {iterations}")
    
    for i in range(iterations):
        comb = combinations[i]
        params = dict(zip(keys, comb))
        
        Trader = get_trader_class(params)
        
        # Test on day 0, day 1, day 2
        total_pnl = 0
        all_results = []
        for day in [0, 1, 2]:
            if not has_day_data(file_reader, 3, day):
                continue
                
            from datamodel import Order
            sys.modules['datamodel'].Order = Order # ensure it's in datamodel
            
            res = run_backtest(
                Trader(),
                file_reader,
                3, # round
                day,
                False, # print_output
                TradeMatchingMode.all,
                True, # no_names
                False, # show_progress_bar
            )
            all_results.append(res)
            
        metrics = risk_metrics_full_period(all_results)
        pnl = metrics.final_pnl
        
        print(f"Iter {i+1}/{iterations} | PnL: {pnl:,.0f} | Params: {params}")
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = params
            print(f"--> NEW BEST! PnL: {best_pnl:,.0f}")
            with open("best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)
                
    print(f"Optimization complete. Best PnL: {best_pnl:,.0f}")

if __name__ == "__main__":
    main()

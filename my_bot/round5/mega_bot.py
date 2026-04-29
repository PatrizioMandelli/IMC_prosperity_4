import json
import math
import collections
import numpy as np
from datamodel import Order, TradingState, OrderDepth, Symbol
from typing import List, Dict, Any
from collections import deque

class Trader:
    def __init__(self):
        # ─── GALAXY CONFIG ───────────────────────────────────────────────────
        self.galaxy_products = [
            'GALAXY_SOUNDS_BLACK_HOLES',
            'GALAXY_SOUNDS_DARK_MATTER',
            'GALAXY_SOUNDS_PLANETARY_RINGS',
            'GALAXY_SOUNDS_SOLAR_FLAMES',
            'GALAXY_SOUNDS_SOLAR_WINDS'
        ]
        self.galaxy_configs = {
            'GALAXY_SOUNDS_BLACK_HOLES': ('PEBBLES_S', 20500, -1, 450),
            'GALAXY_SOUNDS_DARK_MATTER': ('UV_VISOR_YELLOW', 6150, 0.4, 200),
            'GALAXY_SOUNDS_PLANETARY_RINGS': ('GALAXY_SOUNDS_DARK_MATTER', 8000, 0.3, 300),
            'GALAXY_SOUNDS_SOLAR_WINDS': ('PANEL_1X4', 1550, -0.5, 300),
            'GALAXY_SOUNDS_SOLAR_FLAMES': ('GALAXY_SOUNDS_SOLAR_WINDS', 14000, -0.3, 420)
        }
        self.galaxy_pos_limit = 10

        # ─── TRANSLATOR CONFIG ──────────────────────────────────────────────
        self.translator_products = [
            "TRANSLATOR_ASTRO_BLACK",
            "TRANSLATOR_ECLIPSE_CHARCOAL",
            "TRANSLATOR_GRAPHITE_MIST",
            "TRANSLATOR_SPACE_GRAY",
            "TRANSLATOR_VOID_BLUE"
        ]
        self.translator_limits = {p: 10 for p in self.translator_products}
        self.translator_alpha = 0.01
        self.translator_entry_z = 1.5
        self.translator_exit_z = 0.2

        # ─── CHIP CONFIG ────────────────────────────────────────────────────
        self.chip_limit = 10
        self.chip_hard_short = {"MICROCHIP_OVAL"}
        self.chip_configs = {
            "MICROCHIP_SQUARE": (+1, 80, 15, 5, 4.5, 40),
            "MICROCHIP_CIRCLE": (+1, 100, 10, 10, 8.0, None),
            "MICROCHIP_TRIANGLE": (-1, 100, 10, 10, 5.0, None),
            "MICROCHIP_RECTANGLE": (-1, 100, 10, 10, 5.0, 50),
        }
        all_chip_products = self.chip_hard_short | set(self.chip_configs.keys())
        max_chip_w = max(cfg[1] for cfg in self.chip_configs.values())
        self.chip_prices: Dict[str, deque] = {
            p: deque(maxlen=max_chip_w) for p in all_chip_products
        }

        # ─── PEBBLES CONFIG ─────────────────────────────────────────────────
        self.pebbles_limit = 10
        self.pebbles_window = 200
        self.pebbles_z_entry = 1.5
        self.pebbles_z_exit = 0
        self.pebbles_trend_w = 50
        self.pebbles_leader = "PEBBLES_XL"
        self.pebbles_basket = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
        self.pebbles_all = self.pebbles_basket + [self.pebbles_leader]
        self.pebbles_mm_only = {"PEBBLES_S", "PEBBLES_L"}
        self.pebbles_trend_filter = {"PEBBLES_M", "PEBBLES_XS"}
        self.pebbles_prices: Dict[str, deque] = {p: deque(maxlen=max(self.pebbles_window, self.pebbles_trend_w)) for p in self.pebbles_all}
        self.pebbles_spread_hist: deque = deque(maxlen=self.pebbles_window)

        # ─── ROBO CONFIG ────────────────────────────────────────────────────
        self.robo_models = {
            'ROBOT_VACUUMING': (8883.57, 0.2096, -0.1053, 227.85),
            'ROBOT_MOPPING':   (10467.18, -0.2101, 0.1729, 480.70),
            'ROBOT_DISHES':    (12052.99, -0.2806, 0.0192, 310.83),
            'ROBOT_LAUNDRY':   (6119.02, 0.3793, 0.0442, 306.61),
            'ROBOT_IRONING':   (6538.64, 0.3789, -0.0689, 352.55)
        }
        self.robo_pos_limit = 7
        self.robo_spread_margins = {p: 2 for p in self.robo_models}

        # ─── SLEEP POD CONFIG ───────────────────────────────────────────────
        self.sleep_limit = 10
        self.sleep_skew_thresh = 9
        self.sleep_trend_window = 50
        self.sleep_products = [
            "SLEEP_POD_COTTON", "SLEEP_POD_NYLON", "SLEEP_POD_POLYESTER",
            "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_SUEDE"
        ]
        self.sleep_trend_filter = {"SLEEP_POD_LAMB_WOOL", "SLEEP_POD_COTTON", "SLEEP_POD_SUEDE"}
        self.sleep_price_history: Dict[str, deque] = {p: deque(maxlen=self.sleep_trend_window) for p in self.sleep_products}

        # ─── SNACK CONFIG ───────────────────────────────────────────────────
        self.snack_limit = 10
        self.snack_products = [
            'SNACKPACK_CHOCOLATE', 'SNACKPACK_PISTACHIO', 'SNACKPACK_RASPBERRY',
            'SNACKPACK_STRAWBERRY', 'SNACKPACK_VANILLA'
        ]
        self.snack_avg_prices = {
            'SNACKPACK_CHOCOLATE': 10000.0, 'SNACKPACK_PISTACHIO': 9500.0,
            'SNACKPACK_RASPBERRY': 10000.0, 'SNACKPACK_STRAWBERRY': 10500.0,
            'SNACKPACK_VANILLA': 10000.0
        }
        snack_total_avg = sum(self.snack_avg_prices.values())
        self.snack_weights = {p: self.snack_avg_prices[p] / snack_total_avg for p in self.snack_products}
        self.snack_alpha = 0.05
        self.snack_edge = 1.0
        self.snack_risk_factor = 3.0

        # ─── VR CONFIG ──────────────────────────────────────────────────────
        self.vr_limit = 10
        self.vr_alpha = 0.05
        self.vr_products = ["UV_VISOR_MAGENTA", "UV_VISOR_ORANGE", "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_RED"]

        # ─── SNACK PACK TEST CONFIG ─────────────────────────────────────────
        self.snack_test_baskets = {
            'SNACK_DUO': {
                'target': 'SNACKPACK_CHOCOLATE',
                'others': ['SNACKPACK_VANILLA'],
                'weights': {'SNACKPACK_VANILLA': -1.0411},
                'lots': {'SNACKPACK_CHOCOLATE': 10, 'SNACKPACK_VANILLA': 10},
                'z_entry': 2.0, 'z_exit': 0.5, 'window_size': 100
            },
            'SNACK_TRIO': {
                'target': 'SNACKPACK_PISTACHIO',
                'others': ['SNACKPACK_RASPBERRY', 'SNACKPACK_STRAWBERRY'],
                'weights': {'SNACKPACK_RASPBERRY': -0.9056, 'SNACKPACK_STRAWBERRY': -0.4025},
                'lots': {'SNACKPACK_PISTACHIO': 10, 'SNACKPACK_RASPBERRY': 9, 'SNACKPACK_STRAWBERRY': 4},
                'z_entry': 2.0, 'z_exit': 0.5, 'window_size': 100
            }
        }

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def _passive_order_chip(self, symbol: str, depth: OrderDepth, pos: int, target: int) -> List[Order]:
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        orders: List[Order] = []
        diff = target - pos
        if diff > 0:
            orders.append(Order(symbol, best_bid + 1, diff))
        elif diff < 0:
            orders.append(Order(symbol, best_ask - 1, diff))
        return orders

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        # ─── LOAD MASTER DATA ───────────────────────────────────────────────
        master_data = {}
        if state.traderData:
            try:
                master_data = json.loads(state.traderData)
            except:
                master_data = {}

        # Pre-calculate all mids
        mids = {}
        for p in state.order_depths:
            mids[p] = self.get_mid_price(state.order_depths[p])

        # ─── GALAXY LOGIC ───────────────────────────────────────────────────
        for target in self.galaxy_products:
            if target not in self.galaxy_configs or target not in state.order_depths:
                continue
            oracle, alpha, beta, std = self.galaxy_configs[target]
            oracle_mid = mids.get(oracle)
            if oracle_mid is None: continue
            fair_target = alpha + beta * oracle_mid
            current_pos = state.position.get(target, 0)
            target_depth = state.order_depths[target]
            buy_orders_sorted = collections.OrderedDict(sorted(target_depth.buy_orders.items(), reverse=True))
            sell_orders_sorted = collections.OrderedDict(sorted(target_depth.sell_orders.items()))
            spread = mids[target] - fair_target
            z_score = spread / std
            orders = []
            target_pos = 0
            if z_score < -1.0: target_pos = self.galaxy_pos_limit
            elif z_score > 1.0: target_pos = -self.galaxy_pos_limit
            elif abs(z_score) < 0.2: target_pos = 0
            else: target_pos = current_pos
            diff = target_pos - current_pos
            if diff > 0:
                for price, vol in sell_orders_sorted.items():
                    if price <= fair_target:
                        buy_qty = min(diff, -vol)
                        orders.append(Order(target, price, buy_qty))
                        diff -= buy_qty
                    if diff <= 0: break
                if diff > 0:
                    best_bid = max(target_depth.buy_orders.keys())
                    orders.append(Order(target, best_bid + 1, diff))
            elif diff < 0:
                diff = -diff
                for price, vol in buy_orders_sorted.items():
                    if price >= fair_target:
                        sell_qty = min(diff, vol)
                        orders.append(Order(target, price, -sell_qty))
                        diff -= sell_qty
                    if diff <= 0: break
                if diff > 0:
                    best_ask = min(target_depth.sell_orders.keys())
                    orders.append(Order(target, best_ask - 1, -diff))
            if orders: result[target] = orders

        # ─── TRANSLATOR LOGIC ───────────────────────────────────────────────
        trans_data = master_data.get("translator", {})
        ema_state = trans_data.get("ema", {})
        m2_state = trans_data.get("m2", {})
        count = trans_data.get("count", 0)
        valid_trans_mids = [mids[p] for p in self.translator_products if p in mids]
        if len(valid_trans_mids) >= 3:
            basket_mean = sum(valid_trans_mids) / len(valid_trans_mids)
            count += 1
            signals = {}
            stds = {}
            for p in self.translator_products:
                if p in mids:
                    rel_price = mids[p] - basket_mean
                    curr_ema = ema_state.get(p, rel_price)
                    new_ema = self.translator_alpha * rel_price + (1 - self.translator_alpha) * curr_ema
                    ema_state[p] = new_ema
                    diff = rel_price - new_ema
                    curr_m2 = m2_state.get(p, 70.0**2)
                    new_m2 = self.translator_alpha * (diff**2) + (1 - self.translator_alpha) * curr_m2
                    m2_state[p] = new_m2
                    stds[p] = math.sqrt(new_m2)
                    signals[p] = diff
            for p in self.translator_products:
                if p not in signals or p not in stds: continue
                sig, std = signals[p], stds[p]
                z = sig / std if std > 0 else 0
                pos = state.position.get(p, 0)
                limit = self.translator_limits[p]
                depth = state.order_depths[p]
                best_ask, best_bid = min(depth.sell_orders.keys()), max(depth.buy_orders.keys())
                orders = []
                if z < -self.translator_entry_z:
                    qty = limit - pos
                    if qty > 0: orders.append(Order(p, min(best_bid + 1, best_ask - 1), qty))
                elif z > self.translator_entry_z:
                    qty = -limit - pos
                    if qty < 0: orders.append(Order(p, max(best_ask - 1, best_bid + 1), qty))
                elif abs(z) < self.translator_exit_z:
                    if pos > 0: orders.append(Order(p, max(best_ask - 1, best_bid + 1), -pos))
                    elif pos < 0: orders.append(Order(p, min(best_bid + 1, best_ask - 1), -pos))
                if orders: result[p] = orders
            master_data["translator"] = {"ema": ema_state, "m2": m2_state, "count": count}

        # ─── CHIP LOGIC ─────────────────────────────────────────────────────
        for p in self.chip_prices:
            if mids.get(p): self.chip_prices[p].append(mids[p])
        for p in self.chip_hard_short:
            depth = state.order_depths.get(p)
            if depth:
                orders = self._passive_order_chip(p, depth, state.position.get(p, 0), -self.chip_limit)
                if orders: result[p] = orders
        for p, (direction, trend_w, s_start, s_end, threshold, short_w) in self.chip_configs.items():
            depth = state.order_depths.get(p)
            if not depth or p not in self.chip_prices or len(self.chip_prices[p]) < trend_w: continue
            hist = list(self.chip_prices[p])
            long_trend = np.mean(hist[-s_end:]) - np.mean(hist[:s_start])
            long_confirms = (direction > 0 and long_trend > threshold) or (direction < 0 and long_trend < -threshold)
            target = 0
            if long_confirms:
                if short_w is not None:
                    short_hist = hist[-short_w:]
                    short_confirms = (direction > 0 and (np.mean(short_hist[-5:]) - np.mean(short_hist[:5])) > 0) or \
                                     (direction < 0 and (np.mean(short_hist[-5:]) - np.mean(short_hist[:5])) < 0)
                    target = self.chip_limit * direction if short_confirms else 0
                else: target = self.chip_limit * direction
            if state.position.get(p, 0) != target:
                orders = self._passive_order_chip(p, depth, state.position.get(p, 0), target)
                if orders: result[p] = orders

        # ─── PEBBLES LOGIC ───────────────────────────────────────────────────
        for p in self.pebbles_all:
            if mids.get(p): self.pebbles_prices[p].append(mids[p])
        z_pebbles = 0.0
        if all(p in mids for p in self.pebbles_all):
            basket_avg = np.mean([mids[p] for p in self.pebbles_basket])
            spread = mids[self.pebbles_leader] - basket_avg
            self.pebbles_spread_hist.append(spread)
            if len(self.pebbles_spread_hist) >= self.pebbles_window:
                arr = np.array(self.pebbles_spread_hist)
                if arr.std() > 0: z_pebbles = (spread - arr.mean()) / arr.std()
        for p in self.pebbles_all:
            depth = state.order_depths.get(p)
            if not depth: continue
            best_bid, best_ask = max(depth.buy_orders), min(depth.sell_orders)
            pos, orders = state.position.get(p, 0), []
            if p == self.pebbles_leader:
                if z_pebbles > self.pebbles_z_entry and pos > -self.pebbles_limit:
                    qty = self.pebbles_limit + pos
                    if qty > 0: orders.append(Order(p, best_bid, -qty))
                elif z_pebbles < -self.pebbles_z_entry and pos < self.pebbles_limit:
                    qty = self.pebbles_limit - pos
                    if qty > 0: orders.append(Order(p, best_ask, qty))
                elif abs(z_pebbles) < self.pebbles_z_exit and pos != 0:
                    orders.append(Order(p, best_bid if pos > 0 else best_ask, -pos))
                else:
                    buy_qty, sell_qty = max(0, self.pebbles_limit - pos), max(0, self.pebbles_limit + pos)
                    if buy_qty > 0: orders.append(Order(p, best_bid + 1, int(buy_qty)))
                    if sell_qty > 0: orders.append(Order(p, best_ask - 1, -int(sell_qty)))
            elif p in self.pebbles_mm_only:
                buy_qty, sell_qty = max(0, self.pebbles_limit - pos), max(0, self.pebbles_limit + pos)
                if buy_qty > 0: orders.append(Order(p, best_bid + 1, int(buy_qty)))
                if sell_qty > 0: orders.append(Order(p, best_ask - 1, -int(sell_qty)))
            elif p in self.pebbles_trend_filter:
                hist = self.pebbles_prices[p]
                buy_qty, sell_qty = max(0, self.pebbles_limit - pos), max(0, self.pebbles_limit + pos)
                if len(hist) >= self.pebbles_trend_w:
                    trend = hist[-1] - hist[-self.pebbles_trend_w]
                    if trend > 0: buy_qty, sell_qty = 0, (min(self.pebbles_limit, pos + 5) if pos > 0 else self.pebbles_limit + pos)
                    elif trend < 0: sell_qty, buy_qty = 0, (-pos if pos < 0 else buy_qty)
                if buy_qty > 0: orders.append(Order(p, best_bid + 1, int(buy_qty)))
                if sell_qty > 0: orders.append(Order(p, best_ask - 1, -int(sell_qty)))
            if orders: result[p] = orders

        # ─── ROBO LOGIC ─────────────────────────────────────────────────────
        oval_mid, square_mid = mids.get('MICROCHIP_OVAL'), mids.get('MICROCHIP_SQUARE')
        if oval_mid is not None and square_mid is not None:
            for product, (alpha, b_oval, b_square, std_err) in self.robo_models.items():
                depth = state.order_depths.get(product)
                if not depth: continue
                best_bid, best_ask = max(depth.buy_orders.keys()), min(depth.sell_orders.keys())
                mid_p, current_pos = (best_bid + best_ask) / 2.0, state.position.get(product, 0)
                fair_value = alpha + b_oval * oval_mid + b_square * square_mid
                residual = mid_p - fair_value
                z_score = residual / std_err
                target_pos = max(-self.robo_pos_limit, min(self.robo_pos_limit, int(-z_score * 3)))
                orders = []
                my_bid = int(round(fair_value - self.robo_spread_margins[product] - current_pos * 0.5))
                my_ask = int(round(fair_value + self.robo_spread_margins[product] - current_pos * 0.5))
                buy_vol = target_pos - current_pos if target_pos > current_pos else 0
                sell_vol = target_pos - current_pos if target_pos < current_pos else 0
                if z_score < -1.5:
                    if buy_vol > 0: orders.append(Order(product, best_ask, buy_vol))
                elif z_score > 1.5:
                    if sell_vol < 0: orders.append(Order(product, best_bid, sell_vol))
                else:
                    bv_lim, sv_lim = self.robo_pos_limit - current_pos, -self.robo_pos_limit - current_pos
                    if bv_lim > 0: orders.append(Order(product, min(my_bid, best_bid + 1), bv_lim))
                    if sv_lim < 0: orders.append(Order(product, max(my_ask, best_ask - 1), sv_lim))
                if orders: result[product] = orders

        # ─── SLEEP LOGIC ────────────────────────────────────────────────────
        for symbol in self.sleep_products:
            depth = state.order_depths.get(symbol)
            if not depth: continue
            best_bid, best_ask = max(depth.buy_orders), min(depth.sell_orders)
            mid, pos = (best_bid + best_ask) / 2, state.position.get(symbol, 0)
            self.sleep_price_history[symbol].append(mid)
            buy_price, sell_price = best_bid + 1, best_ask - 1
            if pos >= self.sleep_skew_thresh: sell_price -= 1
            elif pos <= -self.sleep_skew_thresh: buy_price += 1
            if buy_price >= sell_price: buy_price, sell_price = best_bid, best_ask
            buy_qty, sell_qty = max(0, self.sleep_limit - pos), max(0, self.sleep_limit + pos)
            if symbol in self.sleep_trend_filter:
                hist = self.sleep_price_history[symbol]
                if len(hist) >= self.sleep_trend_window:
                    trend = hist[-1] - hist[0]
                    if trend > 0: buy_qty, sell_qty = 0, (pos if pos > 0 else 0)
                    elif trend < 0: sell_qty, buy_qty = 0, (-pos if pos < 0 else 0)
            orders = []
            if buy_qty > 0: orders.append(Order(symbol, buy_price, int(buy_qty)))
            if sell_qty > 0: orders.append(Order(symbol, sell_price, -int(sell_qty)))
            if orders: result[symbol] = orders

        # ─── SNACK LOGIC ────────────────────────────────────────────────────
        snack_data = master_data.get("snack", {"ema_weights": self.snack_weights})
        valid_snack_mids = {p: mids[p] for p in self.snack_products if p in mids}
        if valid_snack_mids:
            cluster_sum = sum(valid_snack_mids.values())
            for p, m in valid_snack_mids.items():
                rel_w = m / cluster_sum
                snack_data["ema_weights"][p] = self.snack_alpha * rel_w + (1 - self.snack_alpha) * snack_data["ema_weights"].get(p, rel_w)
                curr_pos, target_pos = state.position.get(p, 0), max(min(-(rel_w - snack_data["ema_weights"][p]) * 5000.0, self.snack_limit), -self.snack_limit)
                fair_val = m + (target_pos - curr_pos) * self.snack_risk_factor
                depth = state.order_depths[p]
                best_bid, best_ask = max(depth.buy_orders.keys()), min(depth.sell_orders.keys())
                bid_p, ask_p = min(int(math.floor(fair_val - self.snack_edge)), best_bid + 1), max(int(math.ceil(fair_val + self.snack_edge)), best_ask - 1)
                orders = result.get(p, [])
                if curr_pos < self.snack_limit: orders.append(Order(p, bid_p, self.snack_limit - curr_pos))
                if curr_pos > -self.snack_limit: orders.append(Order(p, ask_p, -self.snack_limit - curr_pos))
                result[p] = orders
            master_data["snack"] = snack_data

        # ─── VR LOGIC ───────────────────────────────────────────────────────
        vr_data = master_data.get("vr", {"ema_diff": {}, "var_diff": {}})
        if all(p in mids for p in self.vr_products):
            signals_vr = {p: 0.0 for p in self.vr_products}
            for i in range(len(self.vr_products)):
                for j in range(i + 1, len(self.vr_products)):
                    p1, p2 = self.vr_products[i], self.vr_products[j]
                    pk, diff = f"{p1}_{p2}", mids[p1] - mids[p2]
                    vr_data["ema_diff"][pk] = self.vr_alpha * diff + (1 - self.vr_alpha) * vr_data["ema_diff"].get(pk, diff)
                    dev = diff - vr_data["ema_diff"][pk]
                    vr_data["var_diff"][pk] = self.vr_alpha * (dev ** 2) + (1 - self.vr_alpha) * vr_data["var_diff"].get(pk, 0.0)
                    std_v = math.sqrt(vr_data["var_diff"][pk]) if vr_data["var_diff"][pk] > 0 else 1.0
                    z_v = max(min(dev / std_v, 3.0), -3.0)
                    signals_vr[p1] -= z_v
                    signals_vr[p2] += z_v
            for p in self.vr_products:
                curr_pos = state.position.get(p, 0)
                target_pos = max(min(signals_vr[p] * (self.vr_limit / 6.0), self.vr_limit), -self.vr_limit)
                fair_val = mids[p] + (target_pos - curr_pos) * 1.5
                depth = state.order_depths[p]
                best_bid, best_ask = max(depth.buy_orders.keys()), min(depth.sell_orders.keys())
                bid_p, ask_p = min(int(math.floor(fair_val - 1.5)), best_bid + 1), max(int(math.ceil(fair_val + 1.5)), best_ask - 1)
                orders = result.get(p, [])
                if curr_pos < self.vr_limit: orders.append(Order(p, bid_p, self.vr_limit - curr_pos))
                if curr_pos > -self.vr_limit: orders.append(Order(p, ask_p, -self.vr_limit - curr_pos))
                result[p] = orders
            master_data["vr"] = vr_data

        # ─── SNACK PACK TEST LOGIC ──────────────────────────────────────────
        s_test_data = master_data.get("snack_test", {"spread_history": {b: [] for b in self.snack_test_baskets}})
        for b_name, b_cfg in self.snack_test_baskets.items():
            target, others = b_cfg['target'], b_cfg['others']
            if target in mids and all(o in mids for o in others):
                cur_spread = mids[target] - sum(b_cfg['weights'][o] * mids[o] for o in others)
                history = s_test_data['spread_history'][b_name]
                history.append(cur_spread)
                if len(history) > b_cfg['window_size']: history.pop(0)
                if len(history) >= 20:
                    mean_s = sum(history) / len(history)
                    std_s = math.sqrt(sum((x - mean_s) ** 2 for x in history) / len(history)) or 1.0
                    z_s = (cur_spread - mean_s) / std_s
                    pos_t, t_lot = state.position.get(target, 0), b_cfg['lots'][target]
                    if z_s > b_cfg['z_entry'] and pos_t > -t_lot:
                        result.setdefault(target, []).append(Order(target, max(state.order_depths[target].buy_orders.keys()), -t_lot - pos_t))
                        for o in others: result.setdefault(o, []).append(Order(o, min(state.order_depths[o].sell_orders.keys()), b_cfg['lots'][o] - state.position.get(o, 0)))
                    elif z_s < -b_cfg['z_entry'] and pos_t < t_lot:
                        result.setdefault(target, []).append(Order(target, min(state.order_depths[target].sell_orders.keys()), t_lot - pos_t))
                        for o in others: result.setdefault(o, []).append(Order(o, max(state.order_depths[o].buy_orders.keys()), -b_cfg['lots'][o] - state.position.get(o, 0)))
                    elif abs(z_s) < b_cfg['z_exit'] and pos_t != 0:
                        if pos_t > 0:
                            result.setdefault(target, []).append(Order(target, max(state.order_depths[target].buy_orders.keys()), -pos_t))
                            for o in others: result.setdefault(o, []).append(Order(o, min(state.order_depths[o].sell_orders.keys()), -state.position.get(o, 0)))
                        else:
                            result.setdefault(target, []).append(Order(target, min(state.order_depths[target].sell_orders.keys()), -pos_t))
                            for o in others: result.setdefault(o, []).append(Order(o, max(state.order_depths[o].buy_orders.keys()), -state.position.get(o, 0)))
            s_test_data['spread_history'][b_name] = history
        master_data["snack_test"] = s_test_data

        return result, conversions, json.dumps(master_data)

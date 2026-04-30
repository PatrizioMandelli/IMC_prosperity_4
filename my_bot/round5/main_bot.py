import json
import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from datamodel import Order, OrderDepth, TradingState


# ════════════════════════════════════════════════════════════════════════════
# CHIP — ema_chip_trader
# ════════════════════════════════════════════════════════════════════════════
class ChipTrader:
    LIMIT = 10

    HARD_SHORT = {"MICROCHIP_OVAL"}

    CONFIGS = {
        "MICROCHIP_SQUARE": (+1, 80, 15, 5, 4.5, 40),
        "MICROCHIP_CIRCLE": (+1, 100, 10, 10, 8.0, None),
        "MICROCHIP_TRIANGLE": (-1, 100, 10, 10, 5.0, None),
        "MICROCHIP_RECTANGLE": (-1, 100, 10, 10, 5.0, 50),
    }

    def __init__(self):
        all_products = self.HARD_SHORT | set(self.CONFIGS.keys())
        max_w = max(cfg[1] for cfg in self.CONFIGS.values())
        self.prices: Dict[str, deque] = {
            p: deque(maxlen=max_w) for p in all_products
        }

    def mid(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    def _passive_order(self, symbol: str, depth: OrderDepth,
                       pos: int, target: int) -> List[Order]:
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
        result: Dict[str, List[Order]] = {}

        for p in self.prices:
            depth = state.order_depths.get(p)
            if depth:
                m = self.mid(depth)
                if m:
                    self.prices[p].append(m)

        for p in self.HARD_SHORT:
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            pos = state.position.get(p, 0)
            orders = self._passive_order(p, depth, pos, -self.LIMIT)
            if orders:
                result[p] = orders

        for p, (direction, trend_w, s_start, s_end, threshold, short_w) in self.CONFIGS.items():
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            pos = state.position.get(p, 0)
            hist = list(self.prices[p])

            if len(hist) < trend_w:
                continue

            start_val = np.mean(hist[:s_start])
            end_val = np.mean(hist[-s_end:])
            long_trend = end_val - start_val

            long_confirms = (direction > 0 and long_trend > threshold) or \
                            (direction < 0 and long_trend < -threshold)

            target = 0
            if long_confirms:
                if short_w is not None:
                    short_hist = hist[-short_w:]
                    short_start = np.mean(short_hist[:5])
                    short_end = np.mean(short_hist[-5:])
                    short_trend = short_end - short_start
                    short_confirms = (direction > 0 and short_trend > 0) or \
                                     (direction < 0 and short_trend < 0)
                    target = self.LIMIT * direction if short_confirms else 0
                else:
                    target = self.LIMIT * direction

            if pos != target:
                orders = self._passive_order(p, depth, pos, target)
                if orders:
                    result[p] = orders

        return result, 0, ""


# ════════════════════════════════════════════════════════════════════════════
# GALAXY — ema_galaxy_trader (OU mean reversion)
# ════════════════════════════════════════════════════════════════════════════
class GalaxyTrader:
    def __init__(self):
        self.products = [
            'GALAXY_SOUNDS_DARK_MATTER',
            'GALAXY_SOUNDS_SOLAR_FLAMES',
            'GALAXY_SOUNDS_SOLAR_WINDS',
        ]

        self.ou_mu = {
            'GALAXY_SOUNDS_DARK_MATTER': 10245.34,
            'GALAXY_SOUNDS_SOLAR_FLAMES': 11156.44,
            'GALAXY_SOUNDS_SOLAR_WINDS':  10475.63,
        }
        self.ou_sigma = {
            'GALAXY_SOUNDS_DARK_MATTER': 32.17,
            'GALAXY_SOUNDS_SOLAR_FLAMES': 35.45,
            'GALAXY_SOUNDS_SOLAR_WINDS':  33.41,
        }
        self.ou_half_life = {
            'GALAXY_SOUNDS_DARK_MATTER': 1438,
            'GALAXY_SOUNDS_SOLAR_FLAMES': 1732,
            'GALAXY_SOUNDS_SOLAR_WINDS':  3479,
        }

        self.windows = {p: self.ou_half_life[p] for p in self.products}

        self.emas = {p: None for p in self.products}
        self.position_limit = 10

        self.z_enter = 0.6
        self.z_full = 2.0

        self.mu_anchor_weight = 0.35

    def _fair_value(self, product, ema):
        return ema * (1.0 - self.mu_anchor_weight) + self.ou_mu[product] * self.mu_anchor_weight

    def _target_position(self, z):
        if abs(z) < self.z_enter:
            return 0
        sign = -1 if z > 0 else 1
        scale = min(1.0, (abs(z) - self.z_enter) / max(self.z_full - self.z_enter, 1e-9))
        return int(round(sign * scale * self.position_limit))

    def run(self, state: TradingState):
        result = {}

        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                for p in self.products:
                    if p in saved:
                        self.emas[p] = saved[p]
            except Exception:
                pass

        for product in self.products:
            if product not in state.order_depths:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            if not order_depth.sell_orders or not order_depth.buy_orders:
                continue

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            window = self.windows[product]
            alpha = 2.0 / (window + 1.0)
            if self.emas[product] is None:
                self.emas[product] = self.ou_mu[product]
            self.emas[product] = mid_price * alpha + self.emas[product] * (1.0 - alpha)

            fair_value = self._fair_value(product, self.emas[product])
            sigma = self.ou_sigma[product]
            z = (mid_price - fair_value) / sigma

            current_position = state.position.get(product, 0)
            target_position = self._target_position(z)
            delta = target_position - current_position

            orders: List[Order] = []

            aggressive = abs(z) >= self.z_full * 0.9

            if delta > 0:
                if aggressive:
                    qty = min(delta, -order_depth.sell_orders[best_ask])
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                else:
                    bid_price = min(best_bid + 1, math.floor(fair_value - sigma * self.z_enter))
                    if bid_price < best_ask:
                        orders.append(Order(product, bid_price, delta))
            elif delta < 0:
                if aggressive:
                    qty = min(-delta, order_depth.buy_orders[best_bid])
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                else:
                    ask_price = max(best_ask - 1, math.ceil(fair_value + sigma * self.z_enter))
                    if ask_price > best_bid:
                        orders.append(Order(product, ask_price, delta))
            else:
                if current_position == 0:
                    bid_price = min(best_bid + 1, math.floor(fair_value - sigma * 0.8))
                    ask_price = max(best_ask - 1, math.ceil(fair_value + sigma * 0.8))
                    if bid_price < best_ask:
                        orders.append(Order(product, bid_price, 2))
                    if ask_price > best_bid:
                        orders.append(Order(product, ask_price, -2))

            result[product] = orders

        traderData = json.dumps(self.emas)
        return result, 0, traderData


# ════════════════════════════════════════════════════════════════════════════
# OXYGEN — ema_oxygen_trader (pair trading)
# ════════════════════════════════════════════════════════════════════════════
class OxygenTrader:
    def __init__(self):
        self.LIMIT = 10

        self.P1_A = 'OXYGEN_SHAKE_MORNING_BREATH'
        self.P1_B = 'OXYGEN_SHAKE_EVENING_BREATH'
        self.P1_MEAN_ALPHA = 0.0005
        self.P1_VOL_ALPHA = 0.005
        self.P1_START_MEAN = 250.0
        self.P1_VOL_FLOOR = 30.0
        self.P1_ENTRY_Z = 3
        self.P1_EXIT_Z = 0

        self.P2_A = 'OXYGEN_SHAKE_GARLIC'
        self.P2_B = 'OXYGEN_SHAKE_CHOCOLATE'
        self.P2_MEAN_ALPHA = 0.0005
        self.P2_VOL_ALPHA = 0.005
        self.P2_START_MEAN = 2600
        self.P2_VOL_FLOOR = 20.0
        self.P2_ENTRY_Z = 3.0
        self.P2_EXIT_Z = 0

    def get_mid(self, depth: OrderDepth) -> Optional[float]:
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def process_leg(self, state: TradingState, sym_a: str, sym_b: str,
                    entry_z: float, exit_z: float,
                    mean_alpha: float, vol_alpha: float, starting_mean: Optional[float], vol_floor: float,
                    leg_id: str, trader_data: dict, result: dict):

        mid_a = self.get_mid(state.order_depths.get(sym_a))
        mid_b = self.get_mid(state.order_depths.get(sym_b))

        if mid_a is not None and mid_b is not None:
            spread = mid_a - mid_b

            mean_key = f"{leg_id}_mean"
            init_mean = starting_mean if starting_mean is not None else spread
            current_mean = trader_data.get(mean_key, init_mean)

            new_mean = (spread * mean_alpha) + (current_mean * (1 - mean_alpha))
            trader_data[mean_key] = new_mean

            vol_key = f"{leg_id}_vol"
            current_vol = trader_data.get(vol_key, vol_floor)
            residual = abs(spread - new_mean)
            new_vol = (residual * vol_alpha) + (current_vol * (1 - vol_alpha))
            trader_data[vol_key] = new_vol

            effective_vol = max(new_vol, vol_floor)
            z_score = (spread - new_mean) / effective_vol

            pos_a = state.position.get(sym_a, 0)
            target = pos_a

            if pos_a == 0:
                if z_score > entry_z:
                    target = -self.LIMIT
                elif z_score < -entry_z:
                    target = self.LIMIT
            else:
                if pos_a < 0 and z_score < exit_z:
                    target = 0
                elif pos_a > 0 and z_score > -exit_z:
                    target = 0

            for sym, t_pos in [(sym_a, target), (sym_b, -target)]:
                pos = state.position.get(sym, 0)
                diff = t_pos - pos
                if abs(diff) >= 1:
                    depth = state.order_depths[sym]
                    price = min(depth.sell_orders.keys()) if diff > 0 else max(depth.buy_orders.keys())
                    result.setdefault(sym, []).append(Order(sym, int(price), diff))

    def run(self, state: TradingState):
        result = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        self.process_leg(state, self.P1_A, self.P1_B, self.P1_ENTRY_Z, self.P1_EXIT_Z,
                         self.P1_MEAN_ALPHA, self.P1_VOL_ALPHA, self.P1_START_MEAN, self.P1_VOL_FLOOR,
                         "P1", trader_data, result)

        self.process_leg(state, self.P2_A, self.P2_B, self.P2_ENTRY_Z, self.P2_EXIT_Z,
                         self.P2_MEAN_ALPHA, self.P2_VOL_ALPHA, self.P2_START_MEAN, self.P2_VOL_FLOOR,
                         "P2", trader_data, result)

        return result, 0, json.dumps(trader_data)


# ════════════════════════════════════════════════════════════════════════════
# PANEL — ema_panel_trader (couple vs couple basket)
# ════════════════════════════════════════════════════════════════════════════
class PanelTrader:
    def __init__(self):
        self.LIMIT = 10

        self.COUPLE_LEFT = ['PANEL_1X2', 'PANEL_1X4']
        self.COUPLE_RIGHT = ['PANEL_2X2', 'PANEL_2X4']

        self.START_MEAN = -2350

        self.ENTRY_Z = 1
        self.EXIT_Z = 0

        self.MEAN_ALPHA = 0.0001
        self.VOL_ALPHA = 0.01
        self.VOL_FLOOR = 30.0

    def get_mid(self, symbol: str, state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(symbol)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def run(self, state: TradingState):
        result = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}
        key = "BASKET_USER"

        mids = {}
        for sym in self.COUPLE_LEFT + self.COUPLE_RIGHT:
            mid = self.get_mid(sym, state)
            if mid is None:
                return result, 0, json.dumps(trader_data)
            mids[sym] = mid

        current_spread = (mids['PANEL_1X2'] + mids['PANEL_1X4']) - \
                         (mids['PANEL_2X2'] + mids['PANEL_2X4'])

        if key not in trader_data:
            trader_data[key] = {"mean": self.START_MEAN, "m2": self.VOL_FLOOR ** 2}

        mu = trader_data[key]["mean"]
        m2 = trader_data[key]["m2"]

        mu = (1 - self.MEAN_ALPHA) * mu + self.MEAN_ALPHA * current_spread
        m2 = (1 - self.VOL_ALPHA) * m2 + self.VOL_ALPHA * ((current_spread - mu) ** 2)

        trader_data[key] = {"mean": mu, "m2": m2}
        std = max(math.sqrt(m2), self.VOL_FLOOR)
        z_score = (current_spread - mu) / std

        target_dir = 0
        current_lead_pos = state.position.get('PANEL_1X2', 0)

        if z_score > self.ENTRY_Z:
            target_dir = -1
        elif z_score < -self.ENTRY_Z:
            target_dir = 1
        elif abs(z_score) <= self.EXIT_Z:
            target_dir = 0
        else:
            if current_lead_pos > 0:
                target_dir = 1
            elif current_lead_pos < 0:
                target_dir = -1

        for sym in self.COUPLE_LEFT:
            target_pos = target_dir * self.LIMIT
            current_pos = state.position.get(sym, 0)
            diff = target_pos - current_pos

            if diff != 0:
                depth = state.order_depths[sym]
                if diff > 0:
                    price = min(depth.sell_orders.keys())
                else:
                    price = max(depth.buy_orders.keys())
                result.setdefault(sym, []).append(Order(sym, int(price), diff))

        for sym in self.COUPLE_RIGHT:
            target_pos = -target_dir * self.LIMIT
            current_pos = state.position.get(sym, 0)
            diff = target_pos - current_pos

            if diff != 0:
                depth = state.order_depths[sym]
                if diff > 0:
                    price = min(depth.sell_orders.keys())
                else:
                    price = max(depth.buy_orders.keys())
                result.setdefault(sym, []).append(Order(sym, int(price), diff))

        return result, 0, json.dumps(trader_data)


# ════════════════════════════════════════════════════════════════════════════
# PEBBLES — ema_pebbles_trader
# ════════════════════════════════════════════════════════════════════════════
class PebblesTrader:
    LIMIT = 10
    WINDOW = 200
    Z_ENTRY = 1.5
    Z_EXIT = 0
    TREND_W = 50

    LEADER = "PEBBLES_XL"
    BASKET = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
    ALL = BASKET + [LEADER]

    MM_ONLY = {"PEBBLES_S", "PEBBLES_L"}
    TREND_FILTER = {"PEBBLES_M", "PEBBLES_XS"}

    def __init__(self):
        self.prices: Dict[str, deque] = {p: deque(maxlen=max(self.WINDOW, self.TREND_W)) for p in self.ALL}
        self.spread_hist: deque = deque(maxlen=self.WINDOW)

    def mid(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        mids = {}
        for p in self.ALL:
            depth = state.order_depths.get(p)
            if depth:
                m = self.mid(depth)
                if m:
                    self.prices[p].append(m)
                    mids[p] = m

        z = 0.0
        if len(mids) == len(self.ALL):
            basket_avg = np.mean([mids[p] for p in self.BASKET])
            spread = mids[self.LEADER] - basket_avg
            self.spread_hist.append(spread)
            if len(self.spread_hist) >= self.WINDOW:
                arr = np.array(self.spread_hist)
                std = arr.std()
                if std > 0:
                    z = (spread - arr.mean()) / std

        for p in self.ALL:
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            pos = state.position.get(p, 0)
            orders: List[Order] = []

            if p == self.LEADER:
                if z > self.Z_ENTRY and pos > -self.LIMIT:
                    qty = self.LIMIT + pos
                    if qty > 0:
                        orders.append(Order(p, best_bid, -qty))
                elif z < -self.Z_ENTRY and pos < self.LIMIT:
                    qty = self.LIMIT - pos
                    if qty > 0:
                        orders.append(Order(p, best_ask, qty))
                elif abs(z) < self.Z_EXIT and pos != 0:
                    if pos > 0:
                        orders.append(Order(p, best_bid, -pos))
                    else:
                        orders.append(Order(p, best_ask, -pos))
                else:
                    buy_qty = max(0, self.LIMIT - pos)
                    sell_qty = max(0, self.LIMIT + pos)
                    if buy_qty > 0:
                        orders.append(Order(p, best_bid + 1, int(buy_qty)))
                    if sell_qty > 0:
                        orders.append(Order(p, best_ask - 1, -int(sell_qty)))

            elif p in self.MM_ONLY:
                buy_qty = max(0, self.LIMIT - pos)
                sell_qty = max(0, self.LIMIT + pos)
                if buy_qty > 0:
                    orders.append(Order(p, best_bid + 1, int(buy_qty)))
                if sell_qty > 0:
                    orders.append(Order(p, best_ask - 1, -int(sell_qty)))

            elif p in self.TREND_FILTER:
                hist = self.prices[p]
                buy_qty = max(0, self.LIMIT - pos)
                sell_qty = max(0, self.LIMIT + pos)

                if len(hist) >= self.TREND_W:
                    trend = hist[-1] - hist[-self.TREND_W]
                    if trend > 0:
                        buy_qty = 0
                        sell_qty = min(self.LIMIT, pos + 5) if pos > 0 else self.LIMIT + pos
                    elif trend < 0:
                        sell_qty = 0
                        if pos < 0:
                            buy_qty = -pos

                if buy_qty > 0:
                    orders.append(Order(p, best_bid + 1, int(buy_qty)))
                if sell_qty > 0:
                    orders.append(Order(p, best_ask - 1, -int(sell_qty)))

            if orders:
                result[p] = orders

        return result, 0, ""


# ════════════════════════════════════════════════════════════════════════════
# ROBO — ema_robo_trader (robot pairs)
# ════════════════════════════════════════════════════════════════════════════
class RoboTrader:
    POSITION_LIMIT = 10

    MODELS = {
        'ROBOT_VACUUMING': {
            'alpha': 9805.9465,
            'betas': {
                'ROBOT_MOPPING': -0.1545,
                'ROBOT_DISHES':   0.0903,
                'ROBOT_LAUNDRY':  0.1119,
                'ROBOT_IRONING': -0.1649,
                'MICROCHIP_OVAL': 0.2225,
                'MICROCHIP_SQUARE': -0.0966,
            },
            'std': 205.94,
            'half_life': 579,
        },
        'ROBOT_MOPPING': {
            'alpha': 27185.8246,
            'betas': {
                'ROBOT_VACUUMING': -0.3726,
                'ROBOT_DISHES':   -0.5293,
                'ROBOT_LAUNDRY':  -0.5493,
                'ROBOT_IRONING':  -0.5609,
                'MICROCHIP_OVAL':  0.1404,
                'MICROCHIP_SQUARE': 0.1295,
            },
            'std': 319.84,
            'half_life': 484,
        },
        'ROBOT_DISHES': {
            'alpha': 15573.1830,
            'betas': {
                'ROBOT_VACUUMING':  0.1399,
                'ROBOT_MOPPING':   -0.3401,
                'ROBOT_LAUNDRY':   -0.2639,
                'ROBOT_IRONING':    0.0629,
                'MICROCHIP_OVAL':  -0.3051,
                'MICROCHIP_SQUARE': 0.1087,
            },
            'std': 256.39,
            'half_life': 259,
        },
        'ROBOT_LAUNDRY': {
            'alpha': 13052.7071,
            'betas': {
                'ROBOT_VACUUMING':  0.1806,
                'ROBOT_MOPPING':   -0.3678,
                'ROBOT_DISHES':    -0.2749,
                'ROBOT_IRONING':   -0.2103,
                'MICROCHIP_OVAL':   0.2668,
                'MICROCHIP_SQUARE': 0.1176,
            },
            'std': 261.71,
            'half_life': 643,
        },
        'ROBOT_IRONING': {
            'alpha': 14563.1917,
            'betas': {
                'ROBOT_VACUUMING': -0.3145,
                'ROBOT_MOPPING':   -0.4436,
                'ROBOT_DISHES':     0.0774,
                'ROBOT_LAUNDRY':   -0.2485,
                'MICROCHIP_OVAL':   0.4676,
                'MICROCHIP_SQUARE': -0.0158,
            },
            'std': 284.45,
            'half_life': 616,
        },
    }

    Z_FULL = 0.3
    Z_TAKE = 1.0
    Z_DEAD = 0.15

    SPREAD_MARGINS = {
        'ROBOT_VACUUMING': 2,
        'ROBOT_MOPPING':   2,
        'ROBOT_DISHES':    2,
        'ROBOT_LAUNDRY':   2,
        'ROBOT_IRONING':   2,
    }
    POSITION_SKEW = 0.5

    def __init__(self):
        self.products = list(self.MODELS.keys())

    @staticmethod
    def _mid(depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid = max(depth.buy_orders.keys())
        ask = min(depth.sell_orders.keys())
        return bid, ask, (bid + ask) / 2.0

    def _target_position(self, z: float) -> int:
        if abs(z) < self.Z_DEAD:
            return 0
        sign = -1.0 if z > 0 else 1.0
        scale = min(1.0, (abs(z) - self.Z_DEAD) / max(self.Z_FULL - self.Z_DEAD, 1e-9))
        return int(round(sign * scale * self.POSITION_LIMIT))

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        mids: Dict[str, float] = {}
        bidask: Dict[str, tuple] = {}
        for prod, depth in state.order_depths.items():
            m = self._mid(depth)
            if m is not None:
                bidask[prod] = (m[0], m[1])
                mids[prod] = m[2]

        for robot, model in self.MODELS.items():
            if robot not in mids:
                continue
            if not all(c in mids for c in model['betas'].keys()):
                continue

            fv = model['alpha'] + sum(beta * mids[c] for c, beta in model['betas'].items())
            spread = mids[robot] - fv
            z = spread / model['std']

            current = state.position.get(robot, 0)
            target = self._target_position(z)
            delta = target - current

            best_bid, best_ask = bidask[robot]
            depth = state.order_depths[robot]
            orders: List[Order] = []

            if delta == 0:
                result[robot] = orders
                continue

            aggressive = abs(z) >= self.Z_TAKE
            take_buffer = model['std'] * self.Z_DEAD

            if delta > 0:
                remaining = delta
                if aggressive:
                    take_cap = fv + take_buffer
                    for px in sorted(depth.sell_orders.keys()):
                        if remaining <= 0 or px > take_cap:
                            break
                        avail = -depth.sell_orders[px]
                        qty = min(remaining, avail)
                        if qty > 0:
                            orders.append(Order(robot, px, qty))
                            remaining -= qty
                if remaining > 0:
                    cap = math.floor(fv - take_buffer)
                    px = min(best_bid + 1, cap)
                    if px < best_ask:
                        orders.append(Order(robot, px, remaining))
            else:
                remaining = -delta
                if aggressive:
                    take_cap = fv - take_buffer
                    for px in sorted(depth.buy_orders.keys(), reverse=True):
                        if remaining <= 0 or px < take_cap:
                            break
                        avail = depth.buy_orders[px]
                        qty = min(remaining, avail)
                        if qty > 0:
                            orders.append(Order(robot, px, -qty))
                            remaining -= qty
                if remaining > 0:
                    cap = math.ceil(fv + take_buffer)
                    px = max(best_ask - 1, cap)
                    if px > best_bid:
                        orders.append(Order(robot, px, -remaining))

            result[robot] = orders

        return result, 0, ""


# ════════════════════════════════════════════════════════════════════════════
# SLEEP — ema_sleep_pd_trader
# ════════════════════════════════════════════════════════════════════════════
class SleepTrader:
    LIMIT = 10
    SKEW_THRESH = 9
    TREND_WINDOW = 50

    PRODUCTS = [
        "SLEEP_POD_COTTON",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_SUEDE",
    ]

    TREND_FILTER = {"SLEEP_POD_LAMB_WOOL", "SLEEP_POD_COTTON", "SLEEP_POD_SUEDE"}

    def __init__(self):
        self.price_history: Dict[str, deque] = {
            p: deque(maxlen=self.TREND_WINDOW) for p in self.PRODUCTS
        }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for symbol in self.PRODUCTS:
            depth: OrderDepth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            mid = (best_bid + best_ask) / 2
            pos = state.position.get(symbol, 0)

            self.price_history[symbol].append(mid)

            buy_price = best_bid + 1
            sell_price = best_ask - 1

            if pos >= self.SKEW_THRESH:
                sell_price -= 1
            elif pos <= -self.SKEW_THRESH:
                buy_price += 1

            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

            buy_qty = max(0, self.LIMIT - pos)
            sell_qty = max(0, self.LIMIT + pos)

            if symbol in self.TREND_FILTER:
                hist = self.price_history[symbol]
                if len(hist) >= self.TREND_WINDOW:
                    trend = hist[-1] - hist[0]
                    if trend > 0:
                        buy_qty = 0
                        if pos > 0:
                            sell_qty = pos
                    elif trend < 0:
                        sell_qty = 0
                        if pos < 0:
                            buy_qty = -pos

            orders: List[Order] = []
            if buy_qty > 0:
                orders.append(Order(symbol, buy_price, int(buy_qty)))
            if sell_qty > 0:
                orders.append(Order(symbol, sell_price, -int(sell_qty)))

            if orders:
                result[symbol] = orders

        return result, 0, ""


# ════════════════════════════════════════════════════════════════════════════
# SNACK — ema_snack_trader
# ════════════════════════════════════════════════════════════════════════════
class SnackTrader:
    def __init__(self):
        self.limit = 10
        self.products = [
            'SNACKPACK_CHOCOLATE',
            'SNACKPACK_PISTACHIO',
            'SNACKPACK_RASPBERRY',
            'SNACKPACK_STRAWBERRY',
            'SNACKPACK_VANILLA'
        ]

        self.avg_prices = {
            'SNACKPACK_CHOCOLATE': 10000.0,
            'SNACKPACK_PISTACHIO': 9500.0,
            'SNACKPACK_RASPBERRY': 10000.0,
            'SNACKPACK_STRAWBERRY': 10500.0,
            'SNACKPACK_VANILLA': 10000.0
        }

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
                data = {"ema_weights": dict(self.weights)}
        else:
            data = {"ema_weights": dict(self.weights)}

        if "ema_weights" not in data:
            data["ema_weights"] = dict(self.weights)

        current_prices = {}
        for p in self.products:
            mid = self.get_mid_price(state.order_depths.get(p))
            if mid is not None:
                current_prices[p] = mid

        if not current_prices:
            return result, conversions, json.dumps(data)

        cluster_sum = sum(current_prices.values())
        current_rel_weights = {p: current_prices[p] / cluster_sum for p in current_prices}

        for product in current_prices:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)

            if product in data["ema_weights"]:
                data["ema_weights"][product] = self.alpha * current_rel_weights[product] + (1 - self.alpha) * data["ema_weights"][product]
            else:
                data["ema_weights"][product] = current_rel_weights[product]

            weight_dev = current_rel_weights[product] - data["ema_weights"][product]
            target_pos = -weight_dev * 5000.0
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


# ════════════════════════════════════════════════════════════════════════════
# TRANSLATOR — ema_translator_trader (OLS + EMA hybrid)
# ════════════════════════════════════════════════════════════════════════════
class TranslatorTrader:
    def __init__(self):
        self.ols_models = {
            "TRANSLATOR_SPACE_GRAY": (15100, -0.3, +0.13, +0.05, -0.13, -0.3, 105),
            "TRANSLATOR_VOID_BLUE":  (17267, -0.003, -0.34, -0.4, -0.14, +0.17,  91),
        }
        self.MICROCHIPS = [
            "MICROCHIP_CIRCLE", "MICROCHIP_OVAL",
            "MICROCHIP_RECTANGLE", "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"
        ]
        self.ols_position_limit = 10
        self.ols_spread_margins = {
            "TRANSLATOR_SPACE_GRAY": 2,
            "TRANSLATOR_VOID_BLUE":  3,
        }

        self.ema_trade = {
            "TRANSLATOR_ASTRO_BLACK",
            "TRANSLATOR_ECLIPSE_CHARCOAL",
            "TRANSLATOR_GRAPHITE_MIST",
        }
        self.all_translators = [
            "TRANSLATOR_ASTRO_BLACK",
            "TRANSLATOR_ECLIPSE_CHARCOAL",
            "TRANSLATOR_GRAPHITE_MIST",
            "TRANSLATOR_SPACE_GRAY",
            "TRANSLATOR_VOID_BLUE",
        ]
        self.ema_limits = {p: 10 for p in self.ema_trade}
        self.alpha = 0.01
        self.entry_z = 1.5
        self.exit_z = 0.2

    def get_mid(self, product, state: TradingState):
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    def _run_ols(self, state: TradingState, result: dict, mc_mids: dict):
        for product, coeffs in self.ols_models.items():
            depth = state.order_depths.get(product)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            alpha, betas, std_err = coeffs[0], coeffs[1:6], coeffs[6]
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2.0
            pos = state.position.get(product, 0)
            margin = self.ols_spread_margins[product]
            pos_lim = self.ols_position_limit

            fair_value = alpha + sum(b * mc_mids[mc] for b, mc in zip(betas, self.MICROCHIPS))
            z_score = (mid_price - fair_value) / std_err

            target = int(-z_score * 4)
            target = max(-pos_lim, min(pos_lim, target))
            diff = target - pos
            orders = []

            if z_score < -1.0 and diff > 0:
                orders.append(Order(product, best_ask, diff))
            elif z_score > 1.0 and diff < 0:
                orders.append(Order(product, best_bid, diff))
            else:
                my_bid = int(round(fair_value - margin - pos * 0.5))
                my_ask = int(round(fair_value + margin - pos * 0.5))
                if pos_lim - pos > 0:
                    orders.append(Order(product, min(my_bid, best_bid + 1), pos_lim - pos))
                if -pos_lim - pos < 0:
                    orders.append(Order(product, max(my_ask, best_ask - 1), -pos_lim - pos))

            if orders:
                result[product] = orders

    def _run_ema(self, state: TradingState, result: dict, data: dict):
        ema_state = data.get("ema", {})
        m2_state = data.get("m2", {})

        all_mids = {}
        for p in self.all_translators:
            m = self.get_mid(p, state)
            if m is not None:
                all_mids[p] = m

        if len(all_mids) < 3:
            return

        basket_mean = sum(all_mids.values()) / len(all_mids)

        signals, stds = {}, {}
        for p in self.all_translators:
            if p not in all_mids:
                continue
            rel_price = all_mids[p] - basket_mean
            curr_ema = ema_state.get(p, rel_price)
            new_ema = self.alpha * rel_price + (1 - self.alpha) * curr_ema
            ema_state[p] = new_ema

            diff = rel_price - new_ema
            curr_m2 = m2_state.get(p, 70.0 ** 2)
            new_m2 = self.alpha * (diff ** 2) + (1 - self.alpha) * curr_m2
            m2_state[p] = new_m2

            stds[p] = math.sqrt(new_m2)
            signals[p] = diff

        for p in self.ema_trade:
            if p not in signals or p not in stds:
                continue
            z = signals[p] / stds[p] if stds[p] > 0 else 0
            pos = state.position.get(p, 0)
            limit = self.ema_limits[p]
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            best_ask = min(depth.sell_orders)
            best_bid = max(depth.buy_orders)
            orders = []

            if z < -self.entry_z:
                qty = limit - pos
                if qty > 0:
                    orders.append(Order(p, min(best_bid + 1, best_ask - 1), qty))
            elif z > self.entry_z:
                qty = -limit - pos
                if qty < 0:
                    orders.append(Order(p, max(best_ask - 1, best_bid + 1), qty))
            elif abs(z) < self.exit_z:
                if pos > 0:
                    orders.append(Order(p, max(best_ask - 1, best_bid + 1), -pos))
                elif pos < 0:
                    orders.append(Order(p, min(best_bid + 1, best_ask - 1), -pos))

            if orders:
                result[p] = orders

        data["ema"] = ema_state
        data["m2"] = m2_state

    def run(self, state: TradingState):
        result = {}
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except:
                data = {}

        mc_mids = {mc: self.get_mid(mc, state) for mc in self.MICROCHIPS}
        if None not in mc_mids.values():
            self._run_ols(state, result, mc_mids)

        self._run_ema(state, result, data)

        return result, 0, json.dumps(data)


# ════════════════════════════════════════════════════════════════════════════
# VR — ema_vr_trader
# ════════════════════════════════════════════════════════════════════════════
class VRTrader:
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
        if "ema_diff" not in data:
            data["ema_diff"] = {}
        if "var_diff" not in data:
            data["var_diff"] = {}

        current_prices = {}
        for p in self.products:
            mid = self.get_mid_price(state.order_depths.get(p))
            if mid is not None:
                current_prices[p] = mid

        if len(current_prices) < len(self.products):
            return {}, 0, json.dumps(data)

        signals = {p: 0.0 for p in self.products}

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
                z_capped = max(min(z, 3.0), -3.0)

                signals[p1] -= z_capped
                signals[p2] += z_capped

        for product in self.products:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)

            mid = current_prices[product]

            target_pos = signals[product] * (self.limit / 6.0)
            target_pos = max(min(target_pos, self.limit), -self.limit)

            fair_val = mid + (target_pos - curr_pos) * 1.5

            edge = 1.5

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            bid_price = min(int(math.floor(fair_val - edge)), best_bid + 1)
            ask_price = max(int(math.ceil(fair_val + edge)), best_ask - 1)

            if curr_pos < self.limit:
                orders.append(Order(product, bid_price, self.limit - curr_pos))
            if curr_pos > -self.limit:
                orders.append(Order(product, ask_price, -self.limit - curr_pos))

            result[product] = orders

        return result, conversions, json.dumps(data)


# ════════════════════════════════════════════════════════════════════════════
# MASTER — orchestratore: ogni sub-trader ha logica e variabili separate.
# Il traderData è namespacato per sub-trader: ognuno legge/scrive solo il suo.
# ════════════════════════════════════════════════════════════════════════════
class Trader:
    SUB_TRADERS = [
        ("chip", ChipTrader),
        ("galaxy", GalaxyTrader),
        ("oxygen", OxygenTrader),
        ("panel", PanelTrader),
        ("pebbles", PebblesTrader),
        ("robo", RoboTrader),
        ("sleep", SleepTrader),
        ("snack", SnackTrader),
        ("translator", TranslatorTrader),
        ("vr", VRTrader),
    ]

    def __init__(self):
        self.traders = {name: cls() for name, cls in self.SUB_TRADERS}

    def run(self, state: TradingState):
        if state.traderData:
            try:
                master_data = json.loads(state.traderData)
                if not isinstance(master_data, dict):
                    master_data = {}
            except Exception:
                master_data = {}
        else:
            master_data = {}

        merged_result: Dict[str, List[Order]] = {}
        new_master_data: Dict[str, str] = {}
        total_conversions = 0

        original_trader_data = state.traderData

        for name, _ in self.SUB_TRADERS:
            sub_data = master_data.get(name, "")
            if not isinstance(sub_data, str):
                sub_data = json.dumps(sub_data)
            state.traderData = sub_data

            try:
                sub_result, sub_conv, sub_out = self.traders[name].run(state)
            except Exception:
                sub_result, sub_conv, sub_out = {}, 0, sub_data

            for sym, orders in (sub_result or {}).items():
                if not orders:
                    continue
                merged_result.setdefault(sym, []).extend(orders)

            total_conversions += sub_conv or 0
            new_master_data[name] = sub_out if isinstance(sub_out, str) else json.dumps(sub_out)

        state.traderData = original_trader_data

        return merged_result, total_conversions, json.dumps(new_master_data)
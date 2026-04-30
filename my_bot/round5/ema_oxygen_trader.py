import json
from datamodel import Order, TradingState, OrderDepth
from typing import Dict, List, Optional


class Trader:
    def __init__(self):
        self.LIMIT = 10

        # ───────────────────────────────────────────────────────────
        # LEG 1: THE BREATH PAIR (Structural Parity)
        # ───────────────────────────────────────────────────────────
        self.P1_A = 'OXYGEN_SHAKE_MORNING_BREATH'
        self.P1_B = 'OXYGEN_SHAKE_EVENING_BREATH'

        self.P1_MEAN_ALPHA = 0.0005
        self.P1_VOL_ALPHA = 0.005
        self.P1_START_MEAN = 250.0

        # The Floor: Even if the market goes dead, assume at least 10 credits of noise
        self.P1_VOL_FLOOR = 30.0

        self.P1_ENTRY_Z = 3
        self.P1_EXIT_Z = 0

        # ───────────────────────────────────────────────────────────
        # LEG 2: THE FLAVOR PAIR (Statistical Parity)
        # ───────────────────────────────────────────────────────────
        self.P2_A = 'OXYGEN_SHAKE_GARLIC'
        self.P2_B = 'OXYGEN_SHAKE_CHOCOLATE'

        self.P2_MEAN_ALPHA = 0.0005  # Fast memory for regime shifts
        self.P2_VOL_ALPHA = 0.005
        self.P2_START_MEAN = 2600

        # The Floor: Garlic/Choc is wider, assume at least 20 credits of noise
        self.P2_VOL_FLOOR = 20.0

        # Tighter thresholds to increase trade frequency
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

            # --- ADAPTIVE MEMORY ---
            mean_key = f"{leg_id}_mean"
            init_mean = starting_mean if starting_mean is not None else spread
            current_mean = trader_data.get(mean_key, init_mean)

            new_mean = (spread * mean_alpha) + \
                (current_mean * (1 - mean_alpha))
            trader_data[mean_key] = new_mean

            vol_key = f"{leg_id}_vol"
            current_vol = trader_data.get(vol_key, vol_floor)
            residual = abs(spread - new_mean)
            new_vol = (residual * vol_alpha) + (current_vol * (1 - vol_alpha))
            trader_data[vol_key] = new_vol

            # --- THE VOLATILITY FLOOR PROTECTED Z-SCORE ---
            # If new_vol crashes to 1.0, we divide by vol_floor instead.
            effective_vol = max(new_vol, vol_floor)
            z_score = (spread - new_mean) / effective_vol

            # --- BINARY EXECUTION ENGINE (Max Aggression) ---
            pos_a = state.position.get(sym_a, 0)
            target = pos_a

            # Entry
            if pos_a == 0:
                if z_score > entry_z:
                    target = -self.LIMIT
                elif z_score < -entry_z:
                    target = self.LIMIT
            # Exit (Reversion)
            else:
                if pos_a < 0 and z_score < exit_z:
                    target = 0
                elif pos_a > 0 and z_score > -exit_z:
                    target = 0

            # --- MARKET TAKER EXECUTION ---
            for sym, t_pos in [(sym_a, target), (sym_b, -target)]:
                pos = state.position.get(sym, 0)
                diff = t_pos - pos
                if abs(diff) >= 1:
                    depth = state.order_depths[sym]
                    price = min(depth.sell_orders.keys()) if diff > 0 else max(
                        depth.buy_orders.keys())
                    result.setdefault(sym, []).append(
                        Order(sym, int(price), diff))

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        self.process_leg(state, self.P1_A, self.P1_B, self.P1_ENTRY_Z, self.P1_EXIT_Z,
                         self.P1_MEAN_ALPHA, self.P1_VOL_ALPHA, self.P1_START_MEAN, self.P1_VOL_FLOOR,
                         "P1", trader_data, result)

        self.process_leg(state, self.P2_A, self.P2_B, self.P2_ENTRY_Z, self.P2_EXIT_Z,
                         self.P2_MEAN_ALPHA, self.P2_VOL_ALPHA, self.P2_START_MEAN, self.P2_VOL_FLOOR,
                         "P2", trader_data, result)

        return result, 0, json.dumps(trader_data)
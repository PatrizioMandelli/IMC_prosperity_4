from typing import List, Dict, Any
from datamodel import OrderDepth, TradingState, Order
import math

# =====================================================================
# MATH KERNEL
# =====================================================================


def norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call_and_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K), 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
    return price, delta


def find_iv(target_price: float, S: float, K: float, T: float) -> float:
    if target_price <= max(0.0, S - K):
        return 0.0001
    low, high = 0.001, 3.0
    for _ in range(12):
        mid = (low + high) / 2.0
        if bs_call_and_delta(S, K, T, mid)[0] > target_price:
            high = mid
        else:
            low = mid
    return mid


def get_microprice(depth: OrderDepth) -> float:
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    bid_vol = abs(depth.buy_orders[best_bid])
    ask_vol = abs(depth.sell_orders[best_ask])
    total_vol = bid_vol + ask_vol
    return (best_bid * ask_vol + best_ask * bid_vol) / total_vol if total_vol > 0 else 0.0


def get_vwap(depth: OrderDepth) -> float:
    vol_price_sum = 0.0
    total_vol = 0.0
    for p, v in depth.buy_orders.items():
        vol_price_sum += p * abs(v)
        total_vol += abs(v)
    for p, v in depth.sell_orders.items():
        vol_price_sum += p * abs(v)
        total_vol += abs(v)
    return vol_price_sum / total_vol if total_vol > 0 else 0.0

# =====================================================================
# THE COMPARTMENTALIZED MASTER TRADER
# =====================================================================


class Trader:
    def __init__(self):
        self.UL_SYM = "VELVETFRUIT_EXTRACT"
        self.OPT_LIMIT = 300
        self.UL_LIMIT = 200

        # ---------------------------------------------------------
        # TOGGLE FLAGS FOR A/B TESTING
        # ---------------------------------------------------------
        self.FLAGS = {
            # If False, entire options chain uses Basic MM. UL never takes liquidity.
            "ENABLE_GAMMA_SCALPING": True,
            "ENABLE_OPTIONS_MM": True,     # Enables Basic MM on OTM options
            "ENABLE_UL_MM": True           # Enables passive skewed quoting on Underlying
        }

        # --- PARAMETERS ---
        # Distance from VWAP to be considered ATM (Gamma target)
        self.BELLY_DISTANCE = 300
        self.FALLBACK_IV = 0.23
        # Delta deviation required to trigger aggressive taker scalp
        self.GAMMA_BAND = 20.0
        # 10% edge required for Options MM (from MMbasic.py)
        self.MM_EDGE_PCT = 0.10

    def run(self, state: TradingState):
        result = {prod: [] for prod in state.order_depths}

        ul_depth = state.order_depths.get(self.UL_SYM)
        if not ul_depth or not ul_depth.buy_orders or not ul_depth.sell_orders:
            return result, 0, ""

        ul_vwap = get_vwap(ul_depth)
        best_ul_bid = max(ul_depth.buy_orders.keys())
        best_ul_ask = min(ul_depth.sell_orders.keys())
        TTE = max(1e-6, (5.0 - (state.timestamp / 1000000.0)) / 252.0)

        # Dynamic IV Centering (Anchor to the truest ATM option)
        options_list = [sym for sym in state.order_depths.keys()
                        if sym.startswith("VEV_")]
        atm_sym = min(options_list, key=lambda sym: abs(
            float(sym.split('_')[1]) - ul_vwap))
        atm_strike = float(atm_sym.split('_')[1])

        current_market_iv = self.FALLBACK_IV
        if atm_sym in state.order_depths and state.order_depths[atm_sym].buy_orders:
            atm_mid = get_microprice(state.order_depths[atm_sym])
            current_market_iv = find_iv(atm_mid, ul_vwap, atm_strike, TTE)

        total_options_delta = 0.0

        # =====================================================================
        # COMPARTMENT 1: OPTIONS LOGIC
        # =====================================================================
        for opt_sym in options_list:
            depth = state.order_depths.get(opt_sym)
            if not depth.buy_orders or not depth.sell_orders:
                continue

            K = float(opt_sym.split('_')[1])
            pos = state.position.get(opt_sym, 0)

            best_opt_bid = max(depth.buy_orders.keys())
            best_opt_ask = min(depth.sell_orders.keys())
            market_spread = best_opt_ask - best_opt_bid

            # Track portfolio delta (Always tracked so MM can hedge passively)
            fv, delta = bs_call_and_delta(ul_vwap, K, TTE, current_market_iv)
            total_options_delta += pos * delta

            buy_cap = self.OPT_LIMIT - pos
            sell_cap = -self.OPT_LIMIT - pos

            is_belly = abs(K - ul_vwap) <= self.BELLY_DISTANCE

            # --- BEHAVIOR A: GAMMA ACCUMULATOR ---
            if self.FLAGS["ENABLE_GAMMA_SCALPING"] and is_belly:
                # Passively accumulate ATM options to build Gamma
                if buy_cap > 0:
                    my_bid = best_opt_bid + 1
                    if my_bid >= best_opt_ask:
                        my_bid = best_opt_bid
                    result[opt_sym].append(Order(opt_sym, my_bid, buy_cap))

                # If short, cover immediately
                if pos < 0:
                    result[opt_sym].append(
                        Order(opt_sym, best_opt_ask - 1, -pos))

            # --- BEHAVIOR B: BASIC MARKET MAKER (From MMbasic.py) ---
            elif self.FLAGS["ENABLE_OPTIONS_MM"]:
                required_edge = max(1.0, market_spread * self.MM_EDGE_PCT)

                my_bid = min(best_opt_bid + 1,
                             int(math.floor(fv - required_edge)))
                my_ask = max(best_opt_ask - 1,
                             int(math.ceil(fv + required_edge)))

                best_opt_bid = max(depth.buy_orders.keys())
                best_opt_ask = min(depth.sell_orders.keys())
                market_spread = best_opt_ask - best_opt_bid

                # Wide-Spread Filter: Avoid highly efficient markets
                if market_spread < 6.0:
                    continue

                # Passive Inventory Skewing (Directly from MMbasic.py)
                if pos > 100:
                    my_bid -= 1
                    my_ask -= 1
                if pos > 200:
                    my_bid -= 1
                    my_ask -= 1
                if pos < -100:
                    my_bid += 1
                    my_ask += 1
                if pos < -200:
                    my_bid += 1
                    my_ask += 1

                if my_bid >= my_ask:
                    my_bid = best_opt_bid
                    my_ask = best_opt_ask

                if buy_cap > 0:
                    result[opt_sym].append(Order(opt_sym, my_bid, buy_cap))
                if sell_cap < 0:
                    result[opt_sym].append(Order(opt_sym, my_ask, sell_cap))

        # =====================================================================
        # COMPARTMENT 2: UNDERLYING LOGIC
        # =====================================================================
        ul_pos = state.position.get(self.UL_SYM, 0)
        target_ul_pos = -total_options_delta
        inventory_offset = ul_pos - target_ul_pos

        ul_buy_cap = self.UL_LIMIT - ul_pos
        ul_sell_cap = -self.UL_LIMIT - ul_pos

        # --- BEHAVIOR A: AGGRESSIVE GAMMA SCALP ---
        # ONLY triggers if Gamma Scalping is explicitly allowed
        if self.FLAGS["ENABLE_GAMMA_SCALPING"] and abs(inventory_offset) > self.GAMMA_BAND:
            # Need to buy (Delta is positive)
            if inventory_offset < 0 and ul_buy_cap > 0:
                trade_vol = min(int(abs(inventory_offset)), ul_buy_cap)
                result[self.UL_SYM].append(
                    Order(self.UL_SYM, best_ul_ask, trade_vol))  # Cross spread

            # Need to sell (Delta is negative)
            elif inventory_offset > 0 and ul_sell_cap < 0:
                trade_vol = max(int(-inventory_offset), ul_sell_cap)
                result[self.UL_SYM].append(
                    Order(self.UL_SYM, best_ul_bid, trade_vol))  # Cross spread

        # --- BEHAVIOR B: PASSIVE SKEWED MARKET MAKER ---
        elif self.FLAGS["ENABLE_UL_MM"]:
            my_ul_bid = best_ul_bid + 1
            my_ul_ask = best_ul_ask - 1

            # Inventory skewing based on Delta offset
            if inventory_offset > 50:
                my_ul_bid -= 1
                my_ul_ask -= 1
            if inventory_offset > 120:
                my_ul_bid -= 1
                my_ul_ask -= 1
            if inventory_offset < -50:
                my_ul_bid += 1
                my_ul_ask += 1
            if inventory_offset < -120:
                my_ul_bid += 1
                my_ul_ask += 1

            # Strict Maker Bounds
            my_ul_bid = min(my_ul_bid, best_ul_ask - 1)
            my_ul_ask = max(my_ul_ask, best_ul_bid + 1)
            if my_ul_bid >= my_ul_ask:
                my_ul_bid, my_ul_ask = best_ul_bid, best_ul_ask

            if ul_buy_cap > 0:
                result[self.UL_SYM].append(
                    Order(self.UL_SYM, my_ul_bid, ul_buy_cap))
            if ul_sell_cap < 0:
                result[self.UL_SYM].append(
                    Order(self.UL_SYM, my_ul_ask, ul_sell_cap))

        # Telemetry
        if state.timestamp % 10000 == 0:
            print(
                f"MASTER LOG | Scalping: {self.FLAGS['ENABLE_GAMMA_SCALPING']} | Opt Delta: {total_options_delta:.1f} | UL Pos: {ul_pos}")

        return result, 0, ""
import json
import math
from typing import List

import numpy as np

from datamodel import OrderDepth, Order, TradingState


# --- Pricing helpers ---------------------------------------------------------


def norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def calculate_bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def find_iv(target_price: float, S: float, K: float, T: float) -> float:
    """Bisection IV solver. Bounded to [0.001, 3.0] in 15 iterations."""
    if target_price <= max(0.0, S - K):
        return 0.0001
    low, high = 0.001, 3.0
    for _ in range(15):
        mid = (low + high) / 2.0
        if calculate_bs_call(S, K, T, 0.0, mid) > target_price:
            high = mid
        else:
            low = mid
    return mid


def get_microprice(depth: OrderDepth) -> float:
    """Liquidity-weighted mid: weight each side's price by the opposite size."""
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    bid_vol = abs(depth.buy_orders[best_bid])
    ask_vol = abs(depth.sell_orders[best_ask])
    total = bid_vol + ask_vol
    return (best_bid * ask_vol + best_ask * bid_vol) / total if total > 0 else 0.0


# --- Trader ------------------------------------------------------------------


class Trader:
    """EMA trend-follower for the underlying + OTM options, BS quoter for ITM/ATM."""

    def __init__(self):
        self.STARTING_TTE = 5.0
        self.FALLBACK_VOL = 0.23
        self.OPTION_LIMIT = 300

    def run(self, state: TradingState):
        result = {prod: [] for prod in state.order_depths.keys()}

        try:
            memory = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            memory = {}

        ul_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ul_vwap = get_microprice(ul_depth) if ul_depth else None

        otm_options = []
        itm_options = []

        if ul_vwap:
            for product in state.order_depths.keys():
                if product.startswith("VEV_"):
                    strike = float(product.split("_")[1])
                    if strike > ul_vwap:
                        otm_options.append(product)
                    else:
                        itm_options.append(product)

        # Trend module: base assets + OTM options use the EMA logic.
        base_products = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
        for product in base_products + otm_options:
            if product in state.order_depths:
                result[product] = self.logic_ema(
                    state, product, state.order_depths[product], memory
                )

        # Black-Scholes module: only on ITM/ATM options.
        if ul_vwap and ul_depth:
            TTE = max(1e-6, (self.STARTING_TTE - (state.timestamp / 1000000.0)) / 252.0)

            # Fit a quadratic IV smile across both OTM and ITM points for accuracy.
            moneyness_points = []
            iv_points = []
            for opt in otm_options + itm_options:
                depth = state.order_depths.get(opt)
                if not depth or not depth.buy_orders or not depth.sell_orders:
                    continue
                opt_vwap = get_microprice(depth)
                K = float(opt.split("_")[1])
                m = K / ul_vwap
                iv = find_iv(opt_vwap, ul_vwap, K, TTE)
                if 0.01 < iv < 1.5:
                    moneyness_points.append(m)
                    iv_points.append(iv)

            coeffs = (
                np.polyfit(moneyness_points, iv_points, 2)
                if len(moneyness_points) >= 3
                else None
            )

            for product in itm_options:
                depth = state.order_depths.get(product)
                if not depth or not depth.buy_orders or not depth.sell_orders:
                    continue

                best_opt_bid = max(depth.buy_orders.keys())
                best_opt_ask = min(depth.sell_orders.keys())
                market_spread = best_opt_ask - best_opt_bid
                if market_spread < 4.0:
                    continue

                strike = float(product.split("_")[1])
                pos = state.position.get(product, 0)
                fair_iv = (
                    np.polyval(coeffs, strike / ul_vwap)
                    if coeffs is not None
                    else self.FALLBACK_VOL
                )

                fv = calculate_bs_call(ul_vwap, strike, TTE, 0.0, fair_iv)
                base_edge = max(1.0, market_spread * 0.10)

                # Price skew based on inventory tier.
                skew = 0
                if pos > 100:
                    skew = 1
                elif pos > 200:
                    skew = 2
                elif pos < -100:
                    skew = -1
                elif pos < -200:
                    skew = -2

                adj_fv = fv - skew
                buy_cap = self.OPTION_LIMIT - pos
                sell_cap = -self.OPTION_LIMIT - pos
                opt_layers = [(0.3, 0), (0.3, 1), (0.4, 2)]

                if buy_cap > 0:
                    for pct, extra_edge in opt_layers:
                        qty = math.floor(buy_cap * pct)
                        if qty <= 0:
                            continue
                        price = min(
                            best_opt_bid + 1,
                            int(math.floor(adj_fv - base_edge - extra_edge)),
                        )
                        result[product].append(Order(product, price, qty))

                if sell_cap < 0:
                    for pct, extra_edge in opt_layers:
                        qty = math.ceil(sell_cap * pct)
                        if qty >= 0:
                            continue
                        price = max(
                            best_opt_ask - 1,
                            int(math.ceil(adj_fv + base_edge + extra_edge)),
                        )
                        result[product].append(Order(product, price, qty))

        return result, 0, json.dumps(memory)

    def logic_ema(
        self,
        state: TradingState,
        product: str,
        depth: OrderDepth,
        memory: dict,
    ) -> List[Order]:
        """Fast/slow EMA trend MM with volatility-aware sizing and inventory skew."""
        orders: List[Order] = []
        if not depth.sell_orders or not depth.buy_orders:
            return orders

        LIMIT = 200
        current_pos = state.position.get(product, 0)
        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        pkey = product + "_"

        def deep_w(om, rev):
            """Top-3-level volume-weighted price."""
            t, ws = 0, 0
            for p in sorted(om.keys(), reverse=rev)[:3]:
                v = abs(om[p])
                ws += p * v
                t += v
            return ws / t if t else None

        w_bid = deep_w(depth.buy_orders, True)
        w_ask = deep_w(depth.sell_orders, False)
        deep_mid = (
            (w_bid + w_ask) / 2.0 if w_bid and w_ask else (best_bid + best_ask) / 2.0
        )

        # First-tick init.
        if pkey + "fast_fair" not in memory:
            memory[pkey + "fast_fair"] = deep_mid
            memory[pkey + "slow_fair"] = deep_mid
            memory[pkey + "prev_mid"] = deep_mid
            memory[pkey + "var_est"] = 0.0

        prev_mid = memory.get(pkey + "prev_mid", deep_mid)
        price_change = deep_mid - prev_mid
        memory[pkey + "var_est"] = (
            0.05 * (price_change ** 2) + 0.95 * memory.get(pkey + "var_est", 0.0)
        )
        memory[pkey + "prev_mid"] = deep_mid
        vol_est = (
            math.sqrt(memory[pkey + "var_est"]) if memory[pkey + "var_est"] > 0 else 0.0
        )

        diff = abs(memory[pkey + "fast_fair"] - memory[pkey + "slow_fair"])
        a_f = 0.35 if product == "HYDROGEL_PACK" else (0.55 if diff > 1.5 else 0.35)
        a_s = 0.01

        memory[pkey + "fast_fair"] = a_f * deep_mid + (1 - a_f) * memory[pkey + "fast_fair"]
        memory[pkey + "slow_fair"] = a_s * deep_mid + (1 - a_s) * memory[pkey + "slow_fair"]

        trend = memory[pkey + "fast_fair"] - memory[pkey + "slow_fair"]
        fair_value = 0.7 * memory[pkey + "fast_fair"] + 0.3 * memory[pkey + "slow_fair"]

        # Volatility regime drives size and taker threshold.
        skew_div, max_maker, taker_threshold, max_taker = 50.0, 195, 2.0, 50
        if vol_est > 3.5:
            skew_div, max_maker, taker_threshold, max_taker = 18.0, 25, 1.5, 25
        elif vol_est > 1.5 and abs(trend) > 2.0:
            skew_div, max_maker, taker_threshold, max_taker = 65.0, 190, 1.0, 75

        inv_skew = current_pos / skew_div
        if abs(current_pos) > 130:
            sm = ((abs(current_pos) - 130) / 40.0) ** 3 / 2
            inv_skew = (1 if current_pos > 0 else -1) * sm * 10 + (current_pos / skew_div)

        # In low-vol trend regimes, soften skew when our inventory rides the trend.
        if (
            vol_est <= 3.5
            and abs(trend) > 2.0
            and (
                (trend > 0 and current_pos > 0) or (trend < 0 and current_pos < 0)
            )
        ):
            inv_skew *= 0.5

        adjusted_fair = fair_value - inv_skew
        buy_cap = LIMIT - current_pos
        sell_cap = -LIMIT - current_pos

        # Taker side.
        for ask_p in sorted(depth.sell_orders.keys()):
            if ask_p < adjusted_fair - taker_threshold and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[ask_p]), max_taker)
                orders.append(Order(product, ask_p, qty))
                buy_cap -= qty
        for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_p > adjusted_fair + taker_threshold and sell_cap < 0:
                qty = max(sell_cap, -abs(depth.buy_orders[bid_p]), -max_taker)
                orders.append(Order(product, bid_p, qty))
                sell_cap -= abs(qty)

        # Maker quotes; size shrinks exponentially against inventory.
        pos_ratio = current_pos / float(LIMIT)
        bid_mult = math.exp(-2.0 * pos_ratio) if pos_ratio > 0 else (1 - pos_ratio)
        ask_mult = math.exp(2.0 * pos_ratio) if pos_ratio < 0 else (1 + pos_ratio)
        bid_s = max(5, min(max_maker, int(190 * bid_mult)))
        ask_s = max(5, min(max_maker, int(190 * ask_mult)))

        off_b = 1 if abs(pos_ratio) < 0.4 else (2 if abs(pos_ratio) < 0.7 else 3)
        off_a = 1 if abs(pos_ratio) < 0.4 else (2 if abs(pos_ratio) < 0.7 else 3)

        my_bid = best_bid + off_b
        my_ask = best_ask - off_a
        if my_bid >= my_ask:
            my_bid = int(adjusted_fair - 1)
            my_ask = int(adjusted_fair + 1)

        # Hard inventory unwind clamp on extreme positions.
        b_cl, a_cl = -0.5, 0.5
        if current_pos > 150:
            a_cl = -2.0
        elif current_pos < -150:
            b_cl = 2.0

        my_bid = min(my_bid, int(math.floor(fair_value + b_cl)))
        my_ask = max(my_ask, int(math.ceil(fair_value + a_cl)))

        if buy_cap > 0:
            orders.append(Order(product, my_bid, min(buy_cap, bid_s)))
        if sell_cap < 0:
            orders.append(Order(product, my_ask, max(sell_cap, -ask_s)))

        return orders

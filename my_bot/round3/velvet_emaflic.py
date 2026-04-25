"""
velvet_v14 — key fix over v13:
  Remove my_ask = max(my_ask, fair_int) clamping.
  This was pulling asks above the market for options where model price > market price,
  preventing symmetric MM on VEV_5300 (37 trades/day!) and others.
  The residual logic already protects against selling cheap (negative residual raises ask).
  Keep: my_bid = min(my_bid, fair_int) — protects against buying when market > model.
"""

import math
import json
from datamodel import OrderDepth, TradingState, Order

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call_and_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K), (1.0 if S > K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2), norm_cdf(d1)


def find_iv(target_price: float, S: float, K: float, T: float) -> float:
    if T <= 0 or target_price <= max(0.0, S - K):
        return 0.0001
    low, high = 0.001, 5.0
    for _ in range(20):
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
    total = bid_vol + ask_vol
    return (best_bid * ask_vol + best_ask * bid_vol) / total if total > 0 else (best_bid + best_ask) / 2.0


def _polyfit2_fallback(xs: list, ys: list):
    s0 = len(xs)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x * x * x for x in xs)
    s4 = sum(x * x * x * x for x in xs)
    t0 = sum(ys)
    t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum(x * x * y for x, y in zip(xs, ys))
    M = [[s4, s3, s2], [s3, s2, s1], [s2, s1, float(s0)]]
    rhs = [t2, t1, t0]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        if abs(M[col][col]) < 1e-14:
            continue
        for row in range(col + 1, 3):
            f = M[row][col] / M[col][col]
            for k in range(col, 3): M[row][k] -= f * M[col][k]
            rhs[row] -= f * rhs[col]
    coeffs = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1):
        if abs(M[i][i]) < 1e-14:
            continue
        coeffs[i] = (rhs[i] - sum(M[i][j] * coeffs[j] for j in range(i + 1, 3))) / M[i][i]
    return coeffs[0], coeffs[1], coeffs[2]


def fit_smile(moneyness: list, ivs: list):
    n = len(moneyness)
    if n < 3:
        c = sum(ivs) / n if n > 0 else 0.25
        return 0.0, 0.0, c, (n >= 1)
    if _HAS_NUMPY:
        try:
            coeffs = np.polyfit(moneyness, ivs, 2)
            a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
            return max(0.0, a), b, c, True
        except Exception:
            pass
    a, b, c = _polyfit2_fallback(moneyness, ivs)
    return max(0.0, a), b, c, True


class Trader:
    def __init__(self):
        self.UL_SYM = "VELVETFRUIT_EXTRACT"
        self.OPT_LIMIT = 300
        self.UL_LIMIT = 200

        self.RESIDUAL_THRESHOLD = 0.005
        self.MAX_EXPECTED_RESIDUAL = 0.04
        self.BASE_VOL = 10
        self.MAX_TRADE_PCT = 0.30
        self.MONEYNESS_FILTER = 0.25
        self.INV_SKEW_START = 30
        self.INV_SKEW_PER_UNIT = 50
        self.HEDGE_EMERGENCY = 150
        self.HEDGE_SKEW_DIV = 50.0
        self.LOG_INTERVAL = 1000

    def run(self, state: TradingState):
        result = {}

        td = {}
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        prev_ts = td.get("prev_ts", -1)
        day_idx = td.get("day_idx", 0)
        if state.timestamp < prev_ts and prev_ts > 0:
            day_idx += 1
        td["prev_ts"] = state.timestamp
        td["day_idx"] = day_idx

        TTE = max(1e-6, (5.0 - day_idx - state.timestamp / 1_000_000.0) / 252.0)

        ul_depth = state.order_depths.get(self.UL_SYM)
        if not ul_depth or not ul_depth.buy_orders or not ul_depth.sell_orders:
            return {}, 0, json.dumps(td)

        best_ul_bid = max(ul_depth.buy_orders.keys())
        best_ul_ask = min(ul_depth.sell_orders.keys())
        S = get_microprice(ul_depth)

        options_syms = sorted(
            [sym for sym in state.order_depths if sym.startswith("VEV_")],
            key=lambda s: float(s.split("_")[1])
        )

        iv_map = {}
        for sym in options_syms:
            depth = state.order_depths.get(sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            K = float(sym.split("_")[1])
            micro = get_microprice(depth)
            if micro <= 0:
                continue
            iv_map[sym] = find_iv(micro, S, K, TTE)

        xs, ys = [], []
        for sym, iv in iv_map.items():
            K = float(sym.split("_")[1])
            m = math.log(K / S) if S > 0 else 0.0
            if abs(m) <= self.MONEYNESS_FILTER:
                xs.append(m)
                ys.append(iv)

        a, b, c, smile_valid = fit_smile(xs, ys)
        c = max(0.05, min(c, 2.0))

        if state.timestamp % self.LOG_INTERVAL == 0:
            print(f"SMILE,{state.timestamp},{day_idx},{TTE:.6f},{a:.4f},{b:.4f},{c:.4f},{len(xs)}")

        total_delta = 0.0

        for sym in options_syms:
            depth = state.order_depths.get(sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            K = float(sym.split("_")[1])
            pos = state.position.get(sym, 0)
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            spread = best_ask - best_bid
            if spread < 1:
                continue

            m = math.log(K / S) if S > 0 else 0.0
            sigma_fit = max(0.01, a * m * m + b * m + c)
            fair_price, delta_fair = bs_call_and_delta(S, K, TTE, sigma_fit)
            fair_int = int(round(fair_price))

            total_delta += pos * delta_fair

            iv_mkt = iv_map.get(sym)
            if iv_mkt is None:
                continue

            in_filter = abs(m) <= self.MONEYNESS_FILTER
            residual = (iv_mkt - sigma_fit) if in_filter else 0.0

            # ---- Base passive quotes ----
            if spread >= 3:
                my_bid = best_bid + 1
                my_ask = best_ask - 1
            else:
                my_bid = best_bid
                my_ask = best_ask

            # Bid ceiling only: don't bid above BS fair (market above model = expensive to buy)
            # REMOVED: ask floor (it was killing symmetric MM when model > market price)
            my_bid = min(my_bid, fair_int)

            # ---- Residual skewing ----
            scale = min(1.0, abs(residual) / self.MAX_EXPECTED_RESIDUAL) if self.MAX_EXPECTED_RESIDUAL > 0 else 0.0
            res_skew = max(1, int(round(scale * 3)))

            if residual > self.RESIDUAL_THRESHOLD:
                # Expensive: sell bias
                my_bid -= res_skew
                my_ask -= (res_skew - 1)
            elif residual < -self.RESIDUAL_THRESHOLD:
                # Cheap: buy bias (raise ask to avoid selling cheap)
                my_bid += (res_skew - 1)
                my_ask += res_skew

            # ---- Proportional inventory skewing ----
            inv_excess = abs(pos) - self.INV_SKEW_START
            if inv_excess > 0:
                inv_skew = max(1, int(inv_excess / self.INV_SKEW_PER_UNIT)) + 1
                if pos > self.INV_SKEW_START:
                    my_bid -= inv_skew
                    my_ask -= inv_skew
                else:
                    my_bid += inv_skew
                    my_ask += inv_skew

            # Crossing guards
            if my_bid >= my_ask:
                my_bid, my_ask = best_bid, best_ask
            my_bid = min(my_bid, best_ask - 1)
            my_ask = max(my_ask, best_bid + 1)

            # Volume
            max_per_tick = max(self.BASE_VOL, int(self.OPT_LIMIT * self.MAX_TRADE_PCT))
            res_vol = max(self.BASE_VOL, int(scale * int(self.OPT_LIMIT * self.MAX_TRADE_PCT)))
            vol = min(res_vol, max_per_tick)

            buy_cap = max(0, self.OPT_LIMIT - pos)
            sell_cap = max(0, self.OPT_LIMIT + pos)

            safe_buy = min(vol, buy_cap)
            safe_sell = min(vol, sell_cap)

            orders = []
            if safe_buy > 0:
                orders.append(Order(sym, my_bid, safe_buy))
            if safe_sell > 0:
                orders.append(Order(sym, my_ask, -safe_sell))
            if orders:
                result[sym] = orders

        # ---- Delta hedge underlying ----
        ul_pos = state.position.get(self.UL_SYM, 0)
        target_ul = max(-self.UL_LIMIT, min(self.UL_LIMIT, -total_delta))
        offset = ul_pos - target_ul

        ul_buy_cap = max(0, self.UL_LIMIT - ul_pos)
        ul_sell_cap = max(0, self.UL_LIMIT + ul_pos)
        ul_spread = best_ul_ask - best_ul_bid

        emergency_triggered = abs(offset) > self.HEDGE_EMERGENCY
        ul_orders = []

        if emergency_triggered:
            emerg_qty = max(1, int(abs(offset) - self.HEDGE_EMERGENCY))
            hedge_skew_ticks = max(-2, min(2, int(round(offset / self.HEDGE_SKEW_DIV))))
            if offset > 0 and ul_sell_cap > 0:
                q = min(emerg_qty, ul_sell_cap)
                ul_orders.append(Order(self.UL_SYM, best_ul_bid, -q))
                if ul_buy_cap > 0:
                    pb = (best_ul_bid + 1 if ul_spread >= 3 else best_ul_bid) - hedge_skew_ticks
                    pb = min(pb, best_ul_ask - 1)
                    ul_orders.append(Order(self.UL_SYM, pb, min(50, ul_buy_cap)))
            elif offset < 0 and ul_buy_cap > 0:
                q = min(emerg_qty, ul_buy_cap)
                ul_orders.append(Order(self.UL_SYM, best_ul_ask, q))
                if ul_sell_cap > 0:
                    pa = (best_ul_ask - 1 if ul_spread >= 3 else best_ul_ask) - hedge_skew_ticks
                    pa = max(pa, best_ul_bid + 1)
                    ul_orders.append(Order(self.UL_SYM, pa, -min(50, ul_sell_cap)))
        else:
            hedge_skew_ticks = max(-3, min(3, int(round(offset / self.HEDGE_SKEW_DIV))))
            if ul_spread >= 3:
                my_ul_bid = best_ul_bid + 1 - hedge_skew_ticks
                my_ul_ask = best_ul_ask - 1 - hedge_skew_ticks
            else:
                my_ul_bid = best_ul_bid - hedge_skew_ticks
                my_ul_ask = best_ul_ask - hedge_skew_ticks
            my_ul_bid = min(my_ul_bid, best_ul_ask - 1)
            my_ul_ask = max(my_ul_ask, best_ul_bid + 1)
            if my_ul_bid >= my_ul_ask:
                my_ul_bid, my_ul_ask = best_ul_bid, best_ul_ask
            if ul_buy_cap > 0:
                ul_orders.append(Order(self.UL_SYM, my_ul_bid, min(50, ul_buy_cap)))
            if ul_sell_cap > 0:
                ul_orders.append(Order(self.UL_SYM, my_ul_ask, -min(50, ul_sell_cap)))

        if ul_orders:
            buys = [(o.price, o.quantity) for o in ul_orders if o.quantity > 0]
            sells = [(o.price, abs(o.quantity)) for o in ul_orders if o.quantity < 0]
            ul_result = []
            if buys:
                capped = min(sum(q for _, q in buys), ul_buy_cap)
                if capped > 0:
                    ul_result.append(Order(self.UL_SYM, max(p for p, _ in buys), capped))
            if sells:
                capped = min(sum(q for _, q in sells), ul_sell_cap)
                if capped > 0:
                    ul_result.append(Order(self.UL_SYM, min(p for p, _ in sells), -capped))
            if ul_result:
                result[self.UL_SYM] = ul_result

        return result, 0, json.dumps(td)

import math
import statistics
from typing import List, Dict, Any
from datamodel import OrderDepth, TradingState, Order


# =====================================================================
# MATH KERNEL (Black-Scholes & Utilities)
# =====================================================================

def norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call_and_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K), 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
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


def get_mid(depth: OrderDepth) -> float:
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


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
# THE GEMINI VELVET MASTER TRADER - V10 (Passive Sniper)
# =====================================================================

class Trader:
    def __init__(self):
        self.UL_SYM = "VELVETFRUIT_EXTRACT"
        self.OPT_LIMIT = 300
        self.UL_LIMIT = 200

        # DIRETTIVE TECNICHE
        self.HISTORICAL_VOL = 0.25
        self.IV_TOLERANCE = 0.012
        self.EMA_ALPHA = 0.15

        # Limitatori (THROTTLING) per evitare la Whipsaw Death Spiral
        self.OPT_MAX_TRADE = 40  # Massimo 40 lotti per tick sulle opzioni
        self.UL_MAX_TRADE = 50  # Massimo 50 lotti per tick sul sottostante

        self.ema_ul = None

    def run(self, state: TradingState):
        result = {prod: [] for prod in state.order_depths}

        ul_depth = state.order_depths.get(self.UL_SYM)
        if not ul_depth or not ul_depth.buy_orders or not ul_depth.sell_orders:
            return result, 0, ""

        ul_vwap = get_vwap(ul_depth)
        TTE = max(1e-6, (5.0 - (state.timestamp / 1000000.0)) / 252.0)

        # [DIRETTIVA 4] Aggiornamento EMA (Mean Reverting Signal)
        if self.ema_ul is None:
            self.ema_ul = ul_vwap
        else:
            self.ema_ul = self.EMA_ALPHA * ul_vwap + (1 - self.EMA_ALPHA) * self.ema_ul

        options_list = [sym for sym in state.order_depths.keys() if sym.startswith("VEV_")]

        # [DIRETTIVA 1] Calcolo IV Mediana Globale (Focus ATM)
        iv_map = {}
        atm_ivs = []
        for opt_sym in options_list:
            depth = state.order_depths.get(opt_sym)
            if depth and depth.buy_orders and depth.sell_orders:
                mid = get_mid(depth)
                K = float(opt_sym.split('_')[1])
                iv = find_iv(mid, ul_vwap, K, TTE)
                iv_map[opt_sym] = iv
                if abs(K - ul_vwap) <= 150:
                    atm_ivs.append(iv)

        fair_iv = statistics.median(atm_ivs) if atm_ivs else (
            statistics.median(iv_map.values()) if iv_map else self.HISTORICAL_VOL)

        # Calcolo Delta attuale per Hedging
        current_delta = 0.0
        for opt_sym in options_list:
            pos = state.position.get(opt_sym, 0)
            if pos != 0:
                K = float(opt_sym.split('_')[1])
                _, delta = bs_call_and_delta(ul_vwap, K, TTE, iv_map.get(opt_sym, fair_iv))
                current_delta += pos * delta

        # =====================================================================
        # COMPARTMENT 1: OPTIONS LOGIC (Passive RV Skewing)
        # =====================================================================
        for opt_sym in options_list:
            depth = state.order_depths.get(opt_sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            K = float(opt_sym.split('_')[1])
            pos = state.position.get(opt_sym, 0)
            local_iv = iv_map.get(opt_sym, fair_iv)
            fv_fair, delta_fair = bs_call_and_delta(ul_vwap, K, TTE, fair_iv)

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            market_spread = best_ask - best_bid

            # Protezione rigidissima sui Cap per evitare sforamenti
            buy_cap = max(0, self.OPT_LIMIT - pos)
            sell_cap = min(0, -self.OPT_LIMIT - pos)

            # Join the Queue (Spread stretto) o Lead (Spread largo)
            my_bid = best_bid
            my_ask = best_ask
            if market_spread > 2.0:
                my_bid += 1
                my_ask -= 1

            # Centratura sul Fair Value
            my_bid = min(my_bid, int(math.floor(fv_fair - 0.5)))
            my_ask = max(my_ask, int(math.ceil(fv_fair + 0.5)))

            # [DIRETTIVA 1] RV Arbitrage Skewing (SOLO PASSIVO)
            if local_iv > fair_iv + self.IV_TOLERANCE:
                my_bid -= 2  # Cara: Abbasso il bid per non comprarla
                my_ask -= 1  # Voglio venderla
            elif local_iv < fair_iv - self.IV_TOLERANCE:
                my_bid += 1  # Economica: Voglio comprarla
                my_ask += 2  # Alzo l'ask per non venderla

            # Inventory Skewing per protezione
            if pos > 150: my_bid -= 1; my_ask -= 1
            if pos < -150: my_bid += 1; my_ask += 1

            # Sanity Check finale
            if my_bid >= my_ask:
                my_bid = best_bid
                my_ask = best_ask

            # THROTTLE: Limita il volume per non creare onde d'urto sul Delta
            safe_buy = min(buy_cap, self.OPT_MAX_TRADE)
            safe_sell = max(sell_cap, -self.OPT_MAX_TRADE)

            # Evita micro-spread in cui non si coprono le fee
            if market_spread >= 1.0:
                if safe_buy > 0: result[opt_sym].append(Order(opt_sym, my_bid, safe_buy))
                if safe_sell < 0: result[opt_sym].append(Order(opt_sym, my_ask, safe_sell))

        # =====================================================================
        # COMPARTMENT 2: UNDERLYING LOGIC (Smooth Hedging & MR)
        # =====================================================================
        ul_pos = state.position.get(self.UL_SYM, 0)

        # Target del sottostante limitato rigorosamente
        target_ul_pos = max(-self.UL_LIMIT, min(self.UL_LIMIT, -current_delta))
        offset = ul_pos - target_ul_pos

        ul_buy_cap = max(0, self.UL_LIMIT - ul_pos)
        ul_sell_cap = min(0, -self.UL_LIMIT - ul_pos)

        best_ul_bid = max(ul_depth.buy_orders.keys())
        best_ul_ask = min(ul_depth.sell_orders.keys())

        my_ul_bid = best_ul_bid
        my_ul_ask = best_ul_ask
        if best_ul_ask - best_ul_bid > 1.0:
            my_ul_bid += 1
            my_ul_ask -= 1

        # [DIRETTIVA 4] Mean Reverting Skew (Segnale EMA)
        if ul_vwap < self.ema_ul:
            my_ul_ask += 1  # Prezzo basso: alzo Ask per non svendere
        elif ul_vwap > self.ema_ul:
            my_ul_bid -= 1  # Prezzo alto: abbasso Bid per non strapagare

        # Skew di Delta Hedging (Più siamo sbilanciati, più ci spostiamo)
        if offset > 30: my_ul_bid -= 1; my_ul_ask -= 1
        if offset > 100: my_ul_bid -= 1; my_ul_ask -= 1
        if offset < -30: my_ul_bid += 1; my_ul_ask += 1
        if offset < -100: my_ul_bid += 1; my_ul_ask += 1

        if my_ul_bid >= my_ul_ask:
            my_ul_bid, my_ul_ask = best_ul_bid, best_ul_ask

        # Esecuzione Taker d'emergenza SOLO per disallineamenti critici
        if offset > 150 and ul_sell_cap < 0:
            emerge_vol = max(ul_sell_cap, -abs(int(offset)))
            result[self.UL_SYM].append(Order(self.UL_SYM, best_ul_bid, emerge_vol))
        elif offset < -150 and ul_buy_cap > 0:
            emerge_vol = min(ul_buy_cap, abs(int(offset)))
            result[self.UL_SYM].append(Order(self.UL_SYM, best_ul_ask, emerge_vol))
        else:
            # Flusso passivo di routine: limitato a 50 per tick per ridurre i costi
            safe_ul_buy = min(ul_buy_cap, self.UL_MAX_TRADE)
            safe_ul_sell = max(ul_sell_cap, -self.UL_MAX_TRADE)

            if safe_ul_buy > 0: result[self.UL_SYM].append(Order(self.UL_SYM, my_ul_bid, safe_ul_buy))
            if safe_ul_sell < 0: result[self.UL_SYM].append(Order(self.UL_SYM, my_ul_ask, safe_ul_sell))

        return result, 0, ""
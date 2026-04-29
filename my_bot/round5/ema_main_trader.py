import json
import math
import collections
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any


# ============================================================================
# ChipTrader - originally ema_chip_trader.py
# ============================================================================
class ChipTrader:
    def __init__(self):
        self.limit = 10

        # StatArb Parameters
        self.beta_sr = 2
        self.mean_sr = 32350
        self.std_sr = 860

        self.beta_to = 0.5
        self.mean_to = 5860
        self.std_to = 410

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask+ best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = ""

        # 1. Update Signals
        mid_sq = self.get_mid_price(state.order_depths.get("MICROCHIP_SQUARE"))
        mid_rect = self.get_mid_price(state.order_depths.get("MICROCHIP_RECTANGLE"))
        z_sr = 0
        if mid_sq is not None and mid_rect is not None:
            spread_sr = mid_sq + self.beta_sr * mid_rect
            z_sr = (spread_sr - self.mean_sr) / self.std_sr

        mid_tri = self.get_mid_price(state.order_depths.get("MICROCHIP_TRIANGLE"))
        mid_oval = self.get_mid_price(state.order_depths.get("MICROCHIP_OVAL"))
        z_to = 0
        if mid_tri is not None and mid_oval is not None:
            spread_to = mid_tri - self.beta_to * mid_oval
            z_to = (spread_to - self.mean_to) / self.std_to

        # 2. Trade Microchips
        products = ["MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE", "MICROCHIP_OVAL",
                    "MICROCHIP_CIRCLE"]

        for product in products:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid = (best_ask + best_bid) / 2.0

            # Fair Value calculation basato sui segnali StatArb
            fair_val = mid
            if product == "MICROCHIP_SQUARE":
                fair_val = mid - (z_sr * 12)
            elif product == "MICROCHIP_RECTANGLE":
                fair_val = mid - (z_sr * 6)
            elif product == "MICROCHIP_TRIANGLE":
                fair_val = mid - (z_to * 6)
            elif product == "MICROCHIP_OVAL":
                fair_val = mid + (z_to * 6)

            # --- A. TAKER LOGIC (Aggressiva) ---
            # Se il fair value è superiore al prezzo di vendita attuale, compriamo subito
            if fair_val > best_ask:
                buy_qty = min(order_depth.sell_orders[best_ask], self.limit - curr_pos)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    curr_pos += buy_qty

            # Se il fair value è inferiore al prezzo di acquisto attuale, vendiamo subito
            if fair_val < best_bid:
                sell_qty = max(-order_depth.buy_orders[best_bid], -self.limit - curr_pos)
                if sell_qty < 0:
                    orders.append(Order(product, best_bid, sell_qty))
                    curr_pos += sell_qty

            # --- B. MAKER LOGIC (Price Improvement) ---
            # Cerchiamo di essere i primi della coda migliorando il best bid/ask di 1 tick

            if curr_pos < self.limit:
                # Proviamo a metterci sopra il miglior compratore per essere eseguiti per primi
                bid_pr = best_bid + 1
                # Non superiamo mai il nostro fair value (per non comprare in perdita)
                # Sottraiamo un piccolo margine (1) per sicurezza
                bid_pr = min(bid_pr, int(math.floor(fair_val - 1)))

                # Se il prezzo calcolato è valido e non incrocia il book in modo errato
                if bid_pr < best_ask:
                    orders.append(Order(product, bid_pr, self.limit - curr_pos))

            if curr_pos > -self.limit:
                # Proviamo a metterci sotto il miglior venditore
                ask_pr = best_ask - 1
                # Non scendiamo mai sotto il fair value
                ask_pr = max(ask_pr, int(math.ceil(fair_val + 1)))

                if ask_pr > best_bid:
                    orders.append(Order(product, ask_pr, -self.limit - curr_pos))

            result[product] = orders

        return result, conversions, traderData


# ============================================================================
# GalaxyTrader1 - originally ema_galaxy_trader_1.py
# ============================================================================
class GalaxyTrader1:
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


# ============================================================================
# PebblesTrader - originally emaflic_pebbles_trader.py
# ============================================================================
class PebblesTrader:
    """
    Financial Thesis:
    1. Anchor-Based Pricing: Optimized betas and EMA intercept.
    2. Sum Constraint: Strong anchor at 50,000.
    3. Mid-Range Dynamic Edge: XS: 1.2, S: 1.6, M: 1.8, L: 1.8, XL: 2.2.
    4. Adaptive Risk Factor: Skew becomes more aggressive as position nears limits.
    5. Selective Basket Arb: Margin of 2.0.
    """

    CONFIG = {
        "PEBBLES_XS": { "anchor": "UV_VISOR_AMBER",             "beta":  1.4, "edge": 1.2 },
        "PEBBLES_S":  { "anchor": "GALAXY_SOUNDS_BLACK_HOLES",  "beta": -0.8, "edge": 1.6 },
        "PEBBLES_M":  { "anchor": "OXYGEN_SHAKE_MORNING_BREATH","beta": -0.9, "edge": 1.8 },
        "PEBBLES_L":  { "anchor": "TRANSLATOR_GRAPHITE_MIST",   "beta":  0.8, "edge": 1.8 },
        "PEBBLES_XL": { "anchor": "PANEL_2X4",                  "beta":  2.5, "edge": 2.2 },
    }

    LIMIT           = 10
    TARGET_SUM      = 50000.0
    N_TARGETS       = 5
    ALPHA           = 0.01
    RISK_FACTOR     = 1.0
    ARB_MARGIN      = 2.0

    def get_mid(self, depth: OrderDepth):
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def run(self, state: TradingState):
        result = {}

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        data.setdefault("intercepts", {})
        data.setdefault("last_raw_fairs", {})

        raw_fairs  = {}
        best_bids  = {}
        best_asks  = {}
        live_fairs = set()

        for target, cfg in self.CONFIG.items():
            anchor = cfg["anchor"]
            beta   = cfg["beta"]

            depth = state.order_depths.get(target)
            t_mid = self.get_mid(depth)

            anchor_depth = state.order_depths.get(anchor)
            a_mid = self.get_mid(anchor_depth)

            if depth and depth.buy_orders and depth.sell_orders:
                best_bids[target] = max(depth.buy_orders.keys())
                best_asks[target] = min(depth.sell_orders.keys())

            if t_mid is not None and a_mid is not None:
                intercept = t_mid - a_mid * beta
                if target not in data["intercepts"]:
                    data["intercepts"][target] = intercept
                else:
                    data["intercepts"][target] = (
                        (1 - self.ALPHA) * data["intercepts"][target]
                        + self.ALPHA * intercept
                    )
                fv = a_mid * beta + data["intercepts"][target]
                raw_fairs[target] = fv
                data["last_raw_fairs"][target] = fv
                live_fairs.add(target)
            elif target in data["last_raw_fairs"]:
                raw_fairs[target] = data["last_raw_fairs"][target]

        # 1. Selective Basket Arbitrage Check
        basket_buy_opportunity = False
        basket_sell_opportunity = False

        if len(best_asks) == self.N_TARGETS:
            sum_asks = sum(best_asks.values())
            if sum_asks < self.TARGET_SUM - self.ARB_MARGIN:
                basket_buy_opportunity = True

        if len(best_bids) == self.N_TARGETS:
            sum_bids = sum(best_bids.values())
            if sum_bids > self.TARGET_SUM + self.ARB_MARGIN:
                basket_sell_opportunity = True

        # 2. Sum Constraint Adjustment for Fair Values
        if len(raw_fairs) == self.N_TARGETS:
            adjustment = (self.TARGET_SUM - sum(raw_fairs.values())) / self.N_TARGETS
        else:
            adjustment = 0.0

        for target in live_fairs:
            depth = state.order_depths[target]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            fv        = raw_fairs[target] + adjustment
            pos       = state.position.get(target, 0)

            orders = []

            # --- Basket Taker Logic ---
            if basket_buy_opportunity:
                buy_qty = self.LIMIT - pos
                if buy_qty > 0:
                    orders.append(Order(target, best_asks[target], buy_qty))
                    pos += buy_qty

            elif basket_sell_opportunity:
                sell_qty = -self.LIMIT - pos
                if sell_qty < 0:
                    orders.append(Order(target, best_bids[target], sell_qty))
                    pos += sell_qty

            # --- Market Making Logic ---
            buy_limit = self.LIMIT - pos
            sell_limit = -self.LIMIT - pos

            if buy_limit > 0 or sell_limit < 0:
                # Adaptive Risk Factor
                current_risk = self.RISK_FACTOR
                if abs(pos) > 7:
                    current_risk *= 1.5

                skewed_fv = fv - pos * current_risk
                edge = self.CONFIG[target]["edge"]
                my_bid = math.floor(skewed_fv - edge)
                my_ask = math.ceil(skewed_fv + edge)

                my_bid = min(my_bid, best_asks[target] - 1)
                my_ask = max(my_ask, best_bids[target] + 1)

                if buy_limit > 0:
                    orders.append(Order(target, my_bid, buy_limit))
                if sell_limit < 0:
                    orders.append(Order(target, my_ask, sell_limit))

            if orders:
                result[target] = orders

        return result, 0, json.dumps(data)


# ============================================================================
# RoboTrader - originally ema_robo_trader.py
# ============================================================================
class RoboTrader:
    """
    Tesi Finanziaria: Arbitraggio statistico strutturale tra i prodotti ROBOT e i componenti MICROCHIP.
    L'analisi PCA e la regressione lineare multipla (OLS) dimostrano che i prezzi della maggior parte
    dei ROBOT sono pesantemente determinati dai prezzi di MICROCHIP_OVAL e MICROCHIP_SQUARE.
    Essendo i mercati talvolta inefficienti nell'aggiornare simultaneamente i prodotti derivati quando i componenti
    cambiano prezzo, il "residuo" (Prezzo Reale - Fair Value Teorico) tende alla mean reversion.
    Strategia: Market Making direzionale basato sul residuo. Il Fair Value di ogni ROBOT è stimato in real-time
    usando i prezzi dei MICROCHIP e i coefficienti OLS pre-calcolati. Più il residuo è ampio, più aggressivamente
    forniamo liquidità (o prendiamo liquidità) dal lato corretto.
    Rischio principale: Cambiamento strutturale dei coefficienti tra i giorni.
    """

    def __init__(self):
        # Coefficients from OLS: Alpha, b_OVAL, b_SQUARE, Std_Err
        self.models = {
            'ROBOT_VACUUMING': (8883.57, 0.2096, -0.1053, 227.85),
            'ROBOT_MOPPING':   (10467.18, -0.2101, 0.1729, 480.70),
            'ROBOT_DISHES':    (12052.99, -0.2806, 0.0192, 310.83),
            'ROBOT_LAUNDRY':   (6119.02, 0.3793, 0.0442, 306.61),
            'ROBOT_IRONING':   (6538.64, 0.3789, -0.0689, 352.55)
        }
        self.position_limit = 7
        self.spread_margins = {
            'ROBOT_VACUUMING': 2,
            'ROBOT_MOPPING': 2,
            'ROBOT_DISHES': 2,
            'ROBOT_LAUNDRY': 2,
            'ROBOT_IRONING': 2
        }

    def get_mid_price(self, product, state: TradingState):
        if product not in state.order_depths:
            return None
        depth = state.order_depths[product]
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        oval_mid = self.get_mid_price('MICROCHIP_OVAL', state)
        square_mid = self.get_mid_price('MICROCHIP_SQUARE', state)

        if oval_mid is None or square_mid is None:
            return result, conversions, ""

        for product in self.models.keys():
            if product not in state.order_depths:
                continue

            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0
            current_position = state.position.get(product, 0)

            alpha, b_oval, b_square, std_err = self.models[product]

            fair_value = alpha + b_oval * oval_mid + b_square * square_mid
            residual = mid_price - fair_value

            # Se residual < 0, siamo sottovalutati -> COMPRIAMO
            # Se residual > 0, siamo sopravvalutati -> VENDIAMO

            z_score = residual / std_err

            # Target position based on z-score
            # Max position is 10. If z_score is 3.33, we want max pos.
            # This reduces the mean absolute position to not be pinned to the limits.
            target_position = int(-z_score * 3)
            target_position = max(-self.position_limit, min(self.position_limit, target_position))

            # Per evitare di accumulare tutto e rimanere bloccati, piazziamo ordini limit
            # verso il fair value, sbilanciati in base alla target position.

            orders = []

            # Widening spread slightly to reduce unnecessary fills
            my_bid = int(round(fair_value - self.spread_margins[product] - current_position * 0.5))
            my_ask = int(round(fair_value + self.spread_margins[product] - current_position * 0.5))

            buy_vol = target_position - current_position if target_position > current_position else 0
            sell_vol = target_position - current_position if target_position < current_position else 0

            # Se siamo pesantemente sbilanciati (abs(z_score) > 1.5), diventiamo aggressivi
            if z_score < -1.5:
                # Sottovalutato: vogliamo comprare aggressivamente
                if buy_vol > 0:
                    orders.append(Order(product, best_ask, buy_vol)) # prendi liquidita'
            elif z_score > 1.5:
                # Sopravvalutato: vogliamo vendere aggressivamente
                if sell_vol < 0:
                    orders.append(Order(product, best_bid, sell_vol)) # prendi liquidita'
            else:
                # MM normale basato su FV, ma non superare il position limit
                buy_vol_limit = self.position_limit - current_position
                sell_vol_limit = -self.position_limit - current_position

                # Scegliamo di fornire liquidità solo se non superiamo il target position in modo estremo
                # Modifichiamo per quotare solo volumi che ci portano verso il target position o neutrali
                if buy_vol_limit > 0:
                    orders.append(Order(product, min(my_bid, best_bid + 1), buy_vol_limit))
                if sell_vol_limit < 0:
                    orders.append(Order(product, max(my_ask, best_ask - 1), sell_vol_limit))

            if orders:
                result[product] = orders

        return result, conversions, ""


# ============================================================================
# SleepPodTrader - originally ema_sleep_pd_trader.py
# ============================================================================
class SleepPodTrader:
    def __init__(self):
        self.limit = 10
        self.products = ['SLEEP_POD_COTTON', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_NYLON', 'SLEEP_POD_POLYESTER', 'SLEEP_POD_SUEDE']

        self.oracles = {
            'SLEEP_POD_POLYESTER': 'UV_VISOR_AMBER',
            'SLEEP_POD_SUEDE': 'MICROCHIP_SQUARE',
            'SLEEP_POD_NYLON': 'MICROCHIP_TRIANGLE',
            'SLEEP_POD_LAMB_WOOL': 'TRANSLATOR_ECLIPSE_CHARCOAL',
            'SLEEP_POD_COTTON': 'SLEEP_POD_POLYESTER'
        }

        self.params = {
            'SLEEP_POD_POLYESTER': {'alpha': 19140.15, 'beta': -0.9226},
            'SLEEP_POD_SUEDE': {'alpha': 5257.67, 'beta': 0.4516},
            'SLEEP_POD_NYLON': {'alpha': 14314.54, 'beta': -0.4830},
            'SLEEP_POD_LAMB_WOOL': {'alpha': 17729.92, 'beta': -0.7162},
            'SLEEP_POD_COTTON': {'alpha': -729.98 / 0.9638, 'beta': 1 / 0.9638}
        }

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        for product in self.products:
            if product not in state.order_depths: continue

            oracle_name = self.oracles[product]
            if oracle_name not in state.order_depths: continue

            oracle_mid = self.get_mid_price(state.order_depths[oracle_name])
            if oracle_mid is None: continue

            order_depth = state.order_depths[product]
            if not order_depth.sell_orders or not order_depth.buy_orders:
                continue

            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)

            p = self.params[product]
            fair_val = p['alpha'] + p['beta'] * oracle_mid

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            # Use pennying strategy to stay at the extremes and ensure fills in 'worse' mode.
            # This follows the directive to use best_bid + 1 and best_ask - 1.
            bid_price = best_bid + 1
            ask_price = best_ask - 1

            if curr_pos < self.limit:
                orders.append(Order(product, bid_price, self.limit - curr_pos))
            if curr_pos > -self.limit:
                orders.append(Order(product, ask_price, -self.limit - curr_pos))

            result[product] = orders

        return result, conversions, state.traderData


# ============================================================================
# SnackTrader1 - originally ema_snack_trader_1.py
# ============================================================================
class SnackTrader1:
    def __init__(self):
        self.limit = 10
        self.products = [
            'SNACKPACK_CHOCOLATE',
            'SNACKPACK_PISTACHIO',
            'SNACKPACK_RASPBERRY',
            'SNACKPACK_STRAWBERRY',
            'SNACKPACK_VANILLA'
        ]

        # Historical average prices for each flavor (approximate)
        self.avg_prices = {
            'SNACKPACK_CHOCOLATE': 10000.0,
            'SNACKPACK_PISTACHIO': 9500.0,
            'SNACKPACK_RASPBERRY': 10000.0,
            'SNACKPACK_STRAWBERRY': 10500.0,
            'SNACKPACK_VANILLA': 10000.0
        }

        # Relative weights (how much each flavor is of the total cluster sum)
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
                data = {"ema_weights": self.weights}
        else:
            data = {"ema_weights": self.weights}

        current_prices = {}
        for p in self.products:
            mid = self.get_mid_price(state.order_depths.get(p))
            if mid is not None:
                current_prices[p] = mid

        if not current_prices:
            return result, conversions, json.dumps(data)

        cluster_sum = sum(current_prices.values())
        current_rel_weights = {p: current_prices[p] / cluster_sum for p in current_prices}

        # Target position for each product based on its deviation from historical relative weight
        for product in current_prices:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)

            # Update EMA weight
            if product in data["ema_weights"]:
                data["ema_weights"][product] = self.alpha * current_rel_weights[product] + (1 - self.alpha) * data["ema_weights"][product]
            else:
                data["ema_weights"][product] = current_rel_weights[product]

            # Deviation from EMA weight
            # If current weight > EMA weight, flavor is relatively expensive
            # We want a negative position
            # Scaling: a 0.001 (0.1%) weight deviation might justify a full position
            weight_dev = current_rel_weights[product] - data["ema_weights"][product]
            target_pos = -weight_dev * 5000.0 # Scaling factor to tune
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


# ============================================================================
# TranslatorTrader - originally ema_translator_trader.py
# ============================================================================
class TranslatorTrader:
    def __init__(self):
        self.limits = {
            "TRANSLATOR_ECLIPSE_CHARCOAL": 10,
            "TRANSLATOR_VOID_BLUE": 10,
            "TRANSLATOR_ASTRO_BLACK": 10
        }

        # Pair 1: ECLIPSE vs VOID
        self.p1_alpha = 6700
        self.p1_beta = 0.3
        self.p1_std = 315

        # Pair 2: ASTRO vs VOID (Note: ASTRO = alpha - beta * VOID)
        self.p2_alpha = 15500
        self.p2_beta = -0.55
        self.p2_std = 365

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        m_eclipse = self.get_mid_price(state.order_depths.get("TRANSLATOR_ECLIPSE_CHARCOAL"))
        m_void = self.get_mid_price(state.order_depths.get("TRANSLATOR_VOID_BLUE"))
        m_astro = self.get_mid_price(state.order_depths.get("TRANSLATOR_ASTRO_BLACK"))

        if m_void is None:
            return {}, 0, state.traderData

        # 1. Pair 1: ECLIPSE vs VOID
        if m_eclipse is not None:
            spread1 = m_eclipse - (self.p1_alpha + self.p1_beta * m_void)
            z1 = spread1 / self.p1_std
            self.apply_pair_logic(state, "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_VOID_BLUE", z1, result)

        # 2. Pair 2: ASTRO vs VOID
        if m_astro is not None:
            spread2 = m_astro - (self.p2_alpha + self.p2_beta * m_void)
            z2 = spread2 / self.p2_std
            self.apply_pair_logic(state, "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_VOID_BLUE", z2, result)

        return result, conversions, state.traderData

    def apply_pair_logic(self, state, p1, p2, z, result):
        pos1 = state.position.get(p1, 0)
        lim1 = self.limits[p1]

        depth1 = state.order_depths[p1]
        best_ask1, best_bid1 = min(depth1.sell_orders.keys()), max(depth1.buy_orders.keys())

        orders1 = result.get(p1, [])

        # OIB Skew
        oib1 = 0
        if depth1.buy_orders and depth1.sell_orders:
            b_vol = sum(depth1.buy_orders.values())
            s_vol = sum(abs(v) for v in depth1.sell_orders.values())
            oib1 = (b_vol - s_vol) / (b_vol + s_vol)

        entry_z = 1.0 - (oib1 * 0.2)

        # --- NUOVA LOGICA DI PRICING (MAKER) ---
        spread = best_ask1 - best_bid1

        # Prezzo a cui vogliamo COMPRARE (qty > 0)
        if spread > 1:
            my_bid_price = best_bid1 + 1  # Miglioriamo il bid per essere primi
        else:
            my_bid_price = best_bid1  # Spread al minimo, ci accodiamo

        # Prezzo a cui vogliamo VENDERE (qty < 0)
        if spread > 1:
            my_ask_price = best_ask1 - 1  # Miglioriamo l'ask per essere primi
        else:
            my_ask_price = best_ask1  # Spread al minimo, ci accodiamo
        # ---------------------------------------

        if z > entry_z:
            # Dobbiamo VENDERE (qty negativa)
            qty = -lim1 - pos1
            if qty < 0:
                # Usiamo my_ask_price INVECE di best_bid1
                orders1.append(Order(p1, my_ask_price, qty))

        elif z < -entry_z:
            # Dobbiamo COMPRARE (qty positiva)
            qty = lim1 - pos1
            if qty > 0:
                # Usiamo my_bid_price INVECE di best_ask1
                orders1.append(Order(p1, my_bid_price, qty))

        elif abs(z) < 0.2:
            # Chiusura posizione
            if pos1 > 0:
                # Abbiamo comprato, ora dobbiamo vendere per chiudere
                orders1.append(Order(p1, my_ask_price, -pos1))
            if pos1 < 0:
                # Abbiamo venduto, ora dobbiamo comprare per chiudere
                orders1.append(Order(p1, my_bid_price, -pos1))

        if orders1:
            result[p1] = orders1


# ============================================================================
# VRTrader - originally ema_vr_trader.py
# ============================================================================
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

        current_prices = {}
        for p in self.products:
            mid = self.get_mid_price(state.order_depths.get(p))
            if mid is not None:
                current_prices[p] = mid

        if len(current_prices) < len(self.products):
            return {}, 0, state.traderData

        signals = {p: 0.0 for p in self.products}

        # Calculate pairwise differences and Z-scores
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

                # If z > 0, P1 is relatively overvalued compared to P2
                # So we want to sell P1 (negative signal) and buy P2 (positive signal)
                # To prevent extreme trends, we cap the z signal at 3
                z_capped = max(min(z, 3.0), -3.0)

                signals[p1] -= z_capped
                signals[p2] += z_capped

        for product in self.products:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            curr_pos = state.position.get(product, 0)

            mid = current_prices[product]

            # The signal tells us how much we want to hold
            # There are 4 pairs for each product. Max signal sum is 4 * 3 = 12.
            # We scale it to be a target position.
            target_pos = signals[product] * (self.limit / 6.0) # tune scaling
            target_pos = max(min(target_pos, self.limit), -self.limit)

            # Fair value adjustment to attract executions towards target_pos
            # If target_pos > curr_pos, we want to buy, so we increase fair_val
            fair_val = mid + (target_pos - curr_pos) * 1.5

            edge = 1.5

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            # Improvement: place orders at best_bid + 1 and best_ask - 1 to ensure execution in 'worse' match mode
            # while still being constrained by our fair value calculation.
            bid_price = min(int(math.floor(fair_val - edge)), best_bid + 1)
            ask_price = max(int(math.ceil(fair_val + edge)), best_ask - 1)

            if curr_pos < self.limit:
                orders.append(Order(product, bid_price, self.limit - curr_pos))
            if curr_pos > -self.limit:
                orders.append(Order(product, ask_price, -self.limit - curr_pos))

            result[product] = orders

        return result, conversions, json.dumps(data)


# ============================================================================
# MASTER Trader - dispatches to all sub-traders, esclude OXYGEN e PANEL
# ============================================================================
class Trader:
    """
    Master Trader: combina tutti i sub-trader della cartella round5 (escluso OXYGEN e PANEL).
    Ogni sub-trader mantiene la sua logica interna invariata.
    Ogni sub-trader riceve la sua porzione di traderData (incapsulata) per non interferire con gli altri.
    Gli ordini di prodotti uguali vengono concatenati.
    """

    def __init__(self):
        self.subtraders = {
            "chip":       ChipTrader(),
            "galaxy1":    GalaxyTrader1(),
            "pebbles":    PebblesTrader(),
            "robo":       RoboTrader(),
            "sleep_pod":  SleepPodTrader(),
            "snack1":     SnackTrader1(),
            "translator": TranslatorTrader(),
            "vr":         VRTrader(),
        }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            master_data = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(master_data, dict):
                master_data = {}
        except Exception:
            master_data = {}

        original_traderData = state.traderData

        for key, sub in self.subtraders.items():
            try:
                sub_td = master_data.get(key, "")
                if not isinstance(sub_td, str):
                    sub_td = json.dumps(sub_td)
                state.traderData = sub_td

                sub_result, sub_conv, new_sub_td = sub.run(state)

                if sub_result:
                    for product, orders in sub_result.items():
                        if product in result:
                            result[product].extend(orders)
                        else:
                            result[product] = list(orders)

                if isinstance(sub_conv, int):
                    conversions += sub_conv

                master_data[key] = new_sub_td if new_sub_td is not None else ""
            except Exception:
                master_data.setdefault(key, "")
                continue

        state.traderData = original_traderData

        return result, conversions, json.dumps(master_data)

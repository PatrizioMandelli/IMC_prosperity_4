import json
import math
import collections
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any

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
            target_position = int(-z_score * 3)
            target_position = max(-self.position_limit, min(self.position_limit, target_position))

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

                if buy_vol_limit > 0:
                    orders.append(Order(product, min(my_bid, best_bid + 1), buy_vol_limit))
                if sell_vol_limit < 0:
                    orders.append(Order(product, max(my_ask, best_ask - 1), sell_vol_limit))

            if orders:
                result[product] = orders

        return result, conversions, ""

class Trader:
    def __init__(self):
        self.robo_trader = RoboTrader()

    def run(self, state: TradingState):
        return self.robo_trader.run(state)

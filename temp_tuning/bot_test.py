import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List


class Trader:
    def __init__(self):
        # Solo prodotti con half-life corta (mean reversion veloce)
        # Esclusi: BLACK_HOLES (HL~16861) e PLANETARY_RINGS (HL~8455)
        self.products = [
            'GALAXY_SOUNDS_DARK_MATTER',   # HL ~1438
            'GALAXY_SOUNDS_SOLAR_FLAMES',  # HL ~1732
            'GALAXY_SOUNDS_SOLAR_WINDS',   # HL ~3479
        ]

        # Parametri Ornstein-Uhlenbeck stimati
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

        # EMA finestra = half-life (stima coerente con il processo OU)
        self.windows = {p: self.ou_half_life[p] for p in self.products}

        self.emas = {p: None for p in self.products}
        self.position_limit = 10

        # Soglie z-score per scaling della posizione target
        # |z| <= z_enter -> non entrare (vicini alla media)
        # |z| >= z_full  -> posizione massima
        self.z_enter = 0.4
        self.z_full = 1.8

        # Peso del Mu OU rispetto all'EMA per il fair value (anchor)
        # 0.0 = solo EMA, 1.0 = solo Mu
        self.mu_anchor_weight = 0.1

    def _fair_value(self, product, ema):
        return ema * (1.0 - self.mu_anchor_weight) + self.ou_mu[product] * self.mu_anchor_weight

    def _target_position(self, z):
        # z negativo (prezzo sotto media) -> long; z positivo -> short
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

            # Esecuzione: aggressiva quando z e' estremo, passiva altrimenti
            aggressive = abs(z) >= self.z_full * 0.8

            if delta > 0:
                # Comprare
                if aggressive:
                    qty = min(delta, -order_depth.sell_orders[best_ask])
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                else:
                    bid_price = min(best_bid + 1, math.floor(fair_value - sigma * self.z_enter))
                    if bid_price < best_ask:
                        orders.append(Order(product, bid_price, delta))
            elif delta < 0:
                # Vendere
                if aggressive:
                    qty = min(-delta, order_depth.buy_orders[best_bid])
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                else:
                    ask_price = max(best_ask - 1, math.ceil(fair_value + sigma * self.z_enter))
                    if ask_price > best_bid:
                        orders.append(Order(product, ask_price, delta))
            else:
                # Flat target: market making leggero attorno al fair value se gia' a target
                if current_position == 0:
                    bid_price = min(best_bid + 1, math.floor(fair_value - sigma * 0.8))
                    ask_price = max(best_ask - 1, math.ceil(fair_value + sigma * 0.8))
                    if bid_price < best_ask:
                        orders.append(Order(product, bid_price, 3))
                    if ask_price > best_bid:
                        orders.append(Order(product, ask_price, -3))

            result[product] = orders

        traderData = json.dumps(self.emas)
        return result, 0, traderData

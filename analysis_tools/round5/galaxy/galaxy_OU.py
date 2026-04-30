import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Any


class Trader:
    def __init__(self):
        # I nostri 4 prodotti target
        self.products = [
            'GALAXY_SOUNDS_DARK_MATTER',
            'GALAXY_SOUNDS_BLACK_HOLES',
            'GALAXY_SOUNDS_SOLAR_WINDS',
            'GALAXY_SOUNDS_PLANETARY_RINGS'
        ]

        # Gli Half-Life precisi estratti dal tuo modello Ornstein-Uhlenbeck
        self.windows = {
            'GALAXY_SOUNDS_DARK_MATTER': 1438,
            'GALAXY_SOUNDS_BLACK_HOLES': 16861,
            'GALAXY_SOUNDS_SOLAR_WINDS': 3479,
            'GALAXY_SOUNDS_PLANETARY_RINGS': 8455
        }

        # Dizionario per memorizzare l'EMA corrente (pesa zero sulla RAM)
        self.emas = {p: None for p in self.products}

        self.position_limit = 10

        # Spread base leggermente ampliati vista la volatilità (Sigma > 30)
        self.half_spreads = {
            'GALAXY_SOUNDS_DARK_MATTER': 8.0,
            'GALAXY_SOUNDS_BLACK_HOLES': 8.0,
            'GALAXY_SOUNDS_SOLAR_WINDS': 8.0,
            'GALAXY_SOUNDS_PLANETARY_RINGS': 8.0
        }

    def run(self, state: TradingState):
        result = {}

        # --- (OPZIONALE) Ripristino stato EMA se l'engine riavvia l'istanza ---
        # In Prosperity, a volte è utile salvare le variabili nel traderData
        if state.traderData:
            try:
                saved_emas = json.loads(state.traderData)
                for p in self.products:
                    if p in saved_emas:
                        self.emas[p] = saved_emas[p]
            except Exception:
                pass

        for product in self.products:
            if product not in state.order_depths:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Calcolo del Mid Price
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_bid + best_ask) / 2.0
            else:
                continue

            # --- Calcolo EMA (Exponential Moving Average) ---
            window = self.windows[product]
            alpha = 2.0 / (window + 1.0)

            if self.emas[product] is None:
                # Inizializzazione al primo tick
                self.emas[product] = mid_price
            else:
                # Aggiornamento EMA
                self.emas[product] = (mid_price * alpha) + (self.emas[product] * (1.0 - alpha))

            current_ema = self.emas[product]
            half_spread = self.half_spreads[product]

            # Lettura Inventario
            current_position = state.position.get(product, 0)
            max_buy_volume = self.position_limit - current_position
            max_sell_volume = -self.position_limit - current_position

            # =======================================================
            # STRATEGIA 1: MICRO MEAN REVERTING (Con Hard Cap Anti-Taker)
            # Prodotti: DARK_MATTER, BLACK_HOLES, SOLAR_WINDS
            # =======================================================
            if product in ['GALAXY_SOUNDS_DARK_MATTER', 'GALAXY_SOUNDS_BLACK_HOLES', 'GALAXY_SOUNDS_SOLAR_WINDS']:

                # 1. Skew dell'inventario per allontanare i prezzi se siamo troppo esposti
                inventory_skew = current_position * 2.5
                adjusted_fair_value = current_ema - inventory_skew

                # 2. Calcolo del prezzo desiderato (Teorico)
                raw_bid = math.floor(adjusted_fair_value - half_spread)
                raw_ask = math.ceil(adjusted_fair_value + half_spread)

                # 3. L'HARD CAP (Il salvavita)
                # Il nostro Buy non deve mai toccare il best_ask. Al massimo ci mettiamo a best_ask - 1 (pennying).
                # Il nostro Sell non deve mai toccare il best_bid. Al massimo ci mettiamo a best_bid + 1.
                my_bid_price = min(raw_bid, best_ask - 1)
                my_ask_price = max(raw_ask, best_bid + 1)

                # 4. Cut-Loss d'Emergenza (Il "Panic Button")
                # Unica eccezione all'Hard Cap: se stiamo per esplodere contro il limite di 10 posizioni,
                # diventiamo Taker per scaricare l'inventario tossico a mercato.
                if current_position >= 8:
                    my_ask_price = best_bid  # Scarica aggressivamente
                elif current_position <= -8:
                    my_bid_price = best_ask  # Ricompra aggressivamente

                # 5. Piazzamento degli ordini sicuri
                if max_buy_volume > 0:
                    orders.append(Order(product, my_bid_price, max_buy_volume))
                if max_sell_volume < 0:
                    orders.append(Order(product, my_ask_price, max_sell_volume))

            # =======================================================
            # STRATEGIA 2: MACRO TREND FOLLOWING
            # =======================================================
            elif product == 'GALAXY_SOUNDS_PLANETARY_RINGS':
                # Usiamo l'EMA come linea dello spartiacque direzionale
                trend_threshold = 3.0

                if mid_price > current_ema + trend_threshold:  # Trend Rialzista
                    # Taker per coprire short, passivo per il resto della capacità.
                    # Sommati non possono superare max_buy_volume.
                    taker_qty = max(0, -current_position)
                    taker_qty = min(taker_qty, max_buy_volume)
                    passive_qty = max_buy_volume - taker_qty
                    if taker_qty > 0:
                        orders.append(Order(product, best_ask, taker_qty))
                    if passive_qty > 0:
                        orders.append(Order(product, best_bid, passive_qty))

                elif mid_price < current_ema - trend_threshold:  # Trend Ribassista
                    # max_sell_volume è negativo; ragioniamo in valori assoluti.
                    sell_capacity = -max_sell_volume
                    taker_qty = max(0, current_position)
                    taker_qty = min(taker_qty, sell_capacity)
                    passive_qty = sell_capacity - taker_qty
                    if taker_qty > 0:
                        orders.append(Order(product, best_bid, -taker_qty))
                    if passive_qty > 0:
                        orders.append(Order(product, best_ask, -passive_qty))

            result[product] = orders

        # Serializziamo gli EMA per il prossimo tick
        traderData = json.dumps(self.emas)

        return result, 0, traderData
import json
from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # 1. RECUPERO MEMORIA
        trader_data_dict = {}
        if state.traderData:
            try:
                trader_data_dict = json.loads(state.traderData)
            except Exception:
                pass

        history: List[float] = trader_data_dict.get("PEPPER_HISTORY", [])

        for product in state.order_depths:
            if product != "INTARIAN_PEPPER_ROOT":
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Protezione Book Vuoto
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            # 2. SNAPSHOT BOOK
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            # 2. Calcolo dei volumi totali per la riga del VWAP
            total_volume = 0
            total_value = 0

            # Sommiamo il lato Bid (i volumi qui sono già positivi)
            for price, vol in order_depth.buy_orders.items():
                total_volume += vol
                total_value += price * vol

            # Sommiamo il lato Ask (ATTENZIONE: usiamo abs(vol) perché i volumi sono negativi in Prosperity)
            for price, vol in order_depth.sell_orders.items():
                abs_vol = abs(vol)
                total_volume += abs_vol
                total_value += price * abs_vol

            # Riga VWAP
            if total_volume > 0:
                vwap = (total_value / total_volume)

            # 3. STORICO (max 40 tick)
            history.append(mid_price)
            if len(history) > 40:
                history.pop(0)

            # 4. CALCOLO TARGET E TREND
            core_target = 0
            satellite_bias = 0
            trend_adjusted_fair_value = vwap  # Default se non abbiamo abbastanza dati

            if len(history) >= 40:
                # --- TREND CORE: 70 unità ---
                fast_sma = sum(history[-10:]) / 10
                slow_sma = sum(history[-40:]) / 40

                # Calcoliamo la forza e direzione del trend
                trend_strength = fast_sma - slow_sma

                if trend_strength > 0.5:
                    core_target = 70
                elif trend_strength < -0.5:
                    core_target = -70

                # --- SATELLITE SPIKE-UNLOAD (DISATTIVATO TEMPORANEAMENTE) ---
                """
                sma5 = sum(history[-5:]) / 5
                deviation = mid_price - sma5

                # Logica Trend UP
                if core_target > 0:
                    if deviation > 4.0:
                        satellite_bias = -30  # Scarico pesantissimo
                    elif deviation > 2.5:
                        satellite_bias = -15  # Scarico parziale
                    elif deviation < -4.0:
                        satellite_bias = 10   # Buy the Dip

                # Logica Trend DOWN
                elif core_target < 0:
                    if deviation < -4.0:
                        satellite_bias = 30   # Copertura Short
                    elif deviation < -2.5:
                        satellite_bias = 15   # Copertura parziale
                    elif deviation > 4.0:
                        satellite_bias = -10  # Sell the Rip
                """

            # 5. TARGET FINALE (Clipped a 80)
            POSITION_LIMIT = 80
            final_target = max(min(core_target + satellite_bias, POSITION_LIMIT), -POSITION_LIMIT)

            # 6. GESTIONE ORDINI CHIRURGICA
            current_pos = state.position.get(product, 0)
            desired_qty = final_target - current_pos

            # 7. PRICING DINAMICO (Adattivo rispetto al Trend-Adjusted Fair Value)
            if desired_qty > 0:
                # Dobbiamo comprare.
                # Se l'Ask attuale è inferiore al nostro TAFV (il prezzo è destinato a salire
                # più in alto di quanto costa comprare ora), compriamo aggressivamente crossando lo spread.
                if best_ask < trend_adjusted_fair_value:
                    price = best_ask
                else:
                    # Se il prezzo attuale ha già "prezzato" il trend, ci mettiamo passivi sul bid
                    price = best_bid + 1

                orders.append(Order(product, price, desired_qty))

            elif desired_qty < 0:
                # Dobbiamo vendere.
                # Se il Bid attuale è ancora superiore al nostro TAFV (il prezzo è destinato a
                # scendere più in basso di quanto incassiamo ora), vendiamo aggressivamente.
                if best_bid > trend_adjusted_fair_value:
                    price = best_bid
                else:
                    # Altrimenti ci mettiamo passivi sull'ask per raccogliere spread
                    price = best_ask - 1

                orders.append(Order(product, price, desired_qty))

            result[product] = orders

        # 8. SALVATAGGIO MEMORIA
        trader_data_dict["PEPPER_HISTORY"] = history
        return result, conversions, json.dumps(trader_data_dict)
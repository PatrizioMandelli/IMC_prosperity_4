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

            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            # 2. SNAPSHOT BOOK
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            # 3. STORICO (max 40 tick)
            history.append(mid_price)
            if len(history) > 40:
                history.pop(0)

            # 4. CALCOLO TARGET
            core_target = 0
            satellite_bias = 0

            if len(history) >= 40:
                # --- TREND CORE: SMA10 vs SMA40 ---
                # Sfrutta il trend lineare +30% in 3 giorni.
                # Se fast > slow, vogliamo essere long fino a 60 unità.
                fast_sma = sum(history[-10:]) / 10
                slow_sma = sum(history[-40:]) / 40
                if fast_sma > slow_sma + 0.5:
                    core_target = 40
                elif fast_sma < slow_sma - 0.5:
                    core_target = -40

                # --- SATELLITE MEAN REVERSION (Correlazione -0.86, 2600+ istanze) ---
                sma5 = sum(history[-5:]) / 5
                deviation = mid_price - sma5

                # Livello A: Spike confermata (deviation > 4.0) → bias max ±20
                if abs(deviation) > 4.0:
                    satellite_bias = -20 if deviation > 0 else 20
                # Livello B: Pre-allarme (deviation > 2.5, 18% prob. spike imminente) → bias ±10
                elif abs(deviation) > 2.5:
                    satellite_bias = -10 if deviation > 0 else 10
                # Nota: NON usiamo il volume come segnale satellite. In un trend rialzista
                # ask sottile ≠ spike da fadeare, è il trend che prosegue.

            # 5. TARGET FINALE (core + satellite, clipped ai limiti)
            POSITION_LIMIT = 80
            final_target = max(min(core_target + satellite_bias, POSITION_LIMIT), -POSITION_LIMIT)

            # 6. PRICING DINAMICO
            # Default: ordini passivi al top of book (nessuna fee di crossing).
            # Se siamo lontani dal target (>5 unità), ordini aggressivi che attraversano lo spread
            # per garantire il fill e inseguire il trend.
            current_pos = state.position.get(product, 0)
            inventory_offset = current_pos - final_target

            my_bid_price = best_bid      # passivo: aspettiamo fill al miglior bid
            my_ask_price = best_ask      # passivo: aspettiamo fill al miglior ask

            if inventory_offset > 5:
                # Troppo Long rispetto al target: vendiamo aggressivi (hit il bid)
                my_ask_price = best_bid
            elif inventory_offset < -5:
                # Troppo Short rispetto al target: compriamo aggressivi (hit l'ask)
                my_bid_price = best_ask

            # 7. INVIO ORDINI
            max_buy = POSITION_LIMIT - current_pos
            max_sell = -POSITION_LIMIT - current_pos

            if max_buy > 0:
                orders.append(Order(product, my_bid_price, max_buy))
            if max_sell < 0:
                orders.append(Order(product, my_ask_price, max_sell))

            result[product] = orders

        # 8. SALVATAGGIO MEMORIA
        trader_data_dict["PEPPER_HISTORY"] = history
        return result, conversions, json.dumps(trader_data_dict)

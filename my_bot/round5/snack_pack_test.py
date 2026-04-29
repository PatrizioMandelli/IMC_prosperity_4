import json
import math
from typing import Dict, List, Any
from datamodel import Order, Symbol, TradingState

# ─── REGISTRY BASKETS ────────────────────────────────────────────────────────
BASKETS = {
    'SNACK_DUO': {
        'target': 'SNACKPACK_CHOCOLATE',
        'others': ['SNACKPACK_VANILLA'],
        'weights': {'SNACKPACK_VANILLA': -1.0411},
        'lots': {'SNACKPACK_CHOCOLATE': 10, 'SNACKPACK_VANILLA': 10},
        'z_entry': 2.0,
        'z_exit': 0.5,
        'window_size': 100
    },
    'SNACK_TRIO': {
        'target': 'SNACKPACK_PISTACHIO',
        'others': ['SNACKPACK_RASPBERRY', 'SNACKPACK_STRAWBERRY'],
        'weights': {'SNACKPACK_RASPBERRY': -0.9056, 'SNACKPACK_STRAWBERRY': -0.4025},
        'lots': {'SNACKPACK_PISTACHIO': 10, 'SNACKPACK_RASPBERRY': 9, 'SNACKPACK_STRAWBERRY': 4},
        'z_entry': 2.0,
        'z_exit': 0.5,
        'window_size': 100
    }
}


class Trader:
    def __init__(self):
        self.state_data = {}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}

        # 1. Deserializza lo stato precedente
        if state.traderData:
            try:
                self.state_data = json.loads(state.traderData)
            except:
                self.state_data = {}

        # Inizializza lo storico degli spread se non esiste
        if 'spread_history' not in self.state_data:
            self.state_data['spread_history'] = {b_name: [] for b_name in BASKETS.keys()}

        # 2. Itera sui basket configurati
        for b_name, b_cfg in BASKETS.items():
            target = b_cfg['target']
            others = b_cfg['others']

            # Controlla che tutti i prodotti del basket abbiano un orderbook in questo tick
            if target not in state.order_depths or any(o not in state.order_depths for o in others):
                continue

            # Calcola Mid Prices
            mids = {}
            valid_orderbooks = True
            for symbol in [target] + others:
                od = state.order_depths[symbol]
                if len(od.buy_orders) > 0 and len(od.sell_orders) > 0:
                    best_bid = max(od.buy_orders.keys())
                    best_ask = min(od.sell_orders.keys())
                    mids[symbol] = (best_bid + best_ask) / 2.0
                else:
                    valid_orderbooks = False
                    break

            if not valid_orderbooks:
                continue

            # 3. Calcola lo spread corrente: Target - Sum(Weight_i * Other_i)
            synth_price = sum(b_cfg['weights'][o] * mids[o] for o in others)
            current_spread = mids[target] - synth_price

            # Aggiorna la history (finestra mobile)
            history = self.state_data['spread_history'][b_name]
            history.append(current_spread)
            if len(history) > b_cfg['window_size']:
                history.pop(0)

            self.state_data['spread_history'][b_name] = history

            # Se non abbiamo abbastanza dati per calcoli statistici affidabili, salta il trading
            if len(history) < 20:
                continue

            # 4. Calcola Media, Std Dev e Z-Score
            mean = sum(history) / len(history)
            variance = sum((x - mean) ** 2 for x in history) / len(history)
            std = math.sqrt(variance) if variance > 0 else 1.0

            z_score = (current_spread - mean) / std

            # 5. Logica di Trading
            for sym in [target] + others:
                if sym not in orders:
                    orders[sym] = []

            pos_target = state.position.get(target, 0)
            target_lot = b_cfg['lots'][target]

            # ENTRY SHORT SPREAD (Spread troppo alto -> Vendi Target, Compra Others)
            if z_score > b_cfg['z_entry'] and pos_target > -target_lot:
                od_target = state.order_depths[target]
                best_bid_target = max(od_target.buy_orders.keys())
                # Vendi target per raggiungere la size massima short
                orders[target].append(Order(target, best_bid_target, -target_lot - pos_target))

                for o in others:
                    od_other = state.order_depths[o]
                    best_ask_other = min(od_other.sell_orders.keys())
                    other_qty = b_cfg['lots'][o]
                    current_o_pos = state.position.get(o, 0)
                    # Compra gli altri asset per coprire il target
                    orders[o].append(Order(o, best_ask_other, other_qty - current_o_pos))

            # ENTRY LONG SPREAD (Spread troppo basso -> Compra Target, Vendi Others)
            elif z_score < -b_cfg['z_entry'] and pos_target < target_lot:
                od_target = state.order_depths[target]
                best_ask_target = min(od_target.sell_orders.keys())
                # Compra target per raggiungere la size massima long
                orders[target].append(Order(target, best_ask_target, target_lot - pos_target))

                for o in others:
                    od_other = state.order_depths[o]
                    best_bid_other = max(od_other.buy_orders.keys())
                    other_qty = b_cfg['lots'][o]
                    current_o_pos = state.position.get(o, 0)
                    # Vendi gli altri asset per coprire il target
                    orders[o].append(Order(o, best_bid_other, -other_qty - current_o_pos))

            # EXIT (Mean reversion avvenuta)
            elif abs(z_score) < b_cfg['z_exit'] and pos_target != 0:
                if pos_target > 0:  # Chiusura di un Long Spread
                    orders[target].append(Order(target, max(state.order_depths[target].buy_orders.keys()), -pos_target))
                    for o in others:
                        orders[o].append(
                            Order(o, min(state.order_depths[o].sell_orders.keys()), -state.position.get(o, 0)))
                else:  # Chiusura di uno Short Spread
                    orders[target].append(
                        Order(target, min(state.order_depths[target].sell_orders.keys()), -pos_target))
                    for o in others:
                        orders[o].append(
                            Order(o, max(state.order_depths[o].buy_orders.keys()), -state.position.get(o, 0)))

        # 6. Serializza lo stato per il tick successivo
        next_trader_data = json.dumps(self.state_data)

        # Pulisci chiavi vuote dagli ordini
        orders = {k: v for k, v in orders.items() if len(v) > 0}

        conversions = 0
        return orders, conversions, next_trader_data
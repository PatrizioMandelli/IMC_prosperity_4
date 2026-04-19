import json
from typing import List, Dict, Tuple
from datamodel import OrderDepth, TradingState, Order


class Trader:

    PRODUCT = "ASH_COATED_OSMIUM"
    LIMIT = 80
    FAIR_VALUE = 10000
    E_STAR = 2                 # soglia edge per distinguere bad vs good inventory
    MAX_TRADER_DATA = 900      # safety margin vs limite Prosperity (~1000)

    # ────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # ── 1. Load persistent state ────────────────────────────────────
        td = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}

        inventory: list = td.get("inventory", [])
        last_ts: int = td.get("last_ts", -1)

        # ── 2. Apply own_trades ─────────────────────────────────────────
        own_trades = state.own_trades.get(self.PRODUCT, []) if state.own_trades else []
        new_last_ts = last_ts
        for t in own_trades:
            # Skip trade già processati
            if t.timestamp <= last_ts:
                continue
            qty = abs(t.quantity)  # Prosperity: quantity sempre >= 0
            if qty == 0:
                continue

            is_buy = (t.buyer == "SUBMISSION")
            is_sell = (t.seller == "SUBMISSION")

            if is_buy and not is_sell:
                self._apply_fill(inventory, qty, t.price)
            elif is_sell and not is_buy:
                self._apply_fill(inventory, -qty, t.price)
            # trade tra altri bot: ignoro

            if t.timestamp > new_last_ts:
                new_last_ts = t.timestamp

        # ── 3. SYNC: riconcilia inventario con state.position ───────────
        # Se per qualche motivo i fill non sono stati tracciati correttamente
        # (desync da bug, primo run, trade persi), rebuild conservativo.
        actual_pos = state.position.get(self.PRODUCT, 0)
        tracked_pos = sum(q for q, _ in inventory) if inventory else 0

        if tracked_pos != actual_pos:
            if actual_pos == 0:
                inventory = []
            else:
                # Fallback: assume carico medio = FV (conservativo, skew=0)
                inventory = [[actual_pos, self.FAIR_VALUE]]

        # ── 4. Run strategy ─────────────────────────────────────────────
        if self.PRODUCT in state.order_depths:
            raw_orders = self._osmium_strategy(
                state, state.order_depths[self.PRODUCT], inventory
            )
            # Safety: merge duplicati stesso (prezzo, lato) + cap su LIMIT
            merged = self._merge_orders(raw_orders)
            capped = self._cap_orders(merged, actual_pos)
            result[self.PRODUCT] = capped

        # ── 5. Serialize (con truncation safety) ────────────────────────
        trader_data = self._serialize(inventory, new_last_ts)
        return result, conversions, trader_data

    # ────────────────────────────────────────────────────────────────────
    # FIFO inventory update
    # ────────────────────────────────────────────────────────────────────
    def _apply_fill(self, inventory: list, qty: int, price: int):
        """qty con segno: long > 0, short < 0."""
        if qty == 0:
            return
        if not inventory:
            inventory.append([qty, price])
            return

        current_sign = 1 if inventory[0][0] > 0 else -1
        trade_sign = 1 if qty > 0 else -1

        if current_sign == trade_sign:
            inventory.append([qty, price])
            return

        # Segno opposto: consuma FIFO
        remaining = abs(qty)
        while remaining > 0 and inventory:
            head_qty, head_price = inventory[0]
            head_abs = abs(head_qty)
            if head_abs <= remaining:
                remaining -= head_abs
                inventory.pop(0)
            else:
                inventory[0][0] = head_qty - current_sign * remaining
                remaining = 0

        if remaining > 0:
            inventory.append([trade_sign * remaining, price])

    # ────────────────────────────────────────────────────────────────────
    # Strategy
    # ────────────────────────────────────────────────────────────────────
    def _osmium_strategy(
        self, state: TradingState, order_depth: OrderDepth, inventory: list
    ) -> List[Order]:
        orders: List[Order] = []
        current_pos = state.position.get(self.PRODUCT, 0)
        buy_capacity = self.LIMIT - current_pos
        sell_capacity = -self.LIMIT - current_pos

        # ── Arbitrage ───────────────────────────────────────────────────
        if order_depth.sell_orders:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < self.FAIR_VALUE and buy_capacity > 0:
                    take_vol = min(buy_capacity, abs(ask_vol))
                    if take_vol > 0:
                        orders.append(Order(self.PRODUCT, ask_price, take_vol))
                        buy_capacity -= take_vol
                else:
                    break

        if order_depth.buy_orders:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > self.FAIR_VALUE and sell_capacity < 0:
                    take_vol = max(sell_capacity, -bid_vol)
                    if take_vol < 0:
                        orders.append(Order(self.PRODUCT, bid_price, take_vol))
                        sell_capacity -= take_vol
                else:
                    break

        # ── Classifica bad/good ─────────────────────────────────────────
        bad_long_qty = 0
        max_bad_long_price = -10**9
        bad_short_qty = 0
        min_bad_short_price = 10**9

        for qty, price in inventory:
            if qty > 0:
                if self.FAIR_VALUE - price < self.E_STAR:
                    bad_long_qty += qty
                    if price > max_bad_long_price:
                        max_bad_long_price = price
            elif qty < 0:
                if price - self.FAIR_VALUE < self.E_STAR:
                    bad_short_qty += abs(qty)
                    if price < min_bad_short_price:
                        min_bad_short_price = price

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else self.FAIR_VALUE + 5
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else self.FAIR_VALUE - 5

        # ── Dump bad inventory ──────────────────────────────────────────
        if bad_long_qty > 0 and sell_capacity < 0:
            dump_ask_price = max_bad_long_price + 1
            dump_qty = min(bad_long_qty, -sell_capacity)
            if dump_qty > 0:
                orders.append(Order(self.PRODUCT, dump_ask_price, -dump_qty))
                sell_capacity += dump_qty

        if bad_short_qty > 0 and buy_capacity > 0:
            dump_bid_price = min_bad_short_price - 1
            dump_qty = min(bad_short_qty, buy_capacity)
            if dump_qty > 0:
                orders.append(Order(self.PRODUCT, dump_bid_price, dump_qty))
                buy_capacity -= dump_qty

        # ── Normal MM ───────────────────────────────────────────────────
        my_bid = min(self.FAIR_VALUE - 1, best_bid + 1)
        my_ask = max(self.FAIR_VALUE + 1, best_ask - 1)

        if buy_capacity > 0:
            tight_buy_vol = buy_capacity // 2
            deep_buy_vol = buy_capacity - tight_buy_vol
            if tight_buy_vol > 0:
                orders.append(Order(self.PRODUCT, my_bid, tight_buy_vol))
            if deep_buy_vol > 0:
                orders.append(Order(self.PRODUCT, my_bid - 2, deep_buy_vol))

        if sell_capacity < 0:
            tight_sell_vol = sell_capacity // 2
            deep_sell_vol = sell_capacity - tight_sell_vol
            if tight_sell_vol < 0:
                orders.append(Order(self.PRODUCT, my_ask, tight_sell_vol))
            if deep_sell_vol < 0:
                orders.append(Order(self.PRODUCT, my_ask + 2, deep_sell_vol))

        return orders

    # ────────────────────────────────────────────────────────────────────
    # Safety: merge ordini duplicati (stesso prezzo, stesso lato)
    # ────────────────────────────────────────────────────────────────────
    def _merge_orders(self, orders: List[Order]) -> List[Order]:
        merged: Dict[Tuple[int, bool], int] = {}
        for o in orders:
            if o.quantity == 0:
                continue
            key = (o.price, o.quantity > 0)  # (prezzo, is_buy)
            merged[key] = merged.get(key, 0) + o.quantity
        return [Order(self.PRODUCT, price, qty) for (price, _), qty in merged.items() if qty != 0]

    # ────────────────────────────────────────────────────────────────────
    # Safety: cap volumi aggregati per non violare LIMIT
    # Processa ordini in ordine di apparizione (arb prima, dump poi, MM ultimo):
    # gli ordini prioritari vengono preservati, i successivi troncati se serve.
    # ────────────────────────────────────────────────────────────────────
    def _cap_orders(self, orders: List[Order], current_pos: int) -> List[Order]:
        buy_remaining = self.LIMIT - current_pos
        sell_remaining = self.LIMIT + current_pos
        result = []
        for o in orders:
            if o.quantity > 0:  # buy
                take = min(o.quantity, buy_remaining)
                if take > 0:
                    result.append(Order(o.symbol, o.price, take))
                    buy_remaining -= take
            elif o.quantity < 0:  # sell
                take = min(-o.quantity, sell_remaining)
                if take > 0:
                    result.append(Order(o.symbol, o.price, -take))
                    sell_remaining -= take
        return result

    # ────────────────────────────────────────────────────────────────────
    # Safety: serialize con compressione se supera il limite
    # ────────────────────────────────────────────────────────────────────
    def _serialize(self, inventory: list, last_ts: int) -> str:
        data = {"inventory": inventory, "last_ts": last_ts}
        s = json.dumps(data)
        if len(s) <= self.MAX_TRADER_DATA:
            return s

        # Compressione: collassa inventario in una singola entry con avg price
        total_qty = sum(q for q, _ in inventory)
        if total_qty == 0:
            compressed = []
        else:
            weighted = sum(q * p for q, p in inventory)
            avg_price = round(weighted / total_qty)
            compressed = [[total_qty, avg_price]]
        return json.dumps({"inventory": compressed, "last_ts": last_ts})
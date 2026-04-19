import json
from typing import List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    POSITION_LIMIT = 80
    HISTORY_LEN = 20
    CRASH_THRESH = 15
    SPREAD_MULT = 0.6

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        mem = {}
        if state.traderData:
            try:
                mem = json.loads(state.traderData)
            except Exception:
                pass

        history: List[float] = mem.get("h", [])
        regime: str = mem.get("r", "LONG")

        for product in state.order_depths:
            if product != "INTARIAN_PEPPER_ROOT":
                continue

            od = state.order_depths[product]
            orders: List[Order] = []

            if not od.buy_orders or not od.sell_orders:
                continue

            best_ask = min(od.sell_orders.keys())
            best_bid = max(od.buy_orders.keys())
            mid = (best_ask + best_bid) / 2
            spread = best_ask - best_bid

            history.append(mid)
            if len(history) > self.HISTORY_LEN:
                history.pop(0)

            current_pos = state.position.get(product, 0)

            # === REGIME: solo drawdown ===
            if len(history) >= self.HISTORY_LEN:
                if max(history) - mid > self.CRASH_THRESH:
                    regime = "SHORT"
                elif mid - min(history) > self.CRASH_THRESH:
                    regime = "LONG"

            core = self.POSITION_LIMIT if regime == "LONG" else -self.POSITION_LIMIT

            # === SATELLITE: solo se edge > costi ===
            spike_bias = 0
            dev = 0.0
            if len(history) >= 5:
                sma5 = sum(history[-5:]) / 5
                dev = mid - sma5
                breakeven = spread * self.SPREAD_MULT

                if abs(dev) > breakeven:
                    direction = -1 if dev > 0 else 1
                    magnitude = min(30, int(abs(dev) / breakeven) * 10)
                    # Solo TP (scarica nella direzione opposta al core)
                    if direction != (1 if regime == "LONG" else -1):
                        spike_bias = direction * magnitude

            final_target = max(min(core + spike_bias, self.POSITION_LIMIT), -self.POSITION_LIMIT)
            desired_qty = final_target - current_pos

            if desired_qty == 0:
                result[product] = orders
                continue

            # === PRICING ===
            if desired_qty > 0:
                price = best_ask if abs(dev) > spread * self.SPREAD_MULT else best_bid + max(1, spread // 3)
                orders.append(Order(product, price, desired_qty))
            else:
                price = best_bid if abs(dev) > spread * self.SPREAD_MULT else best_ask - max(1, spread // 3)
                orders.append(Order(product, price, desired_qty))

            result[product] = orders

        mem["h"] = history
        mem["r"] = regime
        return result, conversions, json.dumps(mem)
import json
from typing import List

from datamodel import OrderDepth, Order, TradingState


class Trader:
    """Regime-following trader for INTARIAN_PEPPER_ROOT.

    Holds full long/short core based on drawdown regime and trims it back
    with a satellite trade only when the SMA deviation pays for the spread.
    """

    PRODUCT = "INTARIAN_PEPPER_ROOT"
    POSITION_LIMIT = 80
    HISTORY_LEN = 20
    CRASH_THRESH = 15
    SPREAD_MULT = 0.6

    def run(self, state: TradingState):
        result = {}

        mem = {}
        if state.traderData:
            try:
                mem = json.loads(state.traderData)
            except Exception:
                pass

        history: List[float] = mem.get("h", [])
        regime: str = mem.get("r", "LONG")

        for product in state.order_depths:
            if product != self.PRODUCT:
                continue

            od = state.order_depths[product]
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

            # Flip regime only on a sustained drawdown from the recent extreme.
            if len(history) >= self.HISTORY_LEN:
                if max(history) - mid > self.CRASH_THRESH:
                    regime = "SHORT"
                elif mid - min(history) > self.CRASH_THRESH:
                    regime = "LONG"

            core = self.POSITION_LIMIT if regime == "LONG" else -self.POSITION_LIMIT

            # Satellite: only trade if SMA-based edge clears the spread cost.
            spike_bias = 0
            dev = 0.0
            if len(history) >= 5:
                sma5 = sum(history[-5:]) / 5
                dev = mid - sma5
                breakeven = spread * self.SPREAD_MULT
                if abs(dev) > breakeven:
                    direction = -1 if dev > 0 else 1
                    magnitude = min(30, int(abs(dev) / breakeven) * 10)
                    # Only allow the satellite to take profit against the core.
                    if direction != (1 if regime == "LONG" else -1):
                        spike_bias = direction * magnitude

            final_target = max(
                min(core + spike_bias, self.POSITION_LIMIT),
                -self.POSITION_LIMIT,
            )
            desired_qty = final_target - current_pos

            orders: List[Order] = []
            if desired_qty != 0:
                if desired_qty > 0:
                    if abs(dev) > spread * self.SPREAD_MULT:
                        price = best_ask
                    else:
                        price = best_bid + max(1, spread // 3)
                else:
                    if abs(dev) > spread * self.SPREAD_MULT:
                        price = best_bid
                    else:
                        price = best_ask - max(1, spread // 3)
                orders.append(Order(product, price, desired_qty))

            result[product] = orders

        mem["h"] = history
        mem["r"] = regime
        return result, 0, json.dumps(mem)

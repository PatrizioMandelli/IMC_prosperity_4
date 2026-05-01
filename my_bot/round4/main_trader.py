import json
from typing import Dict, List, Optional

from datamodel import Order, TradingState


# Regime EMA: half-life ~7000 ticks.
ALPHA_REGIME = 0.00008

# Seeds: data-mined averages from round 4.
SEEDS: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 5250.098,
    "HYDROGEL_PACK":       9990.807,
    "VEV_4000": 1250.110,
    "VEV_4500":  750.110,
    "VEV_5000":  255.022,
    "VEV_5100":  166.805,
    "VEV_5200":   95.549,
    "VEV_5300":   46.760,
    "VEV_5400":   15.952,
    "VEV_5500":    6.641,
    "VEV_6000":    0.5,
    "VEV_6500":    0.5,
}

LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 200, "HYDROGEL_PACK": 200,
    "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300,
    "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300,
    "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}

# Base edge (spread/2 + 0.5).
EDGES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT":  3.0,
    "HYDROGEL_PACK":        8.4,
    "VEV_4000":            11.0,
    "VEV_4500":             8.5,
    "VEV_5000":             3.5,
    "VEV_5100":             2.6,
    "VEV_5200":             1.9,
    "VEV_5300":             1.5,
    "VEV_5400":             1.2,
    "VEV_5500":             1.1,
    "VEV_6000":             1.0,
    "VEV_6500":             1.0,
}

# Edge floor: never drop below spread/2 (keeps the cross profitable).
EDGES_FLOOR: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT":  2.5,
    "HYDROGEL_PACK":        7.9,
    "VEV_4000":            10.4,
    "VEV_4500":             7.9,
    "VEV_5000":             3.0,
    "VEV_5100":             2.2,
    "VEV_5200":             1.5,
    "VEV_5300":             1.1,
    "VEV_5400":             0.7,
    "VEV_5500":             0.6,
    "VEV_6000":             0.5,
    "VEV_6500":             0.5,
}

# How much to shift the edge in the direction of an informed-flow signal.
# Mark 14 has been shown to lead price on these symbols.
PUSH_MAP: Dict[str, float] = {
    "VEV_4000":      8.0,
    "HYDROGEL_PACK": 5.0,
}

TAKER_MAX = 20


def _vwap(depth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid_px = max(depth.buy_orders)
    ask_px = min(depth.sell_orders)
    bid_vol = depth.buy_orders[bid_px]
    ask_vol = abs(depth.sell_orders[ask_px])
    tot = bid_vol + ask_vol
    if tot == 0:
        return (bid_px + ask_px) / 2.0
    return (bid_px * bid_vol + ask_px * ask_vol) / tot


class Trader:
    """Mean-reversion taker with a Mark 14 / Mark 38 informed-flow edge tilt."""

    def run(self, state: TradingState):
        out: Dict[str, List[Order]] = {}

        mem: dict = {}
        if state.traderData:
            try:
                mem = json.loads(state.traderData)
            except Exception:
                pass

        # Per-tick alpha from observed market trades. Positive -> long bias.
        tick_alphas: Dict[str, float] = {}
        for product, trades in state.market_trades.items():
            if product not in PUSH_MAP:
                continue
            push = PUSH_MAP[product]
            signal = 0.0
            for trade in trades:
                b = trade.buyer or ""
                s = trade.seller or ""
                if b == "Mark 14":
                    signal += push  # informed long
                elif s == "Mark 14":
                    signal -= push  # informed short
                elif b == "Mark 38":
                    signal -= push  # noise long: fade
                elif s == "Mark 38":
                    signal += push  # noise short: fade
            if signal != 0.0:
                tick_alphas[product] = signal

        for sym, seed in SEEDS.items():
            depth = state.order_depths.get(sym)
            if not depth:
                continue

            vwap = _vwap(depth)
            if vwap is None:
                continue

            fair = mem.get(sym, seed)
            fair = (1 - ALPHA_REGIME) * fair + ALPHA_REGIME * vwap
            mem[sym] = fair

            pos = state.position.get(sym, 0)
            limit = LIMITS[sym]
            edge = EDGES[sym]
            floor = EDGES_FLOOR[sym]
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)

            # Floor protects against paying more than half the spread.
            flow = tick_alphas.get(sym, 0.0)
            buy_edge = max(floor, edge - flow)
            sell_edge = max(floor, edge + flow)

            orders: List[Order] = []

            if best_ask < fair - buy_edge:
                qty = min(-depth.sell_orders[best_ask], TAKER_MAX, limit - pos)
                if qty > 0:
                    orders.append(Order(sym, best_ask, qty))

            elif best_bid > fair + sell_edge:
                qty = min(depth.buy_orders[best_bid], TAKER_MAX, pos + limit)
                if qty > 0:
                    orders.append(Order(sym, best_bid, -qty))

            if orders:
                out[sym] = orders

        return out, 0, json.dumps(mem)

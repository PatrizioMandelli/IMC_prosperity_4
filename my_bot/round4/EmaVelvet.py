#643K
"""
Pure Mean Reversion v7.2 — Fixed Prior + Regime EMA
=====================================================
Fix del pattern 1 (inventory accumulation):

CAUSA: EMA con α=0.001 inseguiva il prezzo verso il basso durante le
deviazioni, abbassando il fair value e impedendo il segnale di vendita.
→ Il bot comprava ma non rivendeva mai.

SOLUZIONE: EMA ultra-lenta (α=0.00005, HL≈14k ticks).
Si muove solo su shift strutturali tra round, non durante oscillazioni
intraday → fair value stabile → segnali buy/sell simmetrici → inventory
oscilla intorno a zero invece di accumularsi.
"""
import json
from typing import Dict, List, Optional
from datamodel import Order, TradingState


# ── EMA regime (ultra-lenta) ──────────────────────────────────────────────
# HL ≈ ln(2)/0.00005 ≈ 13863 ticks ≈ 1 giorno
ALPHA_REGIME = 0.00005

# ── Seed (media storica rounds 3-4) ───────────────────────────────────────
SEEDS: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 5251.0,
    "HYDROGEL_PACK":       9990.0,
    "VEV_4000": 1251.0,
    "VEV_4500":  751.0,
    "VEV_5000":  256.0,
    "VEV_5100":  167.0,
    "VEV_5200":   95.0,
    "VEV_5300":   46.0,
    "VEV_5400":   15.0,
    "VEV_5500":    6.3,

}

# ── Position limits ───────────────────────────────────────────────────────
LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK":       200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300
}

# ── Edge (min seashells di vantaggio per incrociare lo spread) ────────────
# Regola: edge >= spread/2 + 0.5
EDGES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT":  3.0,   # spread≈5
    "HYDROGEL_PACK":        8.4,   # spread≈16
    "VEV_4000":            11.0,   # spread≈21, deep ITM illiquido
    "VEV_4500":             8.5,   # spread≈16, deep ITM
    "VEV_5000":             3.5,   # spread≈6
    "VEV_5100":             2.6,   # spread≈4
    "VEV_5200":             1.9,   # spread≈3
    "VEV_5300":             1.5,   # spread≈2
    "VEV_5400":             1.2,   # spread≈1.4
    "VEV_5500":             1.1,   # spread≈1.1
    "VEV_6000":             1,   # spread≈1
    "VEV_6500":             1,   # spread≈1
}

TAKER_MAX = 20


def _vwap(depth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid_px  = max(depth.buy_orders)
    ask_px  = min(depth.sell_orders)
    bid_vol = depth.buy_orders[bid_px]
    ask_vol = abs(depth.sell_orders[ask_px])
    tot     = bid_vol + ask_vol
    if tot == 0:
        return (bid_px + ask_px) / 2.0
    return (bid_px * bid_vol + ask_px * ask_vol) / tot


class Trader:

    def run(self, state: TradingState):
        out: Dict[str, List[Order]] = {}

        mem: dict = {}
        if state.traderData:
            try:
                mem = json.loads(state.traderData)
            except Exception:
                pass

        for sym, seed in SEEDS.items():
            depth = state.order_depths.get(sym)
            if not depth:
                continue

            vwap = _vwap(depth)
            if vwap is None:
                continue

            # Aggiorna regime EMA (ultra-lenta)
            fair = mem.get(sym, seed)
            fair = (1 - ALPHA_REGIME) * fair + ALPHA_REGIME * vwap
            mem[sym] = fair

            pos      = state.position.get(sym, 0)
            limit    = LIMITS[sym]
            edge     = EDGES[sym]
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            orders: List[Order] = []

            # Taker buy: ask sotto il fair → prezzo depresso, compra
            if best_ask < fair - edge:
                qty = min(-depth.sell_orders[best_ask], TAKER_MAX, limit - pos)
                if qty > 0:
                    orders.append(Order(sym, best_ask, qty))

            # Taker sell: bid sopra il fair → prezzo gonfiato, vendi
            elif best_bid > fair + edge:
                qty = min(depth.buy_orders[best_bid], TAKER_MAX, pos + limit)
                if qty > 0:
                    orders.append(Order(sym, best_bid, -qty))

            if orders:
                out[sym] = orders

        return out, 0, json.dumps(mem)
import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional


# ── RLS Model ─────────────────────────────────────────────────────────────────

class RLSModel:
    """
    Recursive Least Squares with exponential forgetting.
    Fits: y = theta @ x  where x = [1, mc1, mc2, mc3, mc4, mc5]
    Updates online every tick — no hardcoded coefficients.
    """

    def __init__(self, n_features: int = 6, lam: float = 0.997,
                 init_var: float = 1e4, alpha_res: float = 0.01):
        self.lam       = lam
        self.alpha_res = alpha_res
        self.n         = n_features
        self.theta     = [0.0] * n_features
        self.P         = [[init_var if i == j else 0.0
                           for j in range(n_features)]
                          for i in range(n_features)]
        self.ema_res2: Optional[float] = None
        self.n_obs     = 0                       # warm-up counter

    # ── serialization (no numpy in traderData) ──

    def to_dict(self) -> dict:
        return {
            "theta":    self.theta,
            "P":        self.P,
            "ema_res2": self.ema_res2,
            "n_obs":    self.n_obs,
        }

    def from_dict(self, d: dict):
        self.theta    = d["theta"]
        self.P        = d["P"]
        self.ema_res2 = d.get("ema_res2")
        self.n_obs    = d.get("n_obs", 0)

    # ── math helpers (pure Python, no numpy) ──

    def _mv(self, M, v):
        """Matrix-vector product."""
        return [sum(M[i][j] * v[j] for j in range(self.n)) for i in range(self.n)]

    def _vv(self, a, b):
        """Dot product."""
        return sum(a[i] * b[i] for i in range(self.n))

    def _outer_sub(self, g, Px):
        """P -= outer(g, Px), then scale by 1/lam."""
        return [[(self.P[i][j] - g[i] * Px[j]) / self.lam
                 for j in range(self.n)]
                for i in range(self.n)]

    # ── update / predict ──

    def update(self, x: list, y: float):
        Px     = self._mv(self.P, x)
        denom  = self.lam + self._vv(x, Px)
        gain   = [px / denom for px in Px]

        pred   = self._vv(self.theta, x)
        resid  = y - pred

        self.theta  = [self.theta[i] + gain[i] * resid for i in range(self.n)]
        self.P      = self._outer_sub(gain, Px)

        r2 = resid ** 2
        self.ema_res2 = (r2 if self.ema_res2 is None
                         else self.alpha_res * r2 + (1 - self.alpha_res) * self.ema_res2)
        self.n_obs += 1

        return pred, math.sqrt(self.ema_res2)

    def predict(self, x: list):
        pred = self._vv(self.theta, x)
        std  = math.sqrt(self.ema_res2) if self.ema_res2 else None
        return pred, std

    @property
    def is_warm(self) -> bool:
        return self.n_obs >= 200


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:
    """
    TG07 HYBRID — RLS edition.

    RLS group  : SPACE_GRAY, VOID_BLUE        (was hardcoded OLS)
    EMA group  : ASTRO_BLACK, ECLIPSE_CHARCOAL, GRAPHITE_MIST
    """

    MICROCHIPS = [
        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL",
        "MICROCHIP_RECTANGLE", "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE",
    ]

    RLS_PRODUCTS = ["TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE"]
    EMA_PRODUCTS = ["TRANSLATOR_ASTRO_BLACK",
                    "TRANSLATOR_ECLIPSE_CHARCOAL",
                    "TRANSLATOR_GRAPHITE_MIST"]
    ALL_TRANSLATORS = RLS_PRODUCTS + EMA_PRODUCTS

    # ── hyperparameters ──
    POS_LIMIT      = 10

    # RLS
    RLS_LAM        = 0.997    # forgetting factor
    RLS_INIT_VAR   = 1e4      # initial P diagonal
    RLS_ALPHA_RES  = 0.01     # EMA speed for std_err
    RLS_MARGIN     = 2        # market-making spread half-width
    RLS_Z_SIGNAL   = 1.0      # z-threshold to directional trade
    RLS_Z_SCALE    = 4        # target = -z * scale (clipped to ±POS_LIMIT)

    # EMA
    EMA_ALPHA      = 0.01
    EMA_ENTRY_Z    = 1.5
    EMA_EXIT_Z     = 0.2

    # ── init ──────────────────────────────────────────────────────

    def __init__(self):
        self.rls: Dict[str, RLSModel] = {
            p: RLSModel(n_features=6,
                        lam=self.RLS_LAM,
                        init_var=self.RLS_INIT_VAR,
                        alpha_res=self.RLS_ALPHA_RES)
            for p in self.RLS_PRODUCTS
        }

    # ── helpers ───────────────────────────────────────────────────

    def get_mid(self, product: str, state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    def _build_feature(self, mc_mids: dict) -> list:
        return [1.0] + [mc_mids[mc] for mc in self.MICROCHIPS]

    # ── RLS strategy ──────────────────────────────────────────────

    def _run_rls(self, state: TradingState, result: dict,
                 mc_mids: dict, data: dict):
        rls_state = data.get("rls", {})

        # Restore persisted model state
        for p, model in self.rls.items():
            if p in rls_state:
                model.from_dict(rls_state[p])

        x = self._build_feature(mc_mids)

        for p, model in self.rls.items():
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            best_bid  = max(depth.buy_orders)
            best_ask  = min(depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2.0
            pos       = state.position.get(p, 0)

            # Always update the model (learns even when not trading)
            fair_value, std_err = model.update(x, mid_price)

            # Skip trading during warm-up
            if not model.is_warm:
                continue

            z_score = (mid_price - fair_value) / std_err
            orders  = []

            if z_score < -self.RLS_Z_SIGNAL:
                # Underpriced → buy
                target = int(min(self.POS_LIMIT,
                                 max(-self.POS_LIMIT, -z_score * self.RLS_Z_SCALE)))
                diff   = target - pos
                if diff > 0:
                    orders.append(Order(p, best_ask, diff))
            elif z_score > self.RLS_Z_SIGNAL:
                # Overpriced → sell
                target = int(min(self.POS_LIMIT,
                                 max(-self.POS_LIMIT, -z_score * self.RLS_Z_SCALE)))
                diff   = target - pos
                if diff < 0:
                    orders.append(Order(p, best_bid, diff))
            else:
                # Market-make around fair value with inventory skew
                skew   = pos * 0.5
                my_bid = int(round(fair_value - self.RLS_MARGIN - skew))
                my_ask = int(round(fair_value + self.RLS_MARGIN - skew))
                room_buy  = self.POS_LIMIT - pos
                room_sell = -self.POS_LIMIT - pos
                if room_buy > 0:
                    orders.append(Order(p, min(my_bid, best_bid + 1), room_buy))
                if room_sell < 0:
                    orders.append(Order(p, max(my_ask, best_ask - 1), room_sell))

            if orders:
                result[p] = orders

        # Persist model state
        data["rls"] = {p: m.to_dict() for p, m in self.rls.items()}

    # ── EMA strategy ──────────────────────────────────────────────

    def _run_ema(self, state: TradingState, result: dict, data: dict):
        ema_state = data.get("ema", {})
        m2_state  = data.get("m2", {})

        # basket_mean over ALL 5 translators (normalisation anchor)
        all_mids = {p: m for p in self.ALL_TRANSLATORS
                    if (m := self.get_mid(p, state)) is not None}

        if len(all_mids) < 3:
            return

        basket_mean = sum(all_mids.values()) / len(all_mids)

        signals: Dict[str, float] = {}
        stds:    Dict[str, float] = {}

        for p, mid in all_mids.items():
            rel    = mid - basket_mean
            ema    = ema_state.get(p, rel)
            new_ema = self.EMA_ALPHA * rel + (1 - self.EMA_ALPHA) * ema
            ema_state[p] = new_ema

            diff   = rel - new_ema
            m2     = m2_state.get(p, 70.0 ** 2)
            new_m2 = self.EMA_ALPHA * diff**2 + (1 - self.EMA_ALPHA) * m2
            m2_state[p]  = new_m2

            signals[p] = diff
            stds[p]    = math.sqrt(new_m2)

        for p in self.EMA_PRODUCTS:
            if p not in signals:
                continue

            z     = signals[p] / stds[p] if stds[p] > 0 else 0.0
            pos   = state.position.get(p, 0)
            depth = state.order_depths.get(p)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            best_ask = min(depth.sell_orders)
            best_bid = max(depth.buy_orders)
            orders   = []

            if z < -self.EMA_ENTRY_Z:
                qty = self.POS_LIMIT - pos
                if qty > 0:
                    orders.append(Order(p, min(best_bid + 1, best_ask - 1), qty))
            elif z > self.EMA_ENTRY_Z:
                qty = -self.POS_LIMIT - pos
                if qty < 0:
                    orders.append(Order(p, max(best_ask - 1, best_bid + 1), qty))
            elif abs(z) < self.EMA_EXIT_Z:
                if pos > 0:
                    orders.append(Order(p, max(best_ask - 1, best_bid + 1), -pos))
                elif pos < 0:
                    orders.append(Order(p, min(best_bid + 1, best_ask - 1), -pos))

            if orders:
                result[p] = orders

        data["ema"] = ema_state
        data["m2"]  = m2_state

    # ── Main ──────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: dict = {}
        data:   dict = {}

        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        # RLS block — requires all 5 MICROCHIP mids
        mc_mids = {mc: self.get_mid(mc, state) for mc in self.MICROCHIPS}
        if None not in mc_mids.values():
            self._run_rls(state, result, mc_mids, data)

        # EMA block
        self._run_ema(state, result, data)

        return result, 0, json.dumps(data)
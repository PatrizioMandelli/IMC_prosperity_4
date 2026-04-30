import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class RoboPairsTrader:
    """
    Statistical arbitrage on ROBOT_* products using a cointegrating basket
    of the other robots + microchips.

    Each robot R has fair value:
        FV(R) = alpha + sum_i beta_i * mid_i

    z = (mid_R - FV(R)) / std(spread)

    Target position is a linear function of -z (mean reversion):
        target = clamp( -z * SCALE, -limit, +limit )

    z << 0  -> robot is cheap relative to basket -> long
    z >> 0  -> robot is rich relative to basket  -> short

    Execution has two regimes:
      |z| >= Z_TAKE : cross the book to reach target, rest the remainder passively
      |z| <  Z_TAKE : two-sided market making around FV with inventory skew
                       (quotes shift down when long, up when short — encourages
                        natural mean reversion of inventory; mild signals shrink
                        the side that opposes the signal)
    """

    POSITION_LIMIT = 10

    # Basket-derived models (OLS on 3 days, all stationary at 5%)
    MODELS = {
        'ROBOT_VACUUMING': {
            'alpha': 9805.9465,
            'betas': {
                'ROBOT_MOPPING': -0.1545,
                'ROBOT_DISHES':   0.0903,
                'ROBOT_LAUNDRY':  0.1119,
                'ROBOT_IRONING': -0.1649,
                'MICROCHIP_OVAL': 0.2225,
                'MICROCHIP_SQUARE': -0.0966,
            },
            'std': 205.94,
            'half_life': 579,
        },
        'ROBOT_MOPPING': {
            'alpha': 27185.8246,
            'betas': {
                'ROBOT_VACUUMING': -0.3726,
                'ROBOT_DISHES':   -0.5293,
                'ROBOT_LAUNDRY':  -0.5493,
                'ROBOT_IRONING':  -0.5609,
                'MICROCHIP_OVAL':  0.1404,
                'MICROCHIP_SQUARE': 0.1295,
            },
            'std': 319.84,
            'half_life': 484,
        },
        'ROBOT_DISHES': {
            'alpha': 15573.1830,
            'betas': {
                'ROBOT_VACUUMING':  0.1399,
                'ROBOT_MOPPING':   -0.3401,
                'ROBOT_LAUNDRY':   -0.2639,
                'ROBOT_IRONING':    0.0629,
                'MICROCHIP_OVAL':  -0.3051,
                'MICROCHIP_SQUARE': 0.1087,
            },
            'std': 256.39,
            'half_life': 259,
        },
        'ROBOT_LAUNDRY': {
            'alpha': 13052.7071,
            'betas': {
                'ROBOT_VACUUMING':  0.1806,
                'ROBOT_MOPPING':   -0.3678,
                'ROBOT_DISHES':    -0.2749,
                'ROBOT_IRONING':   -0.2103,
                'MICROCHIP_OVAL':   0.2668,
                'MICROCHIP_SQUARE': 0.1176,
            },
            'std': 261.71,
            'half_life': 643,
        },
        'ROBOT_IRONING': {
            'alpha': 14563.1917,
            'betas': {
                'ROBOT_VACUUMING': -0.3145,
                'ROBOT_MOPPING':   -0.4436,
                'ROBOT_DISHES':     0.0774,
                'ROBOT_LAUNDRY':   -0.2485,
                'MICROCHIP_OVAL':   0.4676,
                'MICROCHIP_SQUARE': -0.0158,
            },
            'std': 284.45,
            'half_life': 616,
        },
    }

    # |z| at which target position hits the limit
    Z_FULL = 0.3
    # |z| above which we cross the spread aggressively
    Z_TAKE = 1.0
    # |z| below which we stay flat (avoid noise)
    Z_DEAD = 0.15

    # Half-spread (price units) around FV for two-sided market making
    SPREAD_MARGINS = {
        'ROBOT_VACUUMING': 2,
        'ROBOT_MOPPING':   2,
        'ROBOT_DISHES':    2,
        'ROBOT_LAUNDRY':   2,
        'ROBOT_IRONING':   2,
    }
    # Quote shift per unit of inventory (price units per unit of position)
    POSITION_SKEW = 0.5

    def __init__(self):
        self.products = list(self.MODELS.keys())

    @staticmethod
    def _mid(depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid = max(depth.buy_orders.keys())
        ask = min(depth.sell_orders.keys())
        return bid, ask, (bid + ask) / 2.0

    def _target_position(self, z: float) -> int:
        if abs(z) < self.Z_DEAD:
            return 0
        sign = -1.0 if z > 0 else 1.0
        scale = min(1.0, (abs(z) - self.Z_DEAD) / max(self.Z_FULL - self.Z_DEAD, 1e-9))
        return int(round(sign * scale * self.POSITION_LIMIT))

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # Mid prices for all referenced components
        mids: Dict[str, float] = {}
        bidask: Dict[str, tuple] = {}
        for prod, depth in state.order_depths.items():
            m = self._mid(depth)
            if m is not None:
                bidask[prod] = (m[0], m[1])
                mids[prod] = m[2]

        for robot, model in self.MODELS.items():
            if robot not in mids:
                continue
            # Need every basket component priced this tick
            if not all(c in mids for c in model['betas'].keys()):
                continue

            fv = model['alpha'] + sum(beta * mids[c] for c, beta in model['betas'].items())
            spread = mids[robot] - fv
            z = spread / model['std']

            current = state.position.get(robot, 0)
            target = self._target_position(z)
            delta = target - current

            best_bid, best_ask = bidask[robot]
            depth = state.order_depths[robot]
            orders: List[Order] = []

            if delta == 0:
                result[robot] = orders
                continue

            aggressive = abs(z) >= self.Z_TAKE

            # When aggressive, willing to pay up to FV +/- z_dead * sigma
            take_buffer = model['std'] * self.Z_DEAD

            if delta > 0:
                remaining = delta
                if aggressive:
                    take_cap = fv + take_buffer  # buy as long as price <= FV + buffer
                    for px in sorted(depth.sell_orders.keys()):
                        if remaining <= 0 or px > take_cap:
                            break
                        avail = -depth.sell_orders[px]
                        qty = min(remaining, avail)
                        if qty > 0:
                            orders.append(Order(robot, px, qty))
                            remaining -= qty
                if remaining > 0:
                    cap = math.floor(fv - take_buffer)
                    px = min(best_bid + 1, cap)
                    if px < best_ask:
                        orders.append(Order(robot, px, remaining))
            else:
                remaining = -delta
                if aggressive:
                    take_cap = fv - take_buffer  # sell as long as price >= FV - buffer
                    for px in sorted(depth.buy_orders.keys(), reverse=True):
                        if remaining <= 0 or px < take_cap:
                            break
                        avail = depth.buy_orders[px]
                        qty = min(remaining, avail)
                        if qty > 0:
                            orders.append(Order(robot, px, -qty))
                            remaining -= qty
                if remaining > 0:
                    cap = math.ceil(fv + take_buffer)
                    px = max(best_ask - 1, cap)
                    if px > best_bid:
                        orders.append(Order(robot, px, -remaining))

            result[robot] = orders

        return result, 0, ""


class Trader:
    def __init__(self):
        self.engine = RoboPairsTrader()

    def run(self, state: TradingState):
        return self.engine.run(state)
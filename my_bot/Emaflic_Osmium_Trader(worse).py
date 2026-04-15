from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


def compute_fair_value(order_depth):
    """
    VWAP su tutto il book: pesa ogni livello per il suo volume.
    Più accurato del mid perché riflette dove sta davvero la liquidità.
    Se il book è vuoto (non dovrebbe mai succedere qui) fallback al mid.
    """
    total_value, total_vol = 0, 0
    for price, vol in order_depth.buy_orders.items():
        total_value += price * vol
        total_vol += vol
    for price, vol in order_depth.sell_orders.items():
        total_value += price * abs(vol)
        total_vol += abs(vol)
    if total_vol > 0:
        return total_value / total_vol
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2


def compute_percentile(fair_value, price_window_short, price_window_long):
    """
    Calcola dove si trova il fair value corrente nel range storico osservato.
    Usa due finestre (corta e lunga) e prende il range più largo tra le due,
    così il percentile è significativo sia in mercati stabili che in trend.

    Ritorna un valore tra 0 e 1:
      0.0 = siamo al minimo del range → comprare è ottimale
      1.0 = siamo al massimo del range → vendere è ottimale
      0.5 = centro del range → neutro
    """
    combined = price_window_short + price_window_long
    if len(combined) < 2:
        return 0.5  # nessuna storia ancora, comportamento neutro

    # range più largo tra finestra corta e lunga
    min_short = min(price_window_short) if price_window_short else fair_value
    max_short = max(price_window_short) if price_window_short else fair_value
    min_long  = min(price_window_long)  if price_window_long  else fair_value
    max_long  = max(price_window_long)  if price_window_long  else fair_value

    range_min = min(min_short, min_long)
    range_max = max(max_short, max_long)

    if range_max == range_min:
        return 0.5  # range collassato, nessun segnale

    return (fair_value - range_min) / (range_max - range_min)


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # ── PARAMETRI DI TUNING ───────────────────────────────────────────────
        # Tutti i parametri qui sopra — cambiarli solo qui, mai nel codice sotto

        BASE_EDGE             = 6     # edge base in tick (calibrato su spread live ~16 tick)
        BASELINE_STD          = {"ASH_COATED_OSMIUM": 1.89, "INTARIAN_PEPPER_ROOT": 1.78}
        ALPHA                 = 0.05  # EWMA decay: memoria effettiva ~20 tick
        GAMMA                 = 0.1   # coefficiente skew per |pos| < soglia media
        SKEW_THRESHOLD_HIGH   = 30    # sopra questa pos lo skew diventa aggressivo
        SKEW_THRESHOLD_MID    = 15    # soglia intermedia skew
        CIRCUIT_BREAKER_LIMIT = 70    # blocca il lato pericoloso sopra questa pos
        TAKER_THRESHOLD       = 2     # tick: consuma il book se il prezzo è > X tick migliore del fv
        POS_LIMIT             = 80    # limite posizione Prosperity
        HONEY_TRAP_FRACTION   = 0.25  # frazione della size allocata agli ordini estremi
        HONEY_TRAP_OFFSET     = 4     # tick di distanza dal best bid/ask per gli ordini trappola
        WINDOW_SHORT          = 20    # finestra corta per percentile mean reversion
        WINDOW_LONG           = 100   # finestra lunga per percentile mean reversion

        # ── MEMORIA PERSISTENTE (traderData) ─────────────────────────────────
        # traderData è l'unica memoria che sopravvive tra tick.
        # La serializziamo come JSON e la ricarichiamo ogni tick.
        default_history = {
            "ASH_COATED_OSMIUM": {
                "std_ewma":       BASELINE_STD["ASH_COATED_OSMIUM"],
                "last_price":     None,
                "last_fair_value": None,
                "price_window_short": [],  # ultimi 20 fair value per percentile
                "price_window_long":  [],  # ultimi 100 fair value per percentile
                "passive_sent":   0,
                "passive_filled": 0,
                "taker_filled":   0,
            },
            "INTARIAN_PEPPER_ROOT": {
                "std_ewma":       BASELINE_STD["INTARIAN_PEPPER_ROOT"],
                "last_price":     None,
                "last_fair_value": None,
                "price_window_short": [],
                "price_window_long":  [],
                "passive_sent":   0,
                "passive_filled": 0,
                "taker_filled":   0,
            }
        }
        history = (
            json.loads(state.traderData)
            if state.traderData and state.traderData != ""
            else default_history
        )

        # ── LOOP SUI PRODOTTI ─────────────────────────────────────────────────
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product not in ("ASH_COATED_OSMIUM",):
                result[product] = orders
                continue

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            prod_history = history.get(product, default_history[product])

            # ── 1. DATI DI MERCATO ────────────────────────────────────────────
            fair_value = compute_fair_value(order_depth)
            position   = state.position.get(product, 0)
            best_bid   = max(order_depth.buy_orders.keys())
            best_ask   = min(order_depth.sell_orders.keys())

            # ── 2. FILL RATE TRACKING ─────────────────────────────────────────
            # own_trades contiene i fill del tick PRECEDENTE, non del corrente.
            # Confrontiamo il prezzo del fill con il fair value del tick in cui
            # abbiamo mandato l'ordine per capire se era taker o maker.
            prev_fv = prod_history.get("last_fair_value")
            for trade in state.own_trades.get(product, []):
                qty = trade.quantity
                if prev_fv is not None:
                    # taker: abbiamo attraversato lo spread (comprato sotto fv o venduto sopra)
                    is_taker = (
                        (qty > 0 and trade.price <= prev_fv) or
                        (qty < 0 and trade.price >= prev_fv)
                    )
                    if is_taker:
                        prod_history["taker_filled"] += abs(qty)
                    else:
                        prod_history["passive_filled"] += abs(qty)

            # ── 3. VOLATILITÀ EWMA ────────────────────────────────────────────
            # Aggiorniamo la stima di volatilità ogni tick con un filtro
            # esponenziale: i tick recenti pesano di più, quelli vecchi decadono.
            # std_ewma guida l'edge — più volatilità = spread più largo.
            std_ewma   = prod_history.get("std_ewma", BASELINE_STD[product])
            last_price = prod_history.get("last_price")
            if last_price is not None:
                ret      = abs(fair_value - last_price)
                std_ewma = ALPHA * ret + (1 - ALPHA) * std_ewma
            prod_history["std_ewma"] = std_ewma

            # ── 4. PERCENTILE MEAN REVERSION ─────────────────────────────────
            # Dove siamo nel range storico osservato?
            # Usiamo questa informazione per scalare la size:
            # vicino al massimo → compriamo meno, vendiamo di più (e viceversa).
            price_window_short = prod_history.get("price_window_short", [])
            price_window_long  = prod_history.get("price_window_long",  [])

            price_window_short.append(fair_value)
            price_window_long.append(fair_value)
            if len(price_window_short) > WINDOW_SHORT:
                price_window_short = price_window_short[-WINDOW_SHORT:]
            if len(price_window_long) > WINDOW_LONG:
                price_window_long = price_window_long[-WINDOW_LONG:]

            prod_history["price_window_short"] = price_window_short
            prod_history["price_window_long"]  = price_window_long

            percentile = compute_percentile(fair_value, price_window_short, price_window_long)
            # scalar buy: 1.0 al minimo del range, 0.0 al massimo
            # scalar sell: 0.0 al minimo, 1.0 al massimo
            buy_scalar  = 1.0 - percentile
            sell_scalar = percentile

            # ── 5. EDGE DINAMICO ──────────────────────────────────────────────
            edge = max(1, round(BASE_EDGE * std_ewma / BASELINE_STD[product]))

            # ── 6. SIZES CON PERCENTILE SCALING ──────────────────────────────
            # La size base è limitata dal position limit.
            # Moltiplichiamo per lo scalar del percentile: riduciamo il lato
            # sfavorevole rispetto al range, senza mai azzerarlo completamente
            # (max con 1 garantisce almeno 1 unità se c'è capacità).
            raw_buy_size  = POS_LIMIT - position
            raw_sell_size = POS_LIMIT + position
            buy_size  = max(0, int(raw_buy_size  * buy_scalar))  if raw_buy_size  > 0 else 0
            sell_size = max(0, int(raw_sell_size * sell_scalar)) if raw_sell_size > 0 else 0

            # ── 7. SKEW AVELLANEDA-STOIKOV ────────────────────────────────────
            # Lo skew sposta entrambi i prezzi nella direzione che riduce la pos.
            # Tre livelli: più siamo esposti, più spingiamo aggressivamente.
            if abs(position) > SKEW_THRESHOLD_HIGH:
                skew = 0.25 * position
            elif abs(position) > SKEW_THRESHOLD_MID:
                skew = 0.15 * position
            else:
                skew = GAMMA * position

            # ── 8. PREZZI PASSIVI CON QUEUE JUMP ─────────────────────────────
            # Queue jump: se il nostro prezzo è uguale o peggio del best bid/ask
            # esistente, andiamo 1 tick oltre per essere primi in coda.
            passive_bid = max(int(round(fair_value - edge - skew)), best_bid + 1)
            passive_ask = min(int(round(fair_value + edge - skew)), best_ask - 1)

            # sanity check: se lo spread è 1 tick l'edge collassa, torniamo al best
            if passive_bid >= passive_ask:
                passive_bid = best_bid
                passive_ask = best_ask

            # ── 9. CIRCUIT BREAKER ────────────────────────────────────────────
            # Se la posizione supera il limite di sicurezza, congeliamo il lato
            # che aggraverebbe l'esposizione e mandiamo solo ordini che la riducono.
            if position >= CIRCUIT_BREAKER_LIMIT:
                if raw_sell_size > 0:
                    orders.append(Order(product, passive_ask, -raw_sell_size))
                prod_history["last_price"]      = fair_value
                prod_history["last_fair_value"] = fair_value
                history[product]                = prod_history
                result[product]                 = orders
                continue

            if position <= -CIRCUIT_BREAKER_LIMIT:
                if raw_buy_size > 0:
                    orders.append(Order(product, passive_bid, raw_buy_size))
                prod_history["last_price"]      = fair_value
                prod_history["last_fair_value"] = fair_value
                history[product]                = prod_history
                result[product]                 = orders
                continue

            # ── 10. TAKER AGGRESSIVO ──────────────────────────────────────────
            # Consumiamo immediatamente qualsiasi ordine nel book che è
            # > TAKER_THRESHOLD tick migliore del fair value: fill garantito,
            # zero rischio di adverse selection su questi trade.
            if buy_size > 0:
                for ask_p, ask_vol in sorted(order_depth.sell_orders.items()):
                    if ask_p < fair_value - TAKER_THRESHOLD:
                        qty = min(-ask_vol, buy_size)
                        orders.append(Order(product, ask_p, qty))
                        buy_size -= qty
                    else:
                        break
                    if buy_size <= 0:
                        break

            if sell_size > 0:
                for bid_p, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_p > fair_value + TAKER_THRESHOLD:
                        qty = min(bid_vol, sell_size)
                        orders.append(Order(product, bid_p, -qty))
                        sell_size -= qty
                    else:
                        break
                    if sell_size <= 0:
                        break

            # ── 11. MARKET MAKING PASSIVO + HONEY TRAP ───────────────────────
            # Dividiamo la size residua in due parti:
            #   - 2/3 al prezzo passivo normale (queue jump, edge calibrato)
            #   - 1/3 agli estremi del book (honey trap)
            #
            # La honey trap cattura i bot "stupidi" di Prosperity che attraversano
            # l'intero spread senza guardare il prezzo. Se non vengono fillati
            # non costa nulla — sono ordini passivi.

            if buy_size > 0:
                honey_buy_size  = max(1, int(buy_size * HONEY_TRAP_FRACTION))
                normal_buy_size = buy_size - honey_buy_size
                honey_bid       = best_bid - HONEY_TRAP_OFFSET

                if normal_buy_size > 0:
                    orders.append(Order(product, passive_bid, normal_buy_size))
                    prod_history["passive_sent"] += normal_buy_size
                if honey_bid > 0:
                    orders.append(Order(product, honey_bid, honey_buy_size))
                    prod_history["passive_sent"] += honey_buy_size

            if sell_size > 0:
                honey_sell_size  = max(1, int(sell_size * HONEY_TRAP_FRACTION))
                normal_sell_size = sell_size - honey_sell_size
                honey_ask        = best_ask + HONEY_TRAP_OFFSET

                if normal_sell_size > 0:
                    orders.append(Order(product, passive_ask, -normal_sell_size))
                    prod_history["passive_sent"] += normal_sell_size
                orders.append(Order(product, honey_ask, -honey_sell_size))
                prod_history["passive_sent"] += honey_sell_size

            # ── 12. AGGIORNA HISTORY ──────────────────────────────────────────
            prod_history["last_price"]      = fair_value
            prod_history["last_fair_value"] = fair_value
            history[product]                = prod_history
            result[product]                 = orders

        conversions = 0
        traderData  = json.dumps(history)
        return result, conversions, traderData
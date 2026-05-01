import pandas as pd
import numpy as np

def calculate_pin_proxy(classified_trades_df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Calcola un proxy della PIN (Probability of Informed Trading) basato sull'imbalance
    degli ordini taker. Un valore alto suggerisce un'attività di trading direzionale e informata.

    Args:
        classified_trades_df (pd.DataFrame): DataFrame dei trade, deve contenere
                                             le colonne 'global_time' e 'maker_taker'.
        window (int): Finestra mobile per il calcolo.

    Returns:
        pd.Series: Una serie temporale con l'indicatore di tossicità.
    """
    if 'maker_taker' not in classified_trades_df.columns or classified_trades_df.empty:
        return pd.Series(dtype=float)

    df = classified_trades_df.set_index('global_time').sort_index()

    # Identifica trade aggressivi (taker)
    is_taker = (df['maker_taker'] == 'TAKER')
    
    # Identifica la direzione (assumiamo che TAKER buy -> dist > 0, TAKER sell -> dist < 0)
    # Se 'dist_from_mid' non c'è, usiamo una stima basata sul prezzo rispetto ai livelli L1.
    if 'dist_from_mid' in df.columns:
        taker_buys = (is_taker & (df['dist_from_mid'] > 0)).astype(int)
        taker_sells = (is_taker & (df['dist_from_mid'] < 0)).astype(int)
    else:
        # Fallback se non ci sono le colonne pre-calcolate
        taker_buys = (is_taker & (df['price'] >= df['ask_price_1'])).astype(int)
        taker_sells = (is_taker & (df['price'] <= df['bid_price_1'])).astype(int)
    
    # Calcola il flusso netto su una finestra mobile
    rolling_buys = taker_buys.rolling(window).sum()
    rolling_sells = taker_sells.rolling(window).sum()
    
    # Formula del PIN Proxy: |Buys - Sells| / (Buys + Sells)
    numerator = (rolling_buys - rolling_sells).abs()
    denominator = rolling_buys + rolling_sells

    pin_proxy = pd.Series(
        np.where(denominator > 0, numerator / denominator, 0.0),
        index=denominator.index,
    ).fillna(0.0)

    return pin_proxy.rename("trade_flow_toxicity")

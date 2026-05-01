import pandas as pd
import numpy as np

# Definizione di default per i basket noti. L'utente può sovrascriverla.
DEFAULT_BASKETS = {
    "GIFT_BASKET": {
        "constituents": {
            "CHOCOLATE": 4,
            "STRAWBERRIES": 6,
            "ROSES": 1,
        },
        "etf_fee": 300  # Costo di "creazione/redenzione" del paniere
    }
}

class BasketArbitrage:
    """
    Calcola il Net Asset Value (NAV) di un basket/ETF e lo spread (premium/discount)
    rispetto al prezzo di mercato dell'ETF stesso.
    """

    def __init__(self, full_df: pd.DataFrame, basket_asset: str, basket_definition: dict):
        self.full_df = full_df
        self.basket_asset = basket_asset
        self.constituents = basket_definition.get("constituents", {})
        self.etf_fee = basket_definition.get("etf_fee", 0)

    def calculate_nav_and_spread(self) -> pd.DataFrame:
        """
        Calcola il NAV e lo spread.

        Returns:
            pd.DataFrame: Un DataFrame con colonne [global_time, basket_price, nav, spread]
        """
        if not self.constituents or self.basket_asset not in self.full_df['product'].unique():
            return pd.DataFrame(columns=['global_time', 'basket_price', 'nav', 'spread'])

        # Pivota il DataFrame per avere i prezzi degli asset come colonne
        price_pivot = self.full_df.pivot_table(
            index='global_time',
            columns='product',
            values='mid_price'
        ).ffill().bfill() # Forward e backward fill per gestire tick mancanti

        # Estrai i dati dell'ETF
        basket_prices = price_pivot[[self.basket_asset]].copy().rename(columns={self.basket_asset: 'basket_price'})

        # Calcola il NAV sommando i prezzi pesati dei componenti
        nav_series = pd.Series(0, index=price_pivot.index)
        for asset, weight in self.constituents.items():
            if asset in price_pivot.columns:
                nav_series += price_pivot[asset] * weight
        
        # Aggiungi il costo di creazione/redenzione al NAV
        nav_series += self.etf_fee
        
        result_df = basket_prices.join(nav_series.rename('nav'))
        
        # Calcola lo spread (Premium/Discount)
        result_df['spread'] = result_df['basket_price'] - result_df['nav']
        
        # Aggiunge lo Z-score dello spread per i segnali
        spread_mean = result_df['spread'].rolling(window=100, min_periods=20).mean()
        spread_std = result_df['spread'].rolling(window=100, min_periods=20).std()
        result_df['spread_zscore'] = (result_df['spread'] - spread_mean) / spread_std.clip(lower=1e-6)
        
        return result_df.reset_index()

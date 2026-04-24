import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class PricePredictor:
    """
    Un semplice modello di Machine Learning (LightGBM) per prevedere la direzione
    del prossimo movimento di prezzo.
    """
    def __init__(self, df: pd.DataFrame, horizon: int = 10):
        self.df = df.copy()
        self.horizon = horizon

    def _engineer_features(self):
        """
        Crea feature (variabili indipendenti) e il target (variabile dipendente).
        """
        # Lag features (guard against missing columns)
        if 'ofi' in self.df.columns:
            self.df['ofi_lag1'] = self.df['ofi'].shift(1)
        if 'micro_price' in self.df.columns:
            self.df['micro_price_lag1'] = self.df['micro_price'].shift(1)
        if 'spread' in self.df.columns:
            self.df['spread_lag1'] = self.df['spread'].shift(1)
        self.df['returns'] = self.df['mid_price'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(50).std()

        # Target: direzione del prezzo futuro
        self.df['future_price'] = self.df['mid_price'].shift(-self.horizon)
        self.df['target'] = np.sign(self.df['future_price'] - self.df['mid_price'])

        # Drop NaN ONLY on model-relevant columns (not all 40+ original columns)
        drop_subset = ['returns', 'volatility', 'future_price', 'target']
        for c in ('ofi', 'ofi_lag1', 'micro_price', 'micro_price_lag1', 'spread', 'spread_lag1'):
            if c in self.df.columns:
                drop_subset.append(c)
        self.df = self.df.dropna(subset=drop_subset)

        # np.sign returns float64; map expects matching types
        self.df['target'] = self.df['target'].astype(int).map({1: 2, 0: 1, -1: 0})

    def train_and_evaluate(self):
        """
        Allena il classificatore LightGBM e ne valuta le performance.
        """
        self._engineer_features()
        
        all_features = [
            'ofi', 'ofi_lag1', 'micro_price', 'micro_price_lag1',
            'spread', 'spread_lag1', 'returns', 'volatility',
            'bid_depth', 'ask_depth'
        ]
        features = [f for f in all_features if f in self.df.columns]
        target = 'target'

        X = self.df[features]
        y = self.df[target]

        if len(X) < 100:
            return None, "Dati insufficienti per l'allenamento.", None

        # Suddivisione in set di training e test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Modello LightGBM
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=20
        )
        
        model.fit(X_train, y_train)

        # Valutazione
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred, labels=[0, 1, 2],
            target_names=['Down', 'Flat', 'Up'],
            output_dict=True, zero_division=0,
        )
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Salva le predizioni per plottarle
        self.df.loc[X_test.index, 'prediction'] = y_pred
        
        return report, feature_importance, self.df.loc[X_test.index]

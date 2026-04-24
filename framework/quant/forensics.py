import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.tsa.stattools as ts

class AssetForensics:
    """
    Modulo di analisi forense non invasivo per IMC Prosperity 4.
    Fornisce metriche avanzate di Regime, Frequenza, Microstruttura e Volatilità.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'global_time' in self.df.columns:
            self.df = self.df.sort_values('global_time')

    # ── MICROSTRUCTURE ───────────────────────────────────────────────────────

    def order_book_imbalance(self) -> pd.Series:
        """OBI statico = (Bid Depth - Ask Depth) / (Bid Depth + Ask Depth)"""
        bid_depth = self.df.get('bid_depth', self.df['bid_volume_1'].fillna(0) + self.df.get('bid_volume_2', 0) + self.df.get('bid_volume_3', 0))
        ask_depth = self.df.get('ask_depth', self.df['ask_volume_1'].fillna(0) + self.df.get('ask_volume_2', 0) + self.df.get('ask_volume_3', 0))
        return (bid_depth - ask_depth) / (bid_depth + ask_depth).clip(lower=1)

    def amihud_illiquidity(self, window: int = 100) -> pd.Series:
        """Amihud Illiquidity Ratio = |Return| / Volume"""
        rets = self.df['mid_price'].pct_change()
        vol = (self.df['bid_volume_1'].fillna(0) + self.df['ask_volume_1'].fillna(0)) / 2
        amihud = rets.abs() / vol.clip(lower=1)
        return amihud.rolling(window=window).mean()

    def return_moments(self) -> dict:
        """Kurtosis e Skewness dei rendimenti."""
        rets = self.df['mid_price'].pct_change().dropna()
        if len(rets) < 2:
            return {'skewness': 0.0, 'kurtosis': 0.0}
        return {
            'skewness': float(stats.skew(rets)),
            'kurtosis': float(stats.kurtosis(rets))
        }

    # ── REGIME ───────────────────────────────────────────────────────────────

    def rolling_hurst_exponent(self, window: int = 100) -> pd.Series:
        """
        Rolling Hurst Exponent (H).
        H < 0.5: Mean Reverting | H == 0.5: Random Walk | H > 0.5: Trending
        """
        def _hurst(ts):
            if len(ts) < 20:
                return 0.5
            lags = range(2, min(20, len(ts) // 2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            if np.any(np.isnan(tau)) or np.any(np.array(tau) == 0):
                return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(np.clip(poly[0] * 2.0, 0.0, 1.0))

        mid = self.df['mid_price'].dropna()
        if len(mid) < window:
            return pd.Series(0.5, index=self.df.index)
        return mid.rolling(window=window).apply(_hurst, raw=True)

    def shannon_entropy(self, window: int = 100, bins: int = 10) -> pd.Series:
        """Shannon Entropy sui rendimenti (in bit). Misura il disordine/incertezza."""
        rets = self.df['mid_price'].pct_change().dropna()

        def _entropy(x):
            if len(x) < 2:
                return 0.0
            counts, _ = np.histogram(x, bins=bins)
            total = counts.sum()
            if total == 0:
                return 0.0
            p = counts[counts > 0] / total   # true probabilities, not density
            return float(-np.sum(p * np.log2(p)))

        if len(rets) < window:
            return pd.Series(0.0, index=self.df.index)
        return rets.rolling(window=window).apply(_entropy, raw=True)

    def variance_ratio_test(self, lag: int = 8) -> dict:
        """
        Lo-MacKinlay Variance Ratio Test.
        VR < 1 → Mean Reverting | VR ≈ 1 → Random Walk | VR > 1 → Trending.
        More statistically robust than the Hurst exponent alone.
        """
        prices = self.df['mid_price'].dropna().values
        if len(prices) < lag * 4:
            return {'vr': 1.0, 'interpretation': 'Insufficient data'}

        log_prices = np.log(np.clip(prices, 1e-10, None))
        rets_1 = np.diff(log_prices)
        n = len(rets_1)

        var_1 = np.var(rets_1, ddof=1)
        if var_1 == 0:
            return {'vr': 1.0, 'interpretation': 'No variance'}

        # Variance of overlapping lag-period returns
        rets_q = np.array([np.sum(rets_1[i:i + lag]) for i in range(n - lag + 1)])
        var_q = np.var(rets_q, ddof=1)

        vr = var_q / (lag * var_1)

        if vr < 0.88:
            interpretation = "Mean Reverting"
        elif vr > 1.12:
            interpretation = "Trending"
        else:
            interpretation = "Random Walk"

        return {'vr': float(vr), 'interpretation': interpretation}

    def lag_autocorrelation(self, lag: int = 1) -> float:
        """
        Lag-N autocorrelation of tick returns.
        Negative → mean-reverting micro-structure. Positive → momentum/trending.
        """
        rets = self.df['mid_price'].pct_change().dropna()
        if len(rets) < lag + 2:
            return 0.0
        return float(rets.autocorr(lag=lag))

    def price_reversal_rate(self) -> float:
        """
        Fraction of non-zero price moves that reverse direction.
        > 0.55 → mean reverting | < 0.45 → trending | ≈ 0.50 → random walk.
        """
        mid = self.df['mid_price'].dropna()
        if len(mid) < 3:
            return 0.5
        changes = mid.diff().dropna()
        nonzero = changes[changes != 0]
        if len(nonzero) < 2:
            return 0.5
        signs = np.sign(nonzero.values)
        reversals = int(np.sum(signs[1:] != signs[:-1]))
        return float(reversals / (len(signs) - 1))

    # ── FREQUENCY ────────────────────────────────────────────────────────────

    def dominant_frequency_fft(self) -> dict:
        """FFT per individuare cicli dominanti o stagionalità."""
        mid = self.df['mid_price'].dropna().values
        n = len(mid)
        if n < 10:
            return {'dominant_period_ticks': 0, 'power': 0}

        try:
            mid_detrended = mid - np.polyval(np.polyfit(np.arange(n), mid, 1), np.arange(n))
            fft_vals = np.fft.rfft(mid_detrended)
            fft_power = np.abs(fft_vals)
            freqs = np.fft.rfftfreq(n, d=1.0)

            if len(fft_power) > 1:
                idx_dom = np.argmax(fft_power[1:]) + 1
                dom_freq = freqs[idx_dom]
                period = 1.0 / dom_freq if dom_freq > 0 else np.inf
                return {
                    'dominant_period_ticks': float(period),
                    'power': float(fft_power[idx_dom])
                }
        except Exception:
            pass
        return {'dominant_period_ticks': 0, 'power': 0}

    # ── VOLATILITY ───────────────────────────────────────────────────────────

    def advanced_volatility(self, bar_ticks: int = 100) -> pd.DataFrame:
        """Parkinson e Garman-Klass Volatility aggregando i tick in barre."""
        df_copy = self.df.copy()
        bar_ticks = max(bar_ticks, 5)
        df_copy['bar_id'] = np.arange(len(df_copy)) // bar_ticks
        bars = df_copy.groupby('bar_id').agg(
            open=('mid_price', 'first'),
            high=('mid_price', 'max'),
            low=('mid_price', 'min'),
            close=('mid_price', 'last')
        ).dropna()

        if bars.empty:
            return pd.DataFrame(columns=['parkinson_vol', 'garman_klass_vol'])

        hl_ratio = np.log(bars['high'] / bars['low'].clip(lower=0.1))
        bars['parkinson_vol'] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (hl_ratio ** 2))

        bars['garman_klass_vol'] = np.sqrt(
            0.5 * (np.log(bars['high'] / bars['low'].clip(lower=0.1)) ** 2) -
            (2 * np.log(2) - 1) * (np.log(bars['close'] / bars['open'].clip(lower=0.1)) ** 2)
        )

        return bars[['parkinson_vol', 'garman_klass_vol']]

    # ── CORRELATION ──────────────────────────────────────────────────────────

    @staticmethod
    def engle_granger_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
        """Engle-Granger Co-integration Test (ADF on OLS residuals)."""
        df_temp = pd.concat([series_a, series_b], axis=1).dropna()
        if len(df_temp) < 50:
            return {'is_cointegrated': False, 'p_value': 1.0, 'hedge_ratio': 1.0}

        a = df_temp.iloc[:, 0].values
        b = df_temp.iloc[:, 1].values

        slope, intercept, _, _, _ = stats.linregress(b, a)
        residuals = a - (slope * b + intercept)

        try:
            _adf = ts.adfuller(residuals)
            adf_stat, p_value = _adf[0], _adf[1]
            return {
                'is_cointegrated': bool(p_value < 0.05),
                'p_value': float(p_value),
                'hedge_ratio': float(slope),
                'adf_stat': float(adf_stat)
            }
        except Exception:
            return {'is_cointegrated': False, 'p_value': 1.0, 'hedge_ratio': 1.0}


# ── INTERFACCIA COMANDI (CLI Hooks) ──────────────────────────────────────────

def handle_discover(df: pd.DataFrame, asset: str) -> str:
    """/discover [asset] -> DNA Profile multi-signal con confidence score."""
    asset_df = df[df['product'] == asset] if 'product' in df.columns else df
    if asset_df.empty:
        return f"Asset {asset} non trovato."

    forensics = AssetForensics(asset_df)
    n = len(asset_df)

    # ── 1. Hurst Exponent ──
    hurst_w = min(200, max(20, n // 3))
    hurst_series = forensics.rolling_hurst_exponent(window=hurst_w).dropna()
    mean_hurst = float(hurst_series.mean()) if not hurst_series.empty else 0.5

    # ── 2. Variance Ratio (more robust than Hurst alone) ──
    vr_result = forensics.variance_ratio_test(lag=8)

    # ── 3. Shannon Entropy ──
    ent_w = min(200, max(20, n // 3))
    entropy_series = forensics.shannon_entropy(window=ent_w).dropna()
    mean_entropy = float(entropy_series.mean()) if not entropy_series.empty else 0.0

    # ── 4. Lag-1 Autocorrelation ──
    autocorr = forensics.lag_autocorrelation(lag=1)

    # ── 5. Price Reversal Rate ──
    reversal_rate = forensics.price_reversal_rate()

    # ── 6. Return Distribution ──
    moments = forensics.return_moments()

    # ── 7. Volatility Class ──
    bar_size = min(100, max(5, n // 20))
    adv_vol = forensics.advanced_volatility(bar_ticks=bar_size)
    mean_gk_vol = float(adv_vol['garman_klass_vol'].mean()) if not adv_vol.empty else 0.0
    vol_class = "High" if mean_gk_vol > 0.005 else ("Medium" if mean_gk_vol > 0.002 else "Low")

    # ── 8. FFT Dominant Cycle ──
    fft = forensics.dominant_frequency_fft()

    # ── Composite Scoring ──────────────────────────────────────────────────
    # Each signal contributes +1 toward MR or Trend; score range 0-6
    mr_score = 0
    trend_score = 0

    if mean_hurst < 0.47:
        mr_score += 2
    elif mean_hurst > 0.53:
        trend_score += 2

    if vr_result['vr'] < 0.88:
        mr_score += 2
    elif vr_result['vr'] > 1.12:
        trend_score += 2

    if autocorr < -0.05:
        mr_score += 1
    elif autocorr > 0.05:
        trend_score += 1

    if reversal_rate > 0.55:
        mr_score += 1
    elif reversal_rate < 0.45:
        trend_score += 1

    total_signals = 6
    if mr_score > trend_score:
        regime = "MEAN REVERTING"
        model = "Mean Reversion / Market Making"
        confidence = f"{mr_score}/{total_signals}"
    elif trend_score > mr_score:
        regime = "TRENDING"
        model = "Trend Following / Momentum"
        confidence = f"{trend_score}/{total_signals}"
    else:
        regime = "RANDOM WALK"
        model = "Market Making / Spread Capture"
        confidence = f"3/{total_signals} (tie)"

    def _hurst_tag(h):
        if h < 0.47: return "↓ MR"
        if h > 0.53: return "↑ Trend"
        return "~ RW"

    def _autocorr_tag(a):
        if a < -0.05: return "↓ MR"
        if a > 0.05:  return "↑ Trend"
        return "~ Neutral"

    def _reversal_tag(r):
        if r > 0.55: return "↓ MR"
        if r < 0.45: return "↑ Trend"
        return "~ RW"

    cycle_str = f"{fft['dominant_period_ticks']:.0f} ticks" if fft['dominant_period_ticks'] > 0 else "N/A"

    return (
        f"╔══════════════════════════════════════════╗\n"
        f"  DNA Profile — {asset}\n"
        f"  Regime   : {regime}\n"
        f"  Confidence: {confidence}\n"
        f"╚══════════════════════════════════════════╝\n"
        f"\n── Regime Signals ─────────────────────────\n"
        f"  Hurst Exponent      : {mean_hurst:.3f}  {_hurst_tag(mean_hurst)}\n"
        f"  Variance Ratio (8)  : {vr_result['vr']:.3f}  [{vr_result['interpretation']}]\n"
        f"  Lag-1 AutoCorr      : {autocorr:+.3f}  {_autocorr_tag(autocorr)}\n"
        f"  Price Reversal Rate : {reversal_rate:.1%}  {_reversal_tag(reversal_rate)}\n"
        f"\n── Distribution ───────────────────────────\n"
        f"  Shannon Entropy     : {mean_entropy:.3f} bit\n"
        f"  Return Skewness     : {moments['skewness']:+.3f}\n"
        f"  Return Kurtosis     : {moments['kurtosis']:.3f}  {'[Fat tails]' if moments['kurtosis'] > 1 else '[Normal]'}\n"
        f"\n── Volatility & Structure ─────────────────\n"
        f"  Volatility Class    : {vol_class}  (GK: {mean_gk_vol:.5f})\n"
        f"  Dominant FFT Cycle  : {cycle_str}\n"
        f"\n── Suggested Model ────────────────────────\n"
        f"  {model}"
    )


def handle_forensic_report(df: pd.DataFrame) -> str:
    """/forensic-report -> Matrice di co-integrazione cross-asset."""
    if 'product' not in df.columns:
        return "Il DataFrame deve contenere multipli asset per generare il report."

    pivot = df.pivot_table(index="global_time", columns="product", values="mid_price").ffill()
    assets = pivot.columns.tolist()
    if len(assets) < 2:
        return "Dati insufficienti per il report multi-asset."

    report = ["=== Cross-Asset Co-Integration Report ==="]

    for i, a in enumerate(assets):
        for b in assets[i + 1:]:
            res = AssetForensics.engle_granger_cointegration(pivot[a], pivot[b])
            status = "COINTEGRATED" if res['is_cointegrated'] else "Independent"
            report.append(f"{a} vs {b}: {status} (p-value: {res['p_value']:.4f}, hedge_ratio: {res['hedge_ratio']:.3f})")

    return "\n".join(report)


def handle_impact_test(df: pd.DataFrame, asset: str) -> str:
    """/impact-test [asset] -> Slippage teorico e size limite via Amihud Ratio."""
    asset_df = df[df['product'] == asset] if 'product' in df.columns else df
    if asset_df.empty:
        return f"Asset {asset} non trovato."

    forensics = AssetForensics(asset_df)
    amihud = forensics.amihud_illiquidity(window=500).dropna()
    mean_amihud = float(amihud.mean()) if not amihud.empty else 0.0

    mid_price = asset_df['mid_price'].iloc[-1]

    size_test = 10
    impact_pct = mean_amihud * size_test
    impact_ticks = impact_pct * mid_price

    max_slippage_ticks = 2.0
    if mean_amihud > 0:
        max_size = (max_slippage_ticks / mid_price) / mean_amihud
    else:
        max_size = np.inf

    return (f"[Impact Test - {asset}]\n"
            f"Mean Amihud Ratio   : {mean_amihud:.6e}\n"
            f"Expected Slippage   : {impact_ticks:.2f} ticks (for Size = {size_test})\n"
            f"Suggested Max Size  : {max_size:.0f} (capped at 2-tick slippage)")

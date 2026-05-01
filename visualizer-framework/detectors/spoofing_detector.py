import pandas as pd
import numpy as np
from framework.detectors.base import BotDetector, DetectionResult

class SpoofingDetector(BotDetector):
    """
    Detects potential spoofing or layering activity in the order book.
    Pattern:
    1. A large increase in resting volume at L2 or L3 (the "spoof layer").
    2. This layer is quickly cancelled (persists for a short duration).
    3. The cancellation often coincides with aggressive trades on the opposite side.
    """
    def __init__(self, size_threshold_std: float = 3.0, duration_threshold_ticks: int = 500):
        self.size_threshold_std = size_threshold_std
        self.duration_threshold_ticks = duration_threshold_ticks

    def detect(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> DetectionResult:
        product = prices_df["product"].iloc[0] if "product" in prices_df.columns else "UNKNOWN"
        df = prices_df.copy()

        if len(df) < self.duration_threshold_ticks:
            return self._insufficient("SpoofingDetector", product, len(df))

        # Calcola le variazioni (delta) nei volumi a L2 e L3
        df['l23_bid_vol'] = df['bid_volume_2'].fillna(0) + df['bid_volume_3'].fillna(0)
        df['l23_ask_vol'] = df['ask_volume_2'].fillna(0) + df['ask_volume_3'].fillna(0)
        df['l23_bid_delta'] = df['l23_bid_vol'].diff()
        df['l23_ask_delta'] = df['l23_ask_vol'].diff()

        # Definisci la soglia per un "grande" volume come multiplo della deviazione standard
        bid_size_threshold = df['l23_bid_delta'].std() * self.size_threshold_std
        ask_size_threshold = df['l23_ask_delta'].std() * self.size_threshold_std

        # Trova gli eventi di "layering" (grande aggiunta di volume)
        bid_layers = df[df['l23_bid_delta'] > bid_size_threshold]
        ask_layers = df[df['l23_ask_delta'] > ask_size_threshold]

        spoofing_events = []
        for side_layers, delta_col in [(bid_layers, 'l23_bid_delta'), (ask_layers, 'l23_ask_delta')]:
            if side_layers.empty:
                continue

            for index, row in side_layers.iterrows():
                start_time = row['global_time']
                initial_size = row[delta_col]
                
                # Cerca la cancellazione corrispondente entro la finestra di durata
                window_df = df[(df['global_time'] > start_time) & (df['global_time'] <= start_time + self.duration_threshold_ticks)]
                
                # La cancellazione è un delta negativo di dimensione simile
                cancellations = window_df[window_df[delta_col] < -initial_size * 0.8]
                
                if not cancellations.empty:
                    # Abbiamo trovato un evento di layering e rapida cancellazione
                    end_time = cancellations.iloc[0]['global_time']
                    spoofing_events.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'size': initial_size,
                        'side': 'BID' if 'bid' in delta_col else 'ASK'
                    })

        if not spoofing_events:
            return DetectionResult(
                detector_name="SpoofingDetector",
                product=product,
                confidence=0.0,
                pattern_description="No significant layering/spoofing activity detected.",
                counter_strategy="N/A",
                evidence={'events_found': 0}
            )
            
        # Calcola la confidenza in base alla frequenza e alla rapidità degli eventi
        avg_duration = np.mean([e['duration'] for e in spoofing_events])
        frequency_score = min(1.0, len(spoofing_events) / 50.0) # Normalizza su 50 eventi
        duration_score = 1.0 - min(1.0, avg_duration / self.duration_threshold_ticks) # Più veloce è, più alto il punteggio
        confidence = np.clip(frequency_score * duration_score, 0.0, 1.0)
        
        flagged_ts = [e['start_time'] for e in spoofing_events]

        return DetectionResult(
            detector_name="SpoofingDetector",
            product=product,
            confidence=float(confidence),
            pattern_description=f"Detected {len(spoofing_events)} instances of large, transient order book layers. "
                              f"Average spoof duration: {avg_duration:.0f} ticks.",
            counter_strategy="This pattern attempts to induce panic. Fade the initial move. Place passive orders on the "
                             "opposite side of the spoofed layer to capture the reversal when the layer is pulled.",
            evidence={
                'events_found': len(spoofing_events),
                'avg_duration_ticks': avg_duration,
                'size_threshold_std': self.size_threshold_std
            },
            flagged_timestamps=flagged_ts
        )

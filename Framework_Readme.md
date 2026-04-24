# Prosperity 4 - Universal Analysis Framework

Toolkit "plug-and-play" per analisi post-trade, scoperta di pattern e identificazione di opportunità di arbitraggio per IMC Prosperity 4.

---

## 1. Setup e Avvio

```bash
pip install -r requirements.txt
streamlit run framework/dashboard.py
```

Il dashboard si apre automaticamente nel browser sulla porta 8501.

---

## 2. Struttura del Progetto

```
framework/
├── dashboard.py              ← Entry point Streamlit (8 tab)
├── pipeline/
│   ├── loader.py             ← UniversalLoader: .log / .json / CSV
│   ├── normalizer.py         ← LOBNormalizer: micro_price, OFI, TAKER/MAKER
│   └── registry.py           ← RoundRegistry: auto-scan directory round/day
├── microstructure/
│   └── toxicity.py           ← calculate_pin_proxy() → Trade Flow Toxicity
├── detectors/
│   ├── hidden_taker.py       ← HiddenTakerDetector
│   ├── insider.py            ← InsiderSimulator (markout alpha)
│   ├── pseudo_directional.py ← PseudoDirectionalDetector
│   └── spoofing_detector.py  ← SpoofingDetector (layering / iceberg)
├── quant/
│   ├── forensics.py          ← AssetForensics + handle_discover (DNA)
│   ├── microstructure.py     ← micro_price_series, ofi_series
│   ├── signals.py            ← EMA fair value, MACD, RSI, inventory pressure
│   └── statarb.py            ← pair_spread, spread_zscore, OU half-life
├── ml/
│   └── predictor.py          ← PricePredictor (LightGBM direction classifier)
├── charts/
│   ├── heatmap.py            ← lob_heatmap (order book depth + trade markers)
│   ├── quant_panels.py       ← mm_panel, statarb_panel, vol_panel
│   ├── arbitrage_panels.py   ← basket_arb_panel, vol_arb_panel
│   └── bot_report.py         ← detection_summary_table, bot_detection_figure
└── arbitrage/
    └── basket.py             ← DEFAULT_BASKETS (ETF compositions)
```

---

## 3. Flusso Dati

```
Data Input (loader.py)
    ↓
LOBNormalizer.normalize()        → mid_price, micro_price, spread, OFI
LOBNormalizer.classify_trades()  → maker_taker, dist_from_mid
    ↓
[4 Bot Detectors] [Quant Signals] [DNA Forensics] [ML Predictor]
    ↓
Dashboard (8 Tab Streamlit)
```

### Classificazione TAKER/MAKER

| Condizione                    | Label  | Significato                        |
|-------------------------------|--------|------------------------------------|
| `price >= ask_price_1`        | TAKER  | Acquisto aggressivo (lift the ask) |
| `price <= bid_price_1`        | TAKER  | Vendita aggressiva (hit the bid)   |
| `bid < price < ask` (spread)  | MAKER  | Ordine passivo eseguito nel mezzo  |
| Prezzo non abbinabile         | UNKNOWN| Tick senza snapshot LOB precedente |

> **Nota**: Prima del fix (rev. 2025-04), `price <= bid1` era erroneamente classificato MAKER, rendendo il PIN proxy sempre 0 sul lato sell. Ora entrambi i lati della tossicità sono catturati correttamente.

---

## 4. Tab Dashboard

| Tab | Descrizione |
|-----|-------------|
| **Overview** | PnL totale e per asset, prezzo con overlay bid/ask, inventario cumulato |
| **LOB Heatmap** | Mappa di profondità del book + overlay "Taker Buy" (verde ▲) / "Taker Sell" (rosso ▼) |
| **Bot Detectors** | 4 detector avversariali con confidence score e counter-strategy |
| **Quant Signals** | EMA fair value, MACD, RSI, Stat Arb (cointegrazione + OU half-life), volatilità |
| **Arbitrage** | Basket/ETF arbitrage (Z-score vs NAV), Volatility Arbitrage (GK vs storica) |
| **Forensics** | **DNA Discovery**, Impact Test, Co-integration Report, Trade Flow Toxicity |
| **ML Predictor** | LightGBM direction classifier (Down/Flat/Up) con feature importance |
| **Data Explorer** | Esplorazione raw data normalizzato + export CSV |

---

## 5. Modulo Forense — DNA Discovery

### `handle_discover(df, asset)` → DNA Profile

Esegui cliccando **"🔍 Run DNA Discovery"** nella tab Forensics. Output:

```
╔══════════════════════════════════════════╗
  DNA Profile — STARFRUIT
  Regime   : MEAN REVERTING
  Confidence: 5/6
╚══════════════════════════════════════════╝

── Regime Signals ─────────────────────────
  Hurst Exponent      : 0.421  ↓ MR
  Variance Ratio (8)  : 0.712  [Mean Reverting]
  Lag-1 AutoCorr      : -0.183  ↓ MR
  Price Reversal Rate : 61.4%  ↓ MR

── Distribution ───────────────────────────
  Shannon Entropy     : 2.847 bit
  Return Skewness     : +0.023
  Return Kurtosis     : 3.142  [Fat tails]

── Volatility & Structure ─────────────────
  Volatility Class    : Low  (GK: 0.00041)
  Dominant FFT Cycle  : 84 ticks

── Suggested Model ────────────────────────
  Mean Reversion / Market Making
```

### Segnali inclusi nel DNA Profile

| Segnale | Descrizione | MR se | Trend se |
|---------|-------------|-------|----------|
| **Hurst Exponent** | "Memoria" della serie. Basato su scaling delle varianze | H < 0.47 | H > 0.53 |
| **Variance Ratio (Lo-MacKinlay)** | Test statistico più robusto di Hurst per MR vs Trend | VR < 0.88 | VR > 1.12 |
| **Lag-1 AutoCorr** | Correlazione tra rendimenti consecutivi | < -0.05 | > +0.05 |
| **Price Reversal Rate** | % di mosse di prezzo che cambiano direzione | > 55% | < 45% |
| **Shannon Entropy** | Incertezza della distribuzione dei rendimenti | — | — |
| **Kurtosis** | Code grasse (> 1 = non-normale, più MR) | — | — |
| **Garman-Klass Vol** | Volatilità intraday (low/medium/high) | — | — |
| **FFT Dominant Cycle** | Periodo ciclico dominante nella serie prezzi | — | — |

Il **Confidence score** (es. `5/6`) conta quanti dei 4 segnali di regime votano nella stessa direzione.

### Altri tool forensici

- **⚖️ Impact Test** → Amihud ratio + slippage stimato per size=10 + max size suggerita (cap 2 tick)
- **🧬 Cross-Asset Co-integration** → Matrice Engle-Granger per tutti i pairs; individua opportunità Stat Arb

---

## 6. Trade Flow Toxicity (PIN Proxy)

Visualizzato in fondo alla tab Forensics come grafico temporale.

**Formula**: `PIN = |Taker Buys - Taker Sells| / (Taker Buys + Taker Sells)` su finestra mobile di 60 tick.

- **PIN ≈ 0**: flusso bilanciato, mercato sano per il market making
- **PIN → 1**: flusso fortemente direzionale, probabile informed trader attivo → **aumenta spread, riduci size**

Il grafico mostra la curva smoothed (media 10 tick). Picchi sostenuti precedono spesso grandi movimenti.

---

## 7. Bot Detector

| Detector | Cosa rileva | Confidence alta se |
|----------|-------------|-------------------|
| **HiddenTakerDetector** | Erosione del best bid/ask senza ordini visibili | Frequenza alta di hit consecutivi |
| **PseudoDirectionalDetector** | Bot a orari fissi, intervalli regolari, bande di prezzo rigide | Pattern ripetitivi con bassa varianza |
| **InsiderSimulator** | Trader con vero alpha (markout positivo futuro) | Markout medio > soglia su N trade |
| **SpoofingDetector** | Layering: ordini L2/L3 grandi e temporanei | Order placed/cancelled in < K tick |

---

## 8. Quant Signals

### Market Making Panel
- EMA fair value con banda ±0.5 spread
- MACD (12/26/9) + RSI(14) per confirmation
- Inventory pressure (net position vs limit)

### Stat Arb Panel
- Engle-Granger co-integration test tra due asset selezionati
- Spread Z-score con soglie ±2σ
- **OU Half-Life** (Ornstein-Uhlenbeck): tempo atteso al mean revert — fondamentale per sizing del tempo di holding

### Volatility Panel
- Rolling StdDev (10/50 tick)
- Confronto Parkinson vs Garman-Klass (tab Forensics)
- Volatility Arbitrage: GK reactiva vs storica lenta → anticipa breakout

---

## 9. Arbitrage

### Basket / ETF Arbitrage
- Confronto prezzo ETF vs NAV (somma pesata dei componenti)
- Z-score dello spread: `> +2σ` → short ETF / long basket; `< -2σ` → long ETF / short basket
- I basket sono configurati in `framework/arbitrage/basket.py`

### Volatility Arbitrage
- GK vol reattiva (intraday) vs vol storica lenta
- Quando GK spike mentre storica è bassa → imminente espansione della volatilità

---

## 10. ML Predictor

LightGBM multi-class: **Down / Flat / Up** sul prossimo N tick (configurabile 5-50).

**Feature**: OFI, micro_price, spread, bid/ask vol L1, rolling volatility, inventory pressure.

Output: classification report (precision/recall/F1) + feature importance. Utile per confermare segnali o come filtro per entrare in trade.

---

## 11. Workflow Consigliato

1. **Sidebar** → carica `.log` dal backtest oppure directory CSV (`data/round2`)
2. **Overview** → controlla PnL e posizioni per identificare asset problematici
3. **Bot Detectors** → cerca confidence > 70%; leggi le counter-strategy
4. **Forensics / DNA Discovery** → determina il regime dell'asset (MR vs Trend) per scegliere la strategia
5. **Forensics / Toxicity** → controlla se ci sono spike di flusso informato prima dei movimenti
6. **Quant Signals / Stat Arb** → verifica co-integrazione e half-life per calibrare il pairs trading
7. **Arbitrage** → controlla Z-score basket per opportunità di arbitraggio immediate
8. **ML Predictor** → allena il modello sull'asset scelto e analizza feature importance
9. **Data Explorer** → esporta CSV per analisi offline (Jupyter, Excel)

---

*Framework aggiornato: fix TAKER classification (sell-side), fix Shannon entropy, aggiunta Variance Ratio + Lag-1 AutoCorr + Price Reversal Rate nel DNA profile.*

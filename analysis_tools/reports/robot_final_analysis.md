# ROBOT Products Final Analysis (Round 5)

## 1. Tesi Finanziaria e Fonte dell'Edge
La strategia di successo per il cluster `ROBOT` (`ROBOT_DISHES`, `ROBOT_VACUUMING`, `ROBOT_MOPPING`, `ROBOT_LAUNDRY`, `ROBOT_IRONING`) si basa su **Arbitraggio Statistico di Struttura (Stat Arb)**.
L'analisi dei dati ha rivelato una forte cointegrazione e correlazione tra i prezzi dei `ROBOT` e i prodotti della famiglia `MICROCHIP` (in particolare `MICROCHIP_OVAL` e `MICROCHIP_SQUARE`). 
Il mercato non è perfettamente efficiente nel prezzare simultaneamente i derivati (i ROBOT assemblati) in risposta a variazioni nel costo dei componenti primari (i MICROCHIP). L'edge deriva dallo sfruttare questo ritardo: prevedendo il "Fair Value" teorico di ogni `ROBOT` tramite una regressione lineare multipla (OLS) sui prezzi mid dei chip, si stima un residuo (Prezzo di Mercato - Fair Value). Poiché la relazione strutturale costringe i prezzi a riallinearsi, i grandi residui tendono inevitabilmente alla *mean reversion*.

## 2. Perché il mercato prezza male l'asset?
A causa dell'asimmetria nell'aggiornamento degli ordini: quando i prezzi di `MICROCHIP_OVAL` subiscono shock esogeni, l'order book dei prodotti dipendenti (`ROBOT_VACUUMING`, ecc.) non si aggiusta istantaneamente con la stessa proporzione. I market maker lasciano un *lag* nei propri limit orders che crea sacche di mispricing localizzato temporaneo.

## 3. Rischio Principale
Il rischio maggiore che potrebbe invalidare la strategia è un **cambio strutturale dei pesi (coefficienti alfa o beta)**. Qualora la "ricetta" o l'importanza dei componenti dovesse cambiare drasticamente nei giorni successivi non testati, i modelli OLS pre-calcolati restituirebbero stime errate causando accumuli permanenti di posizione contraria.

## 4. Analisi Dettagliata e Iterazioni Effettuate
### Fase Statistica
1. **Analisi Autocorrelazione (Hurst & Lag):** L'esponente di Hurst ha dimostrato che i prodotti sono vicini a un random walk (~0.49), sebbene `ROBOT_DISHES` mostrasse segni di mean reversion a lag ristretti.
2. **Analisi Cross-Cluster:** Test PCA e OLS (su input in `robot_comprehensive_analysis.py`) hanno evidenziato la schiacciante influenza di `MICROCHIP_OVAL` e `MICROCHIP_SQUARE`. Ad esempio, OVAL ha una correlazione di -0.82 con `ROBOT_DISHES` e +0.87 con `ROBOT_VACUUMING`.

### Ciclo di Sviluppo Strategico
- **Iterazione 1 (`robot_v1.py`):** Pure Market Making con aggiustamento inventario (basato su Simple Moving Average - SMA). Ha generato un **PnL fortemente negativo (-1.41M)** con flag `--match-trades worse`, dimostrando che la pura media mobile subiva un severo adverse selection a causa dei fill pessimistici e della direzionalità innescata dai componenti.
- **Iterazione 2 (`robot_v2.py`):** Modello Stat-Arb puro basato sui coefficienti OLS calcolati (`Fair Value = alpha + b1 * OVAL + b2 * SQUARE`). La strategia sbilanciava quote di bid-ask in base al Z-Score del residuo, andando *market/aggressiva* sopra specifiche soglie di z-score (1.5). Questa iterazione ha portato a un PnL super positivo (150-160k totali), ma infrangeva il vincolo di `Mean Absolute Position < 8` (era attestato su 9-10).
- **Iterazione 3 (Affinamento `robot_v2.py` in `robot_best.py`):** Abbassamento rigido del moltiplicatore scalare e riduzione del `position_limit` interno a `7` per garantire in modo deterministico una `Mean absolute position < 8`.

## 5. Risultati Finali (`robot_best.py`)
Il bot finale incontra rigidamente **tutti i criteri e limiti qualitativi** della challenge:
- **PnL Totale:** +105,744
  - Day 2: +29,994
  - Day 3: +33,360
  - Day 4: +42,391
- **Sharpe Annualizzato:** 32.62 (> 1.0)
- **Max Drawdown Pct:** 0.85% (< 30%)
- **Mean Absolute Position:** Tutte ampiamente sotto 8.00 (Max: `ROBOT_LAUNDRY` a 7.39)

Il log del backtester finale è stato salvato in `backtests/robot_best.log` ed il codice sorgente è in `my_bot/round5/robot_best.py`.

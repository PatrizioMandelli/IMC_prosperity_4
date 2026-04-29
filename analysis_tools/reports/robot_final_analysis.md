# ROBOT Products Final Analysis (Round 5)

## 1. Tesi Finanziaria e Fonte dell'Edge
La strategia per il cluster `ROBOT` (`DISHES`, `IRONING`, `LAUNDRY`, `MOPPING`, `VACUUMING`) si basa su **Arbitraggio Statistico (Stat-Arb)** cross-cluster. 
- **Edge**: I prezzi dei ROBOT sono strutturalmente legati ai prezzi dei loro componenti fondamentali, i `MICROCHIP`. In particolare, `MICROCHIP_OVAL` e `MICROCHIP_SQUARE` spiegano oltre l'80% della varianza nei prezzi dei ROBOT.
- **Meccanismo**: Esiste un ritardo (lag) tra il movimento dei prezzi dei chip e l'aggiornamento dei book dei ROBOT. Sfruttando questo ritardo tramite una regressione OLS, è possibile calcolare il "Fair Value" istantaneo e identificare situazioni di mispricing.

## 2. Perché il mercato prezza male l'asset?
Il mispricing deriva dalla frammentazione della liquidità e dalla latenza dei market maker sui prodotti derivati (i ROBOT). Mentre i chip sono asset primari con scoperta del prezzo rapida, i ROBOT tendono a seguire con un leggero ritardo, creando opportunità di mean reversion dei residui.

## 3. Rischio Principale
Il rischio maggiore è un **breakdown della correlazione** o un cambio nei pesi strutturali (es. un ROBOT che inizia a usare più chip di un tipo diverso). In tali casi, il modello OLS fornirebbe un fair value errato, portando a perdite sistematiche.

## 4. Analisi Dettagliata e Sviluppo
### Fase di Analisi
1. **Correlazione**: Identificata una correlazione fortissima tra i ROBOT e la coppia OVAL/SQUARE.
2. **Regressione Multivariata**: Calcolati i coefficienti alpha e beta per ogni prodotto.
   - `ROBOT_VACUUMING = 8883.57 + 0.2096 * OVAL - 0.1053 * SQUARE`
   - `ROBOT_DISHES = 12052.99 - 0.2806 * OVAL + 0.0192 * SQUARE`
3. **Z-Score**: I residui (Prezzo - Fair) mostrano una forte tendenza alla mean reversion.

### Iterazioni del Bot
- **ROBOT_v1**: Market Making puro intorno al Fair Value. PnL eccellente (+208k) ma posizioni medie troppo alte (> 8).
- **ROBOT_v2 (Best)**: Introdotto limite di posizione interno (7) e skew d'inventario cubico aggressivo. Aggiunta logica di "Liquidity Taking" market-order per Z-Score > 1.8 per catturare i movimenti più rapidi.

## 5. Risultati Finali (`robot_best.py`)
Backtest eseguito su Round 5 (Day 2-4) con flag `--match-trades worse`.

| Metrica | Valore |
| :--- | :--- |
| **PnL Totale** | **+138.428** |
| **Sharpe Ratio (Ann.)** | **38,46** |
| **Max Drawdown** | **1,0%** |
| **PnL Day 2** | **10.132** |
| **PnL Day 3** | **20.745** |
| **PnL Day 4** | **107.551** |

### Gestione Inventario
Tutte le posizioni medie assolute sono inferiori a 8.00 (Max: `ROBOT_LAUNDRY` 7.25), rispettando pienamente i vincoli di qualità.

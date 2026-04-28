"""
Calcola i valori ottimali di b1 e b2 in base alla distribuzione
stimata dei tipi di giocatori nel mercato.

Modello:
  - Ogni tipo di giocatore ha una distribuzione di b2 caratteristica
  - avg_b2 viene calcolata come media pesata su tutti i tipi
  - Si esegue una grid search su (b1, b2) per massimizzare il profitto atteso
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

PLAYER_TYPES = {
    "Nash equilibrium": {
        "fraction": 0.70,          # percentuale di giocatori di questo tipo
        "b2_mean":  835,           # b2 medio per questo tipo
        "b2_std":   0,             # deviazione standard (0 = deterministico)
    },
    "Semi-analitico": {
        "fraction": 0.30,
        "b2_mean":  858,
        "b2_std":   10,
    },
}

# Griglia prezzi di riserva: 670, 675, ..., 920
RESERVE_PRICES = np.arange(670, 921, 5)   # 51 valori
FAIR_VALUE     = 920
MIN_PRICE      = 670

# ──────────────────────────────────────────────
# FUNZIONI CORE
# ──────────────────────────────────────────────

def compute_avg_b2(player_types: dict, n_samples: int = 200_000) -> float:
    """
    Stima avg_b2 campionando dalla distribuzione di ciascun tipo.
    Usa una normale troncata nell'intervallo [670, 920] per i tipi
    con std > 0, altrimenti usa il valore deterministico.
    """
    samples = []
    for ptype, params in player_types.items():
        n = int(params["fraction"] * n_samples)
        mu, sigma = params["b2_mean"], params["b2_std"]
        if sigma == 0:
            vals = np.full(n, mu, dtype=float)
        else:
            raw = np.random.normal(mu, sigma, size=n * 4)
            # tronca e arrotonda ai multipli di 5 più vicini
            raw = np.clip(raw, MIN_PRICE, FAIR_VALUE)
            raw = np.round(raw / 5) * 5
            vals = raw[:n]
        samples.append(vals)
    all_b2 = np.concatenate(samples)
    return float(np.mean(all_b2))


def profit(b1: float, b2: float, avg_b2: float) -> dict:
    """
    Calcola il profitto atteso (proporzionale a N) per una coppia (b1, b2)
    dato avg_b2. Restituisce un dict con breakdown.
    """
    # Profitto da bid1: controparti con reserve <= b1
    n1     = int(np.sum(RESERVE_PRICES <= b1))
    margin1 = FAIR_VALUE - b1
    p1     = n1 * margin1

    # Controparti raggiungibili solo da bid2: b1 < reserve <= b2
    n2     = int(np.sum((RESERVE_PRICES > b1) & (RESERVE_PRICES <= b2)))
    margin2 = FAIR_VALUE - b2

    if b2 > avg_b2:
        # Caso A: nessuna penalità
        p2      = n2 * margin2
        penalty = 1.0
        case    = "A"
    else:
        # Caso B: penalità cubica
        if b2 >= FAIR_VALUE:
            penalty = 0.0
        else:
            penalty = ((FAIR_VALUE - avg_b2) / (FAIR_VALUE - b2)) ** 3
        p2   = n2 * margin2 * penalty
        case = "B"

    return {
        "total":   p1 + p2,
        "p1":      p1,
        "p2":      p2,
        "n1":      n1,
        "n2":      n2,
        "margin1": margin1,
        "margin2": margin2,
        "penalty": penalty,
        "case":    case,
    }


def grid_search(avg_b2: float) -> tuple:
    """
    Cerca i valori ottimali di b1 e b2 su tutta la griglia (multipli di 5).
    Restituisce (b1_opt, b2_opt, profitto_max, matrice_profitti).
    """
    grid = RESERVE_PRICES  # 670..920 step 5
    profit_matrix = np.zeros((len(grid), len(grid)))

    for i, b1 in enumerate(grid):
        for j, b2 in enumerate(grid):
            if b2 <= b1:
                profit_matrix[i, j] = np.nan
                continue
            profit_matrix[i, j] = profit(b1, b2, avg_b2)["total"]

    idx = np.nanargmax(profit_matrix)
    i_opt, j_opt = np.unravel_index(idx, profit_matrix.shape)
    return grid[i_opt], grid[j_opt], profit_matrix[i_opt, j_opt], profit_matrix


def sensitivity_analysis(player_types: dict, n_scenarios: int = 500):
    """
    Analisi di sensitività: campiona n_scenarios realizzazioni di avg_b2
    (una per ogni scenario di mercato) e per ognuna calcola il profitto
    di tre strategie fisse: Nash, Robusta, e l'ottimale di scenario.
    """
    strategies = {
        "Nash (750/835)":     (750, 835),
        "Robusta (770/875)":  (770, 875),
    }
    records = {k: [] for k in strategies}
    records["Ottimale di scenario"] = []
    avg_b2_samples = []

    for _ in range(n_scenarios):
        avg_b2 = compute_avg_b2(player_types, n_samples=20_000)
        avg_b2_samples.append(avg_b2)
        for name, (b1, b2) in strategies.items():
            records[name].append(profit(b1, b2, avg_b2)["total"])
        b1_opt, b2_opt, p_opt, _ = grid_search(avg_b2)
        records["Ottimale di scenario"].append(p_opt)

    return avg_b2_samples, records


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    np.random.seed(42)

    # 1) Verifica che le frazioni sommino a 1
    total_frac = sum(v["fraction"] for v in PLAYER_TYPES.values())
    assert abs(total_frac - 1.0) < 1e-9, f"Le frazioni devono sommare a 1, attuale: {total_frac}"

    # 2) Calcola avg_b2
    print("=" * 60)
    print("IMC Prosperity 4 – Manual Challenge Optimizer")
    print("=" * 60)
    print("\nDistribuzione giocatori:")
    for name, p in PLAYER_TYPES.items():
        print(f"  {name:25s} {p['fraction']*100:5.1f}%  b2_mean={p['b2_mean']}  b2_std={p['b2_std']}")

    avg_b2 = compute_avg_b2(PLAYER_TYPES)
    print(f"\nAvg_b2 stimato:  {avg_b2:.2f}")

    # 3) Grid search
    b1_opt, b2_opt, p_opt, profit_matrix = grid_search(avg_b2)
    res = profit(b1_opt, b2_opt, avg_b2)

    print(f"\n{'─'*60}")
    print(f"  Ottimale:  b1 = {int(b1_opt)}   b2 = {int(b2_opt)}")
    print(f"  Profitto totale atteso: {p_opt:.1f}")
    print(f"  Profitto da bid1:       {res['p1']:.1f}  ({res['n1']} controparti, margine {res['margin1']})")
    print(f"  Profitto da bid2:       {res['p2']:.1f}  ({res['n2']} controparti, margine {res['margin2']})")
    print(f"  Penalità bid2:          {res['penalty']:.4f}  (Caso {res['case']})")
    print(f"{'─'*60}")

    # Confronto con strategie fisse
    nash = profit(750, 835, avg_b2)
    rob  = profit(770, 875, avg_b2)
    print(f"\nConfronto strategie:")
    print(f"  Nash    (750/835):  {nash['total']:.1f}  (Caso {nash['case']}, penalità {nash['penalty']:.4f})")
    print(f"  Robusta (770/875):  {rob['total']:.1f}  (Caso {rob['case']}, penalità {rob['penalty']:.4f})")
    print(f"  Ottimale trovata:   {p_opt:.1f}")

    # 4) Analisi di sensitività
    print(f"\nAnalisi di sensitività su 500 scenari di mercato...")
    avg_b2_samples, records = sensitivity_analysis(PLAYER_TYPES, n_scenarios=500)

    # 5) Plot – solo heatmap
    fig, ax_heat = plt.subplots(figsize=(9, 7))
    fig.suptitle("IMC Prosperity 4 – Heatmap profitto b1/b2", fontsize=13, fontweight='bold')

    im = ax_heat.imshow(
        profit_matrix, origin='lower', aspect='auto',
        extent=[RESERVE_PRICES[0], RESERVE_PRICES[-1], RESERVE_PRICES[0], RESERVE_PRICES[-1]],
        cmap='YlOrRd', interpolation='nearest'
    )
    ax_heat.set_xlabel("b2", fontsize=11)
    ax_heat.set_ylabel("b1", fontsize=11)
    ax_heat.set_title(f"avg_b2 stimato = {avg_b2:.0f}", fontsize=10, color='gray')
    fig.colorbar(im, ax=ax_heat, label="profitto atteso")
    ax_heat.plot(b2_opt, b1_opt, 'w*', markersize=16, label=f"ottimale ({int(b1_opt)}, {int(b2_opt)})", zorder=5)
    ax_heat.plot(835, 750, 'b^', markersize=9, label="Nash (750, 835)", zorder=5)
    ax_heat.plot(875, 770, 'g^', markersize=9, label="Robusta (770, 875)", zorder=5)
    ax_heat.axvline(avg_b2, color='orange', linewidth=1.5, linestyle='--', label=f"avg_b2 = {avg_b2:.0f}")
    ax_heat.legend(fontsize=9, loc='upper left')

    # Salva il grafico all'indirizzo specificato
    plt.savefig(r"C:\Desktop\BlackSwan\prosperity_analysis.png", dpi=150, bbox_inches='tight')
    print("\nGrafico salvato: prosperity_analysis.png")

    # 6) Riepilogo sensitività
    print(f"\nSensitività su 500 scenari:")
    for name in ["Nash (750/835)", "Robusta (770/875)", "Ottimale di scenario"]:
        arr = np.array(records[name])
        print(f"  {name:28s}  media={arr.mean():.1f}  std={arr.std():.1f}  min={arr.min():.1f}  max={arr.max():.1f}")

    print("\nFatto.")


if __name__ == "__main__":
    main()

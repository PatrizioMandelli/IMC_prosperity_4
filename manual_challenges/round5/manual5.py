"""
IMC Prosperity Round 5 — Ignith Exchange Manual Trading Strategy
================================================================
Long / short portfolio on all 9 Ignith products.
Both directions count toward the 100 % budget constraint.

Fee (per position):   fee(v%) = (v/100)² × BUDGET  — convex, penalises concentration.
Analytic KKT optimum: a_i* = (|μ_i| − λ) / 2
                       λ = (Σ|μ_i| − 2) / N_active   [Lagrange multiplier]

With L+S available:  theoretical max EV ≈ 167 k  vs  long-only ≈ 71 k  (+135 %).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc="", **__):
            self._it = iterable or []; self._n = 0; self._desc = desc
        def __iter__(self):
            for x in self._it:
                self._n += 1; print(f"\r{self._desc} {self._n}", end="", flush=True); yield x
            print()
        def __enter__(self): return self
        def __exit__(self, *__): print()
        def update(self, n=1): self._n += n; print(f"\r{self._desc} {self._n}", end="", flush=True)
        def set_description(self, s: str): self._desc = s

warnings.filterwarnings("ignore")
plt.style.use("dark_background")


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
BUDGET      = 1_000_000
N_SCENARIOS = 200_000
N_STEPS     = 100
SEED        = 42
OUTPUT_PDF  = "round5-manual.pdf"

GRID_STEP   = 10      # % granularity for magnitude grid (9 assets → 43 k combos)
CEM_ITERS   = 20
CEM_POP     = 400
CEM_ELITE   = 0.15
CEM_SMOOTH  = 0.55
MC_N        = 800
BIASED_N    = 800
FULL_TOP    = 20      # candidates per metric for full-scenario eval
N_QUICK     = 15_000  # scenarios for quick search evaluation
SCATTER_N   = 2000

METRIC_MAX  = ["EV", "Median", "Sharpe", "Sortino", "Win_Rate", "P95", "CVaR_95"]
METRIC_MIN  = ["Std", "Downside_Vol", "VaR_95"]
ALL_METRICS = METRIC_MAX + METRIC_MIN


# ═══════════════════════════════════════════════════════════════════
# PRODUCT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
# ev_return  : expected terminal return from the LONG perspective
#              → positive = bullish (long is optimal)
#              → negative = bearish (short is optimal, effective return = |ev_return|)
# signal     : "long" | "short"  — optimal news-driven direction
# strength   : 1–10 conviction in the signal
PRODUCTS = {
    "THERMALITE_CORE": {
        "name": "Thermalite Core", "signal": "long", "strength": 9,
        "ev_return":  0.45, "return_std": 0.18,
        "model": "trend_gbm",
        "params": {"mu": 0.45, "sigma": 0.15,
                   "bp": 0.20, "bm": 0.10, "bs": 0.04,
                   "mp": 0.08, "mm": 0.08, "ms": 0.04},
        "thesis": (
            "Analyst forecast: users +174 % (1.42M → 3.89M) next quarter;\n"
            "avg daily activity 16 h 42 m — sustained institutional adoption.\n"
            "Beat prob 20 %, miss prob 8 %. Strongest fundamental signal.\n"
            "→ LONG  EV ≈ +45 %   Model: trend GBM + analyst beat/miss tails."
        ),
    },
    "LAVA_CAKE": {
        "name": "Lava Cake", "signal": "short", "strength": 9,
        "ev_return": -0.42, "return_std": 0.18,
        "model": "crash",
        "params": {"pc": 0.90, "mc": -0.45, "sc": 0.12,
                   "pr": 0.10, "mr":  0.05, "sr": 0.05},
        "thesis": (
            "Actual lava confirmed → health halt, civil lawsuits, vendor returns.\n"
            "90 % crash probability. Triple hit: regulatory + legal + reputational.\n"
            "Recovery scenario only 10 % likely (regulators dismiss quickly).\n"
            "→ SHORT EV ≈ +42 %   Model: crash distribution."
        ),
    },
    "MAGMA_INK": {
        "name": "Magma Ink", "signal": "long", "strength": 8,
        "ev_return":  0.30, "return_std": 0.28,
        "model": "lognormal_jump",
        "params": {"mu": 0.30, "sigma": 0.28, "lam": 0.30, "jmu": 0.14, "jsig": 0.07},
        "thesis": (
            "Assembly line destroyed by obsidian blades → manufacturing halt.\n"
            "Demand spike: Lava Fountain Pen limited-edition, 6-hour queues.\n"
            "Supply-demand pincer → strong right tail, positive jumps.\n"
            "→ LONG  EV ≈ +30 %   Model: GBM + compound Poisson upside jumps."
        ),
    },
    "PYROFLEX_CELL": {
        "name": "Pyroflex Cell", "signal": "short", "strength": 7,
        "ev_return": -0.25, "return_std": 0.12,
        "model": "step_shock",
        "params": {"shock": -0.18, "drift": -0.07, "sigma": 0.10},
        "thesis": (
            "50 % tax cut (PCTC) abolished effective tomorrow → levy doubled overnight.\n"
            "Upgrade cycles disrupted; new purchases slowing — mechanical demand shock.\n"
            "Near-certain, low-uncertainty downside. No recovery catalyst visible.\n"
            "→ SHORT EV ≈ +25 %   Model: immediate step-shock + negative drift."
        ),
    },
    "SULFUR": {
        "name": "Sulfur Ltd.", "signal": "long", "strength": 8,
        "ev_return":  0.20, "return_std": 0.10,
        "model": "index_ramp",
        "params": {"prem": 0.20, "priced": 0.30, "ramp": 0.70,
                   "sigma": 0.08, "cp": 0.05, "cr": -0.06},
        "thesis": (
            "Confirmed inclusion in Elemental Index 118 rebalance.\n"
            "Passive funds must buy mechanically — front-loaded price ramp.\n"
            "70 % of premium already priced in; 5 % tail risk of cancellation.\n"
            "→ LONG  EV ≈ +20 %   Model: inclusion ramp + cancellation jump."
        ),
    },
    "OBSIDIAN_CUTLERY": {
        "name": "Obsidian Cutlery", "signal": "short", "strength": 6,
        "ev_return": -0.20, "return_std": 0.22,
        "model": "lognormal_jump",
        "params": {"mu": -0.20, "sigma": 0.22, "lam": 0.15, "jmu": -0.14, "jsig": 0.07},
        "thesis": (
            "Blades destroyed own assembly line; level-1 contamination + evacuation.\n"
            "Factory officials silent — signals severity. Industry-wide concern.\n"
            "Liability + regulatory + downtime = triple negative catalyst.\n"
            "→ SHORT EV ≈ +20 %   Model: GBM + downside Poisson jumps."
        ),
    },
    "SCORIA_PASTE": {
        "name": "Scoria Paste", "signal": "long", "strength": 5,
        "ev_return":  0.18, "return_std": 0.25,
        "model": "pump_decay",
        "params": {"fund": 0.08, "p0mu": 0.25, "p0s": 0.12, "lam": 3.0, "sigma": 0.20},
        "thesis": (
            "Lava D. Ray (BrewTube) urges stockpiling — essential infrastructure.\n"
            "Product called 'the paste that keeps Ignith together'; deep household use.\n"
            "Pump-decay: initial momentum fades exponentially to fundamental.\n"
            "→ LONG  EV ≈ +18 %   Model: pump-decay GBM. Risk: influencer reversal."
        ),
    },
    "VOLCANIC_INCENSE": {
        "name": "Volcanic Incense", "signal": "long", "strength": 3,
        "ev_return":  0.15, "return_std": 0.38,
        "model": "markov",
        "params": {
            "P":   [[0.65, 0.25, 0.10], [0.10, 0.65, 0.25], [0.05, 0.20, 0.75]],
            "mus": [0.006,  0.001, -0.005],
            "sigs":[0.014,  0.009,  0.018],
        },
        "thesis": (
            "Nostralico pump: concentrated buying in windows around appearances.\n"
            "He openly calls for followers to buy — self-reinforcing momentum.\n"
            "Markov regime-switching: momentum → normal → reversal.\n"
            "→ LONG  EV ≈ +15 %   High vol. Risk: 35 % hard reversal. Small alloc."
        ),
    },
    "ASHES_OF_PHOENIX": {
        "name": "Ashes of Phoenix", "signal": "short", "strength": 5,
        "ev_return": -0.10, "return_std": 0.25,
        "model": "bimodal",
        "params": {"pg": 0.40, "mg": 0.05, "sg": 0.09,
                   "pb": 0.60, "mb": -0.18, "sb": 0.22},
        "thesis": (
            "Viral video: phoenix burned alive for cosmetics ashes — public outrage.\n"
            "Company counter: 'birds are immortal'. PR survival uncertain.\n"
            "Bimodal: 40 % PR survives (+5 %), 60 % persistent damage (−18 %).\n"
            "→ SHORT EV ≈ +10 %   High uncertainty. Model: bimodal mixture."
        ),
    },
}

ASSETS  = list(PRODUCTS.keys())
N       = len(ASSETS)                                                        # 9
NAT_DIR = np.array([1 if PRODUCTS[k]["signal"] == "long" else -1 for k in ASSETS])
EFF_MU  = np.array([abs(PRODUCTS[k]["ev_return"]) for k in ASSETS])         # optimal-direction EV


# ═══════════════════════════════════════════════════════════════════
# ANALYTICAL CLOSED-FORM OPTIMUM
# ═══════════════════════════════════════════════════════════════════
def analytical_optimum() -> tuple[np.ndarray, float]:
    """
    Solves:  max Σ_i [x_i·μ_i − x_i²]  s.t. Σx_i = 1, x_i ≥ 0
    where x_i = a_i / 100 (fractional allocation, optimal direction applied).

    KKT conditions give:  x_i* = max(0, (μ_i − λ) / 2)
    λ is found by binary search / iterative dropping of inactive assets.

    Returns (fracs, expected_pnl) where fracs are the optimal fractions (sum ≤ 1).
    """
    mu     = EFF_MU.copy()
    active = np.ones(N, dtype=bool)
    for _ in range(N):
        n_a = active.sum()
        lam = (mu[active].sum() - 2.0) / n_a
        x   = np.where(active, (mu - lam) / 2.0, 0.0)
        if (x[active] >= 0).all():
            break
        active &= (x >= 0)
    x = np.maximum(x, 0.0)
    ev = BUDGET * float(np.sum(x * mu - x ** 2))
    return x, ev


# ═══════════════════════════════════════════════════════════════════
# RETURN SIMULATORS  (all from LONG perspective)
# ═══════════════════════════════════════════════════════════════════
def simulate_product(key: str, n: int, rng: np.random.Generator) -> np.ndarray:
    p = PRODUCTS[key]["params"]
    m = PRODUCTS[key]["model"]
    T = N_STEPS; dt = 1.0 / T

    if m == "lognormal_jump":
        lr = rng.normal(p["mu"] * dt, p["sigma"] * np.sqrt(dt), (n, T)).sum(1)
        nj = rng.poisson(p["lam"], n)
        mj = int(nj.max()) if nj.size and nj.max() > 0 else 0
        if mj:
            aj   = rng.normal(p["jmu"], p["jsig"], (n, mj))
            mask = np.arange(mj) < nj[:, None]
            lr  += (aj * mask).sum(1)
        return np.exp(lr) - 1.0

    if m == "trend_gbm":
        lr   = rng.normal(p["mu"] * dt, p["sigma"] * np.sqrt(dt), (n, T)).sum(1)
        r    = np.exp(lr) - 1.0
        beat = rng.random(n) < p["bp"]
        miss = (~beat) & (rng.random(n) < p["mp"])
        r   += beat * rng.normal(p["bm"], p["bs"], n)
        r   -= miss * np.abs(rng.normal(p["mm"], p["ms"], n))
        return r

    if m == "pump_decay":
        p0 = np.maximum(rng.normal(p["p0mu"], p["p0s"], n), 0.0)
        r  = np.zeros(n)
        for t in range(T):
            r += rng.normal(p["fund"] * dt + p0 * np.exp(-p["lam"] * t / T) * dt,
                            p["sigma"] * np.sqrt(dt), n)
        return r

    if m == "markov":
        P  = np.array(p["P"]); Pc = np.cumsum(P[:, :-1], 1)
        mu = np.array(p["mus"]); sg = np.array(p["sigs"])
        st = np.zeros(n, dtype=int)
        r  = np.zeros(n)
        for _ in range(T):
            r  += rng.normal(mu[st], sg[st])
            st  = (rng.random(n)[:, None] >= Pc[st]).sum(1).clip(0, 2)
        return r

    if m == "index_ramp":
        net = p["prem"] * (1.0 - p["priced"]); rs = int(T * p["ramp"])
        rmu = net / rs if rs else 0.0
        r   = np.zeros(n)
        for t in range(T):
            r += rng.normal(rmu * dt if t < rs else 0.0, p["sigma"] * np.sqrt(dt), n)
        cancel      = rng.random(n) < p["cp"]
        r[cancel]   = rng.normal(p["cr"], 0.04, int(cancel.sum()))
        return r

    if m == "step_shock":
        return p["shock"] + rng.normal(p["drift"], p["sigma"], n)

    if m == "bimodal":
        g = rng.random(n) < p["pg"]
        return np.where(g, rng.normal(p["mg"], p["sg"], n), rng.normal(p["mb"], p["sb"], n))

    if m == "crash":
        c = rng.random(n) < p["pc"]
        return np.where(c, rng.normal(p["mc"], p["sc"], n), rng.normal(p["mr"], p["sr"], n))

    raise ValueError(f"Unknown model: {m}")


def simulate_all(n: int, rng: np.random.Generator) -> np.ndarray:
    """Returns (N, n) array of LONG-perspective returns for all products."""
    return np.stack([simulate_product(k, n, rng) for k in ASSETS])


# ═══════════════════════════════════════════════════════════════════
# FEE MODEL
# ═══════════════════════════════════════════════════════════════════
def fee(v: float) -> float:
    """Quadratic fee: fee(v%) = (v/100)² × BUDGET."""
    return (v / 100.0) ** 2 * BUDGET


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO EVALUATION  (signed allocations)
# ═══════════════════════════════════════════════════════════════════
def batch_pnl(allocs: np.ndarray, ret_mat: np.ndarray) -> np.ndarray:
    """
    allocs  : (P, N) signed integers; Σ|a_i| = 100 per row
              a_i > 0 → long,  a_i < 0 → short,  a_i = 0 → no position
    ret_mat : (N, S) LONG-perspective terminal returns
    returns : (P, S) PnL in SeaShells
              Long PnL  =  a_i/100 · BUDGET · r_i  − fee(|a_i|)
              Short PnL = −a_i/100 · BUDGET · r_i  − fee(|a_i|)
              (unified: signed a_i handles direction automatically)
    """
    gross = (allocs / 100.0) @ (ret_mat * BUDGET)                    # (P, S)
    fees  = ((np.abs(allocs) / 100.0) ** 2 * BUDGET).sum(1)          # (P,)
    return gross - fees[:, None]


def batch_metrics(pnl: np.ndarray) -> dict:
    ev   = pnl.mean(1); med = np.median(pnl, 1); std = pnl.std(1)
    wr   = (pnl > 0).mean(1)
    var5 = np.percentile(pnl,  5, 1)
    p95  = np.percentile(pnl, 95, 1)
    nt   = max(1, int(0.05 * pnl.shape[1]))
    cvar = np.sort(pnl, 1)[:, :nt].mean(1)
    neg  = np.where(pnl < 0, pnl, np.nan)
    dv   = np.where(np.isnan(neg).all(1), 1e-9, np.nanstd(neg, 1))
    sh   = ev / (std + 1e-9); so = ev / (dv + 1e-9)
    c    = pnl - ev[:, None]
    sk   = (c ** 3).mean(1) / (std ** 3 + 1e-9)
    return {"EV": ev, "Median": med, "Std": std, "Downside_Vol": dv,
            "VaR_95": var5, "CVaR_95": cvar, "P95": p95,
            "Sharpe": sh, "Sortino": so, "Win_Rate": wr, "Skewness": sk}


def scalar_metrics(pnl: np.ndarray) -> dict:
    return {k: float(v[0]) for k, v in batch_metrics(pnl[None, :]).items()}


def eval_batch(allocs: np.ndarray, ret_mat: np.ndarray) -> list:
    results = []
    for i in range(0, len(allocs), 500):
        ch  = allocs[i:i + 500]
        pnl = batch_pnl(ch, ret_mat)
        mts = batch_metrics(pnl)
        for j in range(len(ch)):
            results.append(({k: int(ch[j, ki]) for ki, k in enumerate(ASSETS)},
                            {k: float(v[j]) for k, v in mts.items()}))
    return results


# ═══════════════════════════════════════════════════════════════════
# SIMPLEX / ROUNDING HELPERS
# ═══════════════════════════════════════════════════════════════════
def _lr_round(mags: np.ndarray, total: int = 100) -> np.ndarray:
    """Largest-remainder rounding: non-neg floats → non-neg ints summing to total."""
    floors  = np.floor(mags).astype(int)
    deficit = total - floors.sum()
    top     = np.argsort(mags - floors)[::-1]
    for i in range(max(0, int(deficit))):
        floors[top[i]] += 1
    return floors


def make_signed(dirs: np.ndarray, mags: np.ndarray) -> np.ndarray:
    """dirs: (N,) ±1,  mags: (N,) non-neg floats summing ~100 → signed int alloc."""
    return dirs * _lr_round(mags)


def _grid_simplex(n_assets: int, n_units: int):
    """All non-neg integer n-tuples summing to n_units."""
    if n_assets == 1:
        yield (n_units,)
        return
    for k in range(n_units + 1):
        for rest in _grid_simplex(n_assets - 1, n_units - k):
            yield (k,) + rest


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZER  (signed allocations, L+S)
# ═══════════════════════════════════════════════════════════════════
class Optimizer:
    """
    Four-stage pipeline on the signed simplex Σ|a_i|=100:
    1  Grid  — magnitude grid with natural directions from news
    2  MC    — random direction perturbation + Dirichlet magnitudes
    3  CEM   — cross-entropy on Dirichlet α (magnitudes) + direction probs
    4  Biased— neighbourhood Dirichlet around top-80 pool
    5  Full  — re-evaluate top candidates on N_SCENARIOS
    """

    def __init__(self, ret_full: np.ndarray, rng: np.random.Generator):
        self.rng  = rng
        self.rf   = ret_full                                        # (N, N_SCENARIOS)
        idx       = rng.choice(ret_full.shape[1], N_QUICK, replace=False)
        self.rq   = ret_full[:, idx]                               # (N, N_QUICK)
        self.pool: list = []
        self.vecs: list = []

    def _store(self, results, vecs):
        for (ad, m), v in zip(results, vecs):
            self.pool.append({"allocations": ad, "metrics": m})
            self.vecs.append(v)

    def _top_vecs(self, metric: str, n: int) -> np.ndarray:
        maximize = metric in METRIC_MAX
        vals = np.array([r["metrics"][metric] for r in self.pool])
        idx  = np.argsort(vals)[::-1 if maximize else 1][:n]
        return np.stack([self.vecs[i] for i in idx])

    def stage1_grid(self):
        n_units = 100 // GRID_STEP
        combos  = list(_grid_simplex(N, n_units))
        mags    = np.array(combos, dtype=int) * GRID_STEP      # (P, N)
        allocs  = mags * NAT_DIR[None, :]                      # natural directions
        print(f"  [Grid]   {len(allocs):,} portfolios (step={GRID_STEP}%, natural dirs)")
        self._store(eval_batch(allocs, self.rq), allocs)

    def stage2_mc(self):
        vecs = []
        for _ in range(MC_N):
            dirs = NAT_DIR.copy()
            dirs = np.where(self.rng.random(N) < 0.20, -dirs, dirs)   # 20 % flip
            mags = self.rng.dirichlet(np.ones(N)) * 100
            vecs.append(make_signed(dirs, mags))
        mat = np.stack(vecs)
        self._store(eval_batch(mat, self.rq), mat)

    def stage3_cem(self):
        elite_n  = max(4, int(CEM_POP * CEM_ELITE))
        alpha    = np.ones(N)
        dir_prob = ((NAT_DIR + 1) / 2).clip(0.05, 0.95)   # P(long) per asset

        with tqdm(total=CEM_ITERS, desc="  [CEM]   ") as bar:
            for it in range(CEM_ITERS):
                d_s    = (self.rng.random((CEM_POP, N)) < dir_prob) * 2 - 1
                m_s    = self.rng.dirichlet(alpha, CEM_POP) * 100
                vecs   = np.stack([make_signed(d_s[i], m_s[i]) for i in range(CEM_POP)])
                res    = eval_batch(vecs, self.rq)
                self._store(res, vecs)

                evs  = np.array([r[1]["EV"] for r in res])
                top  = np.argsort(evs)[-elite_n:]

                new_alpha    = np.maximum(0.1, np.abs(vecs[top]).astype(float).mean(0) / 10)
                new_dir_prob = (vecs[top] > 0).astype(float).mean(0).clip(0.05, 0.95)
                alpha    = CEM_SMOOTH * new_alpha    + (1 - CEM_SMOOTH) * alpha
                dir_prob = CEM_SMOOTH * new_dir_prob + (1 - CEM_SMOOTH) * dir_prob

                bar.set_description(f"  [CEM it={it+1:02d}] best={evs[top].max():,.0f}")
                bar.update(1)

    def stage4_biased(self):
        top  = self._top_vecs("EV", 80)
        vecs = []
        with tqdm(total=BIASED_N, desc="  [Biased]") as bar:
            for _ in range(BIASED_N):
                base  = top[self.rng.integers(0, len(top))]
                dirs  = np.sign(base).astype(int)
                dirs[dirs == 0] = 1
                alpha = np.maximum(0.1, np.abs(base).astype(float) / 10)
                mags  = self.rng.dirichlet(alpha) * 100
                vecs.append(make_signed(dirs, mags))
                bar.update(1)
        mat = np.stack(vecs)
        self._store(eval_batch(mat, self.rq), mat)

    def stage5_full_eval(self):
        candidates: set = set()
        for metric in ALL_METRICS:
            for v in self._top_vecs(metric, FULL_TOP):
                candidates.add(tuple(v.tolist()))
        mat = np.array(list(candidates), dtype=int)
        print(f"  [Full]   {len(mat)} candidates × {self.rf.shape[1]:,} scenarios")
        res = eval_batch(mat, self.rf)
        full_map = {tuple(v.tolist()): m for v, (_, m) in zip(mat, res)}
        for i, r in enumerate(self.pool):
            key = tuple(self.vecs[i].tolist())
            if key in full_map:
                self.pool[i]["metrics"]    = full_map[key]
                self.pool[i]["_full_eval"] = True

    def run(self) -> list:
        print("[Stage 1] Grid (natural directions, magnitude grid)")
        self.stage1_grid(); print(f"          pool={len(self.pool):,}")
        print("[Stage 2] MC (random direction flip + Dirichlet magnitudes)")
        self.stage2_mc();   print(f"          pool={len(self.pool):,}")
        print("[Stage 3] CEM (Dirichlet α + direction probabilities)")
        self.stage3_cem();  print(f"          pool={len(self.pool):,}")
        print("[Stage 4] Biased MC (Dirichlet neighbourhood of top-80)")
        self.stage4_biased(); print(f"          pool={len(self.pool):,}")
        print("[Stage 5] Full-scenario evaluation")
        self.stage5_full_eval()
        print(f"          full-eval portfolios={sum(r.get('_full_eval',False) for r in self.pool)}")
        return self.pool


# ═══════════════════════════════════════════════════════════════════
# PARETO FRONTIER
# ═══════════════════════════════════════════════════════════════════
def _pareto_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Non-dominated (minimize x, maximize y) mask."""
    dominated = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        if (((xs <= xs[i]) & (ys >= ys[i])) & ((xs < xs[i]) | (ys > ys[i]))).any():
            dominated[i] = True
    return ~dominated


# ═══════════════════════════════════════════════════════════════════
# VISUAL THEME
# ═══════════════════════════════════════════════════════════════════
BG      = "#0d0d1a"
PANEL   = "#1a1a2e"
PANEL2  = "#252540"
HEADER  = "#3a3a5c"
SPINE   = "#555555"
LONG_C  = "#00FF88"    # green  — long positions
SHORT_C = "#FF4444"    # red    — short positions
ACC1    = "#00BFFF"    # blue   — accent
ACC2    = "#FFD700"    # gold   — accent
PALETTE = [LONG_C, SHORT_C, ACC1, ACC2, "#FF69B4", "#ADFF2F", "#FF6030", "#9B59B6", "#1ABC9C"]


def _ax(ax, grid=False):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    if grid:
        ax.grid(True, color=SPINE, lw=0.35, alpha=0.5)


def _fmtk(v: float) -> str:
    return f"{v/1000:+.1f}k" if abs(v) >= 500 else f"{v:+.0f}"


def _tbl(ax, data, cols, fontsize=6.5):
    t = ax.table(cellText=data, colLabels=cols, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(fontsize)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor(HEADER if r == 0 else (PANEL2 if r % 2 == 0 else PANEL))
        cell.set_text_props(color="white", fontweight="bold" if r == 0 else "normal")
        cell.set_edgecolor(SPINE)
    t.scale(1, 1.4)


def _full_pool(pool: list) -> list:
    fp = [r for r in pool if r.get("_full_eval")]
    return fp if fp else pool


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — COVER
# ═══════════════════════════════════════════════════════════════════
def _page_cover(pdf, pool: list):
    fp    = _full_pool(pool)
    best  = fp[int(np.argmax([r["metrics"]["Sharpe"] for r in fp]))]
    m, a  = best["metrics"], best["allocations"]
    x_ana, ev_ana = analytical_optimum()

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")

    lines = [
        "IMC PROSPERITY  ·  ROUND 5  ·  IGNITH EXCHANGE",
        f"scenarios={N_SCENARIOS:,}   seed={SEED}   budget={BUDGET:,}",
        f"fee = (v/100)²×BUDGET  |  LONG+SHORT, Σ|alloc|=100 %",
        "",
        "━━ ALL 9 PRODUCTS  (direction | EV | conviction) ━━━━━━━━━━━━━━━",
    ]
    for k in ASSETS:
        p   = PRODUCTS[k]
        dir_sym = "▲ LONG " if p["signal"] == "long" else "▼ SHORT"
        eff_ev  = abs(p["ev_return"]) * 100
        bar     = "▉" * p["strength"]
        lines.append(
            f"  {dir_sym}  {p['name']:<22s}  EV={eff_ev:+5.1f}%"
            f"  [{bar:<10s}] {p['strength']}/10"
        )
    lines += [
        "",
        "━━ ANALYTICAL KKT OPTIMUM  (closed-form, unconstrained integer rounding) ━",
    ]
    for i, k in enumerate(ASSETS):
        pct = round(x_ana[i] * 100)
        if pct == 0: continue
        p = PRODUCTS[k]
        bar = "█" * (pct // 2) + ("▌" if pct % 2 else "")
        lines.append(f"  {'L' if p['signal']=='long' else 'S'}  {p['name']:<22s}  {pct:>3d}%  {bar}")
    lines += [f"  Theoretical EV = {_fmtk(ev_ana)} SeaShells  (unconstrained analytic)"]
    lines += [
        "",
        "━━ OPTIMIZER RESULT  (Max Sharpe, Full-Eval) ━━━━━━━━━━━━━━━━━━━",
    ]
    for k in ASSETS:
        av = a.get(k, 0)
        if av == 0: continue
        p = PRODUCTS[k]; col_sym = "L" if av > 0 else "S"
        bar = "█" * (abs(av) // 2) + ("▌" if abs(av) % 2 else "")
        lines.append(f"  {col_sym}  {p['name']:<22s}  {av:>+4d}%  {bar}")
    total_fee = sum(fee(abs(a.get(k, 0))) for k in ASSETS)
    lines += [
        f"  {'─'*55}",
        f"  Σ|alloc|={sum(abs(v) for v in a.values())}%   Total fees = {_fmtk(total_fee)}",
        f"  EV={_fmtk(m['EV'])}   Sharpe={m['Sharpe']:.3f}   Sortino={m['Sortino']:.3f}",
        f"  Win Rate={m['Win_Rate']:.1%}   VaR95={_fmtk(m['VaR_95'])}",
    ]
    ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
            fontsize=9.5, color="white", va="center", ha="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=PANEL2, alpha=0.93))
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — FEE ANALYSIS  (6 panels)
# ═══════════════════════════════════════════════════════════════════
def _page_fee_analysis(pdf):
    """
    Deep-dive on fee(v) = (v/100)² × B.

    Mathematical properties:
    • Quadratic (convex) in v  → every % added is more expensive than the last
    • Marginal fee: d(fee)/dv = 2v/100² × B  → at v=10%: 200 SeaShells/%;
      at v=20%: 400 SeaShells/%  (doubles with position size)
    • Standalone optimal: d(NetEV)/dv = 0  →  v* = 50μ  (in %)
    • Max standalone net EV = B × μ²/4  (achievable per asset)
    • Break-even condition: need |μ| > 2a  (return must exceed 2× allocation fraction)
    • Constrained Lagrangian: a_i* = (μ_i − λ)/2,  λ=(Σμ_i − 2)/N
    """
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.suptitle(
        "Fee Cost Function Analysis  —  fee(v%) = (v/100)² × 1,000,000  (quadratic, convex)",
        fontsize=13, color=ACC2, fontweight="bold"
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38,
                  left=0.07, right=0.97, top=0.90, bottom=0.07)

    vv = np.linspace(0, 100, 500)

    # ── Panel 1: Fee curve + marginal fee ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); _ax(ax, grid=True)
    ax.plot(vv, (vv / 100) ** 2 * BUDGET / 1000, color=ACC2, lw=2, label="Total fee (k)")
    ax2 = ax.twinx()
    ax2.set_facecolor(PANEL)
    ax2.plot(vv, 2 * vv / 100 ** 2 * BUDGET / 1000, color=SHORT_C, lw=1.5, ls="--",
             label="Marginal fee (k/1%)")
    ax2.tick_params(colors="white", labelsize=7)
    ax2.set_ylabel("Marginal fee (k / 1% alloc)", color=SHORT_C, fontsize=8)
    ax.set_xlabel("Allocation v (%)", color="white"); ax.set_ylabel("Fee (k SeaShells)", color="white")
    ax.set_title("Fee Curve + Marginal Cost", color="white", fontsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, facecolor=PANEL, labelcolor="white", fontsize=7)
    ax.text(0.97, 0.15,
            "fee(v)=(v/100)²×B\nd(fee)/dv = 2v/B×10⁻⁴\n→ convex: each % costs more",
            transform=ax.transAxes, color="#CCC", fontsize=6.5, ha="right",
            bbox=dict(facecolor=PANEL2, alpha=0.8, boxstyle="round"))

    # ── Panel 2: Net EV curves per product (standalone optimal) ──────────────
    ax = fig.add_subplot(gs[0, 1]); _ax(ax, grid=True)
    for i, k in enumerate(ASSETS):
        p     = PRODUCTS[k]
        mu    = abs(p["ev_return"])
        net   = vv / 100 * BUDGET * mu - (vv / 100) ** 2 * BUDGET
        v_opt = 50 * mu
        color = LONG_C if p["signal"] == "long" else SHORT_C
        ax.plot(vv, net / 1000, color=color, lw=1.4, alpha=0.8, label=p["name"][:12])
        ax.axvline(v_opt, color=color, lw=0.6, ls=":", alpha=0.5)
    ax.axhline(0, color="white", lw=0.7, ls=":")
    ax.set_xlabel("Allocation v (%)", color="white")
    ax.set_ylabel("Net EV (k SeaShells)", color="white")
    ax.set_title("Net EV(v) per Product  [standalone]", color="white", fontsize=9)
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=5.5, ncol=2)

    # ── Panel 3: Optimal standalone alloc vs |μ|  (linear v* = 50μ) ─────────
    ax = fig.add_subplot(gs[0, 2]); _ax(ax, grid=True)
    mu_arr    = np.linspace(0, 0.6, 200)
    ax.plot(mu_arr * 100, 50 * mu_arr * 100, color=ACC1, lw=2, label="v* = 50·|μ| (analytic)")
    ax.fill_between(mu_arr * 100, 0, 50 * mu_arr * 100, alpha=0.15, color=ACC1)  # type: ignore[arg-type]
    for i, k in enumerate(ASSETS):
        p     = PRODUCTS[k]
        mu    = abs(p["ev_return"])
        v_opt = 50 * mu * 100
        color = LONG_C if p["signal"] == "long" else SHORT_C
        ax.scatter([mu * 100], [v_opt], color=color, s=60, zorder=5)
        ax.text(mu * 100 + 0.4, v_opt + 0.6, p["name"][:8], color=color, fontsize=6)
    ax.axvline(0, color="white", lw=0.5); ax.axhline(0, color="white", lw=0.5)
    ax.set_xlabel("|EV return| (%)", color="white"); ax.set_ylabel("Optimal standalone alloc (%)", color="white")
    ax.set_title("Standalone Optimum: v* = 50·|μ|", color="white", fontsize=9)
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)

    # ── Panel 4: KKT constrained vs standalone optimal ────────────────────────
    ax = fig.add_subplot(gs[1, 0]); _ax(ax, grid=True)
    x_ana, ev_ana = analytical_optimum()
    standalone    = 50 * EFF_MU   # unconstrained in %
    constrained   = x_ana * 100   # KKT with Σ=100% constraint
    names         = [PRODUCTS[k]["name"][:10] for k in ASSETS]
    x_pos         = np.arange(N)
    colors        = [LONG_C if PRODUCTS[k]["signal"] == "long" else SHORT_C for k in ASSETS]
    bars_s = ax.bar(x_pos - 0.2, standalone, 0.35, color=[c + "80" for c in colors], label="Standalone v*")
    bars_c = ax.bar(x_pos + 0.2, constrained, 0.35, color=colors, label="KKT constrained")
    for i, (s, c, col) in enumerate(zip(standalone, constrained, colors)):
        ax.annotate("", xy=(x_pos[i] + 0.2, c + 0.3), xytext=(x_pos[i] - 0.2, s + 0.3),
                    arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
    ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("Allocation (%)", color="white")
    ax.set_title("Standalone vs KKT-Constrained Optimum", color="white", fontsize=9)
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)
    ax.text(0.02, 0.97, f"KKT EV = {_fmtk(ev_ana)}\nλ = {(EFF_MU.sum()-2)/N:.4f}",
            transform=ax.transAxes, color=ACC2, fontsize=7, va="top",
            bbox=dict(facecolor=PANEL2, alpha=0.8, boxstyle="round"))

    # ── Panel 5: Fee-efficiency table by product ──────────────────────────────
    ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor(PANEL); ax.axis("off")
    rows = []
    for i, k in enumerate(ASSETS):
        p      = PRODUCTS[k]
        mu     = abs(p["ev_return"])
        v_opt  = 50 * mu
        v_kkt  = x_ana[i] * 100
        gross  = v_opt / 100 * BUDGET * mu
        f_opt  = fee(v_opt)
        net    = gross - f_opt
        eff    = net / gross * 100 if gross > 0 else 0
        rows.append([
            p["name"][:14],
            "L" if p["signal"] == "long" else "S",
            f"{mu*100:.0f}%",
            f"{v_opt:.1f}%",
            f"{v_kkt:.1f}%",
            _fmtk(gross),
            _fmtk(f_opt),
            f"{eff:.0f}%",
        ])
    _tbl(ax, rows, ["Product", "Dir", "|μ|", "v*SA", "v*KKT", "Gross", "Fee", "Neteff"],
         fontsize=6.0)
    ax.set_title("Fee Efficiency per Product at Standalone v*", color="white", fontsize=8, pad=5)

    # ── Panel 6: Diversification dividend ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2]); _ax(ax, grid=True)
    # Compare: all-in single best asset vs spreading across k assets
    sorted_mu = np.sort(EFF_MU)[::-1]
    ev_1 = np.array([BUDGET * ((sorted_mu[:k].sum() / 100) * sorted_mu[:k].mean()
                               - (sorted_mu[:k].sum() / 100) ** 2)
                     for k in range(1, N + 1)])
    # KKT optimal EV with k assets
    ev_kkt = []
    for k in range(1, N + 1):
        mu_k = sorted_mu[:k]
        lam_k = (mu_k.sum() - 2.0) / k
        x_k   = np.maximum((mu_k - lam_k) / 2.0, 0.0)
        ev_kkt.append(BUDGET * float((x_k * mu_k - x_k ** 2).sum()))

    ax.plot(range(1, N + 1), np.array(ev_1) / 1000, color=SHORT_C, lw=2, marker="o",
            ms=5, label="Equal-split naive")
    ax.plot(range(1, N + 1), np.array(ev_kkt) / 1000, color=LONG_C, lw=2, marker="s",
            ms=5, label="KKT optimal")
    ax.axhline(ev_ana / 1000, color=ACC2, lw=1, ls="--", label=f"Max={_fmtk(ev_ana)}")
    ax.set_xlabel("Number of assets included (sorted by |μ|)", color="white")
    ax.set_ylabel("Expected Net PnL (k SeaShells)", color="white")
    ax.set_title("Diversification Dividend (adding assets)", color="white", fontsize=9)
    ax.set_xticks(range(1, N + 1))
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)

    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — ALL PRODUCTS OVERVIEW  (3×3 grid, return distributions)
# ═══════════════════════════════════════════════════════════════════
def _page_products_overview(pdf, ret_mat: np.ndarray):
    """3×3 mini-histograms with thesis snippets."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), facecolor=BG)
    fig.suptitle("All 9 Ignith Products — Return Distributions (path simulation)",
                 fontsize=13, color="white", fontweight="bold")

    for ax, k in zip(axes.flatten(), ASSETS):
        p     = PRODUCTS[k]
        idx   = ASSETS.index(k)
        r     = ret_mat[idx]
        color = LONG_C if p["signal"] == "long" else SHORT_C
        dir_s = "LONG ▲" if p["signal"] == "long" else "SHORT ▼"

        _ax(ax, grid=True)
        ax.hist(r * 100, bins=100, color=color, alpha=0.65, density=True)
        ax.axvline(r.mean() * 100, color="white", lw=1.8, ls="--",
                   label=f"EV={r.mean()*100:+.1f}%")
        ax.axvline(0, color="#888", lw=0.7, ls=":")

        # effective EV from optimal direction
        eff = (abs(p["ev_return"]) if p["signal"]=="long" else abs(p["ev_return"])) * 100
        v_opt = 50 * abs(p["ev_return"])
        max_ev = BUDGET * (abs(p["ev_return"])) ** 2 / 4

        ax.set_title(f"{p['name']}  [{dir_s}  {p['strength']}/10]",
                     color=color, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Return %  (long perspective)", color="white", fontsize=7)
        ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)
        ax.text(0.03, 0.98,
                f"v* = {v_opt:.0f}%   MaxEV = {_fmtk(max_ev)}\nStd = {r.std()*100:.0f}%\n{p['model']}",
                transform=ax.transAxes, fontsize=6.5, color="#CCC", va="top",
                bbox=dict(facecolor=PANEL2, alpha=0.75, boxstyle="round,pad=0.3"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — PRODUCT THESIS (3 per page × 3 pages)
# ═══════════════════════════════════════════════════════════════════
def _page_product_detail(pdf, keys: list, ret_mat: np.ndarray):
    n = len(keys)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5 * n), facecolor=BG)
    if n == 1:
        axes = axes[None, :]

    for row, k in enumerate(keys):
        p     = PRODUCTS[k]
        idx   = ASSETS.index(k)
        r     = ret_mat[idx]
        color = LONG_C if p["signal"] == "long" else SHORT_C
        mu    = abs(p["ev_return"])

        # ── Col 0: return distribution ─────────────────────────────────────
        ax = axes[row, 0]; _ax(ax, grid=True)
        ax.hist(r * 100, bins=120, color=color, alpha=0.65, density=True)
        ax.axvline(r.mean() * 100, color="white", lw=1.8, ls="--",
                   label=f"Sim EV={r.mean()*100:+.1f}%")
        ax.axvline(p["ev_return"] * 100, color=ACC2, lw=1.2, ls=":",
                   label=f"Model EV={p['ev_return']*100:+.1f}%")
        ax.axvline(0, color="#888", lw=0.6, ls=":")
        q5, q95 = np.percentile(r, 5) * 100, np.percentile(r, 95) * 100
        ax.axvspan(q5, q95, alpha=0.07, color=color)
        ax.set_title(f"{'▲' if p['signal']=='long' else '▼'}  {p['name']}  [{p['signal'].upper()} {p['strength']}/10]",
                     color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("Return % (long perspective)", color="white")
        ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7.5)

        # ── Col 1: post-fee net EV curve ────────────────────────────────────
        ax = axes[row, 1]; _ax(ax, grid=True)
        vv = np.arange(0, 101)
        gross_arr = vv / 100 * BUDGET * mu
        fee_arr   = (vv / 100) ** 2 * BUDGET
        net_arr   = gross_arr - fee_arr
        v_opt     = 50 * mu
        ax.plot(vv, gross_arr / 1000, color=ACC1, lw=1.5, ls="--", label="Gross EV")
        ax.plot(vv, fee_arr  / 1000,  color=SHORT_C, lw=1.5, ls=":", label="Fee")
        ax.plot(vv, net_arr  / 1000,  color=color, lw=2.0, label="Net EV")
        ax.axvline(v_opt, color=color, lw=1.2, ls="--",
                   label=f"v*={v_opt:.0f}%  EV={_fmtk(BUDGET*mu**2/4)}")
        ax.axhline(0, color="white", lw=0.6, ls=":")
        ax.fill_between(vv, net_arr / 1000, 0,
                        where=(net_arr > 0), alpha=0.12, color=color)
        be = 100 * mu * 2    # break-even: v < 100μ (where gross > fee, slope positive)
        if be <= 100:
            ax.axvline(be, color=ACC2, lw=0.8, ls=":", label=f"Break-even v={be:.0f}%")
        ax.set_xlabel("Allocation (%)", color="white"); ax.set_ylabel("SeaShells (k)", color="white")
        ax.set_title("Post-Fee Net EV (standalone)", color="white", fontsize=9)
        ax.legend(facecolor=PANEL, labelcolor="white", fontsize=6.5)

        # ── Col 2: percentile curve + thesis ───────────────────────────────
        ax = axes[row, 2]; ax.set_facecolor(PANEL); ax.axis("off")
        pct_labels = ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"]
        pct_vals   = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        qs         = np.percentile(r, pct_vals) * 100
        text = (
            f"Signal: {'LONG ▲' if p['signal']=='long' else 'SHORT ▼'}  [{p['strength']}/10]\n"
            f"EV (long): {p['ev_return']*100:+.1f}%   EV (optimal dir): +{mu*100:.1f}%\n"
            f"Std: {r.std()*100:.1f}%   Model: {p['model']}\n"
            f"Standalone v* = {v_opt:.0f}%   Max net EV = {_fmtk(BUDGET*mu**2/4)}\n\n"
        )
        for lbl, q in zip(pct_labels, qs):
            text += f"  {lbl}: {q:+.1f}%\n"
        text += f"\n──── THESIS ────\n{p['thesis']}"
        ax.text(0.04, 0.98, text, transform=ax.transAxes,
                fontsize=7.2, color="white", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL2, alpha=0.85))
        ax.set_title("Percentile Summary + Thesis", color="white", fontsize=9, pad=5)

    plt.tight_layout(rect=[0, 0, 1, 1])
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — EFFICIENT FRONTIERS
# ═══════════════════════════════════════════════════════════════════
def _page_frontier(pdf, pool: list, x_metric: str, x_label: str,
                   color_metric: str, color_label: str, title: str):
    fp   = _full_pool(pool)
    evs  = np.array([r["metrics"]["EV"]          for r in fp]) / 1000
    xs   = np.array([r["metrics"][x_metric]       for r in fp]) / 1000
    cols = np.array([r["metrics"][color_metric]   for r in fp])

    fig, (ax_main, ax_dir) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle(title, fontsize=12, color="white", fontweight="bold")

    # ── Main scatter + Pareto ─────────────────────────────────────────────
    _ax(ax_main, grid=True)
    rng_sc = np.random.default_rng(7)
    n_show = min(SCATTER_N, len(evs))
    idx_s  = rng_sc.choice(len(evs), n_show, replace=False)
    norm   = Normalize(vmin=np.percentile(cols, 5), vmax=np.percentile(cols, 95))
    sc = ax_main.scatter(xs[idx_s], evs[idx_s], c=cols[idx_s], norm=norm,
                         cmap="plasma", s=14, alpha=0.55)
    cb = plt.colorbar(sc, ax=ax_main)
    cb.set_label(color_label, color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.ax.yaxis.set_tick_params(color="white")

    fe_mask = _pareto_mask(xs, evs)
    fi      = np.where(fe_mask)[0]
    fi_s    = fi[np.argsort(xs[fi])]
    ax_main.plot(xs[fi_s], evs[fi_s], color=ACC1, lw=2, label="Pareto", zorder=6)
    for mk, mc, lbl, i in [("*", "cyan", "Max "+color_metric, int(np.argmax(cols))),
                            ("D", LONG_C, "Max EV", int(np.argmax(evs)))]:
        ax_main.scatter(xs[i], evs[i], marker=mk, s=160, color=mc, zorder=10, label=lbl)
    ax_main.set_xlabel(f"{x_label} (k)", color="white")
    ax_main.set_ylabel("Expected PnL (k)", color="white")
    ax_main.legend(facecolor=PANEL, labelcolor="white", fontsize=8)

    # ── Long/short breakdown of best portfolios ────────────────────────────
    _ax(ax_dir)
    top5 = [fp[i] for i in np.argsort([r["metrics"]["EV"] for r in fp])[-5:]][::-1]
    bar_w = 0.15; x_pos = np.arange(N)
    for rank, res in enumerate(top5):
        av  = [res["allocations"].get(k, 0) for k in ASSETS]
        col = PALETTE[rank]
        label = f"#{rank+1} EV={_fmtk(res['metrics']['EV'])}"
        ax_dir.bar(x_pos + rank * bar_w, av, bar_w, color=col, alpha=0.85, label=label)
    ax_dir.axhline(0, color="white", lw=0.8)
    ax_dir.set_xticks(x_pos + bar_w * 2)
    ax_dir.set_xticklabels([PRODUCTS[k]["name"][:9] for k in ASSETS],
                            rotation=30, ha="right", fontsize=7)
    ax_dir.set_ylabel("Allocation % (+long / −short)", color="white")
    ax_dir.set_title("Top-5 EV Portfolios — Signed Allocations", color="white", fontsize=9)
    ax_dir.legend(facecolor=PANEL, labelcolor="white", fontsize=7)
    ax_dir.text(0.02, 0.02, "Green bars = LONG\nRed bars = SHORT",
                transform=ax_dir.transAxes, color="#CCC", fontsize=7,
                bbox=dict(facecolor=PANEL2, alpha=0.8, boxstyle="round"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — TOP PORTFOLIOS TABLE
# ═══════════════════════════════════════════════════════════════════
def _page_table(pdf, pool: list):
    fp = _full_pool(pool)
    rows_data = []
    for r in fp:
        m, a = r["metrics"], r["allocations"]
        rows_data.append({**m, **{k: a.get(k, 0) for k in ASSETS}})

    df  = (pd.DataFrame(rows_data).sort_values("EV", ascending=False)
             .head(25).reset_index(drop=True))
    money   = ["EV", "Median", "VaR_95", "CVaR_95"]
    ratios  = ["Sharpe", "Sortino", "Win_Rate"]
    acols   = ASSETS
    short_n = {k: PRODUCTS[k]["name"][:7] for k in ASSETS}
    cols    = money + ratios + acols

    def _fmt(c, v):
        if c in money:   return _fmtk(v)
        if c == "Win_Rate": return f"{v:.1%}"
        if c in ratios:  return f"{v:.2f}"
        return f"{int(v):+d}"

    td    = [[_fmt(c, row[c]) for c in cols] for _, row in df[cols].iterrows()]
    hdrs  = money + ratios + [short_n[k] for k in ASSETS]

    fig, ax = plt.subplots(figsize=(24, 14), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")
    fig.suptitle("Top 25 Portfolios by EV — Full-Eval  (signed: +long / −short)",
                 fontsize=12, color="white", fontweight="bold")
    _tbl(ax, td, hdrs, fontsize=5.8)
    pdf.savefig(fig, facecolor=BG, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — BEST PORTFOLIO DEEP-DIVE (3 portfolios)
# ═══════════════════════════════════════════════════════════════════
def _page_deep_dive(pdf, pool: list, ret_mat: np.ndarray):
    fp   = _full_pool(pool)
    bests = [
        (fp[int(np.argmax([r["metrics"]["Sharpe"]  for r in fp]))], "Max Sharpe",  ACC1),
        (fp[int(np.argmax([r["metrics"]["EV"]      for r in fp]))], "Max EV",      LONG_C),
        (fp[int(np.argmax([r["metrics"]["Sortino"] for r in fp]))], "Max Sortino", ACC2),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), facecolor=BG)
    fig.suptitle("Best Portfolio Deep-Dive (Full-Eval, signed L/S allocations)",
                 fontsize=12, color="white", fontweight="bold")

    rng_dd = np.random.default_rng(SEED + 200)

    for row_i, (result, title, col) in enumerate(bests):
        m, a = result["metrics"], result["allocations"]

        # Reconstruct PnL from simulated returns
        alloc_vec = np.array([a.get(k, 0) for k in ASSETS])
        pnl_disp  = batch_pnl(alloc_vec[None, :], ret_mat)[0]

        # ── Col 0: PnL distribution ────────────────────────────────────────
        ax = axes[row_i, 0]; _ax(ax, grid=True)
        ax.hist(pnl_disp / 1000, bins=130, color=col, alpha=0.72, density=True)
        ax.axvline(m["EV"]     / 1000, color="white",  lw=1.8, ls="--", label=f"EV={_fmtk(m['EV'])}")
        ax.axvline(m["VaR_95"] / 1000, color=SHORT_C,  lw=1.4, ls=":", label=f"VaR5%={_fmtk(m['VaR_95'])}")
        ax.axvline(0, color="#888", lw=0.6, ls=":")
        ax.set_title(f"{title} — PnL Distribution", color=col, fontsize=9)
        ax.set_xlabel("PnL (k SeaShells)", color="white")
        ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)
        ax.text(0.97, 0.97,
                f"Sharpe  = {m['Sharpe']:.2f}\nSortino = {m['Sortino']:.2f}\n"
                f"Win     = {m['Win_Rate']:.1%}\nFee %   = {sum((abs(a.get(k,0))/100)**2 for k in ASSETS)*100:.1f}",
                transform=ax.transAxes, color="white", fontsize=7.5,
                ha="right", va="top", bbox=dict(facecolor=PANEL2, alpha=0.85, boxstyle="round"))

        # ── Col 1: Signed allocation bar chart ────────────────────────────
        ax = axes[row_i, 1]; _ax(ax)
        names  = [PRODUCTS[k]["name"] for k in ASSETS]
        vals   = [a.get(k, 0) for k in ASSETS]
        bar_colors = [LONG_C if v > 0 else (SHORT_C if v < 0 else SPINE) for v in vals]
        bars = ax.barh(names, vals, color=bar_colors, alpha=0.85)
        ax.axvline(0, color="white", lw=0.8)
        for bar, v in zip(bars, vals):
            if v != 0:
                ax.text(v + (0.5 if v > 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                        f"{v:+d}%", va="center", ha="left" if v > 0 else "right",
                        color="white", fontsize=8.5)
        ax.set_xlabel("Allocation % (+long / −short)", color="white")
        ax.set_title(f"{title} — Signed Allocations", color=col, fontsize=9)
        ax.set_xlim(-55, 55)

        # ── Col 2: Percentile / CDF comparison ────────────────────────────
        ax = axes[row_i, 2]; _ax(ax, grid=True)
        pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        qs = np.percentile(pnl_disp / 1000, pctiles)
        ax.plot(pctiles, qs, color=col, lw=2, marker="o", ms=5)
        ax.fill_between(pctiles, qs, 0, where=(np.array(qs) > 0), alpha=0.15, color=LONG_C)
        ax.fill_between(pctiles, qs, 0, where=(np.array(qs) < 0), alpha=0.15, color=SHORT_C)
        ax.axhline(0, color="white", lw=0.7, ls=":")
        for i, (p_, q) in enumerate(zip(pctiles, qs)):
            ax.text(p_, q + (0.5 if q >= 0 else -1.0), f"{q:+.0f}k", color="white", fontsize=6.5)
        ax.set_xlabel("Percentile", color="white"); ax.set_ylabel("PnL (k)", color="white")
        ax.set_title(f"{title} — Percentile Curve", color=col, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PDF PAGE — FEE OPTIMIZATION STUDY (detailed)
# ═══════════════════════════════════════════════════════════════════
def _page_fee_optimization_study(pdf, pool: list):
    """
    Detailed study of how the fee structure shapes portfolio construction.

    Key mathematical results:
    ──────────────────────────────────────────────────────────────────
    1. Single-asset optimal:  v*(%) = 50 × |μ|
       (from dNetEV/dv = 0  ⟹  μ = 2v/100)

    2. Max per-asset EV:  B × μ²/4
       (substituting v* back)

    3. Break-even allocation: v_be = 100 × μ  (% where net EV = 0)

    4. Constrained optimum (KKT, Σa_i = 100%):
       a_i* = (μ_i − λ) / 2,  λ = (Σμ_i − 2) / N_active

    5. Fee drag at optimum:  fee* = B × μ²/4  (equals net EV!)
       i.e. at v*, fee = gross/2  — you keep exactly half the gross PnL.

    6. Portfolio fee budget:  total_fee = Σ fee(a_i*)
       Under KKT: total_fee = B × Σ [(μ_i−λ)²/4]
    ──────────────────────────────────────────────────────────────────
    """
    fp  = _full_pool(pool)
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.suptitle("Fee Optimization Study — Impact of Quadratic Cost on Portfolio Construction",
                 fontsize=12, color=ACC2, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40,
                  left=0.07, right=0.97, top=0.90, bottom=0.08)

    # ── Panel 1: fee* = gross/2 at optimal allocation ─────────────────────
    ax = fig.add_subplot(gs[0, 0]); _ax(ax, grid=True)
    mu_range = np.linspace(0.05, 0.60, 300)
    v_opt    = 50 * mu_range
    gross_at_opt = v_opt / 100 * BUDGET * mu_range
    fee_at_opt   = (v_opt / 100) ** 2 * BUDGET
    net_at_opt   = gross_at_opt - fee_at_opt
    ax.plot(mu_range * 100, gross_at_opt / 1000, color=ACC1,   lw=2, label="Gross EV at v*")
    ax.plot(mu_range * 100, fee_at_opt   / 1000, color=SHORT_C, lw=2, label="Fee at v*")
    ax.plot(mu_range * 100, net_at_opt   / 1000, color=LONG_C,  lw=2, label="Net EV at v*")
    ax.fill_between(mu_range * 100, fee_at_opt / 1000, gross_at_opt / 1000,  # type: ignore[arg-type]
                    alpha=0.10, color=LONG_C, label="Net EV region")
    ax.fill_between(mu_range * 100, 0, fee_at_opt / 1000,  # type: ignore[arg-type]
                    alpha=0.10, color=SHORT_C, label="Fee region")
    ax.set_xlabel("|μ| (%)", color="white"); ax.set_ylabel("SeaShells (k)", color="white")
    ax.set_title("At v*: Fee = Gross/2 (always)", color="white", fontsize=9)
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=6.5)
    ax.text(0.05, 0.97,
            "fee* = B·μ²/4\ngross* = B·μ²/2\nnet* = B·μ²/4\n⟹ fee = gross exactly",
            transform=ax.transAxes, color=ACC2, fontsize=7, va="top",
            bbox=dict(facecolor=PANEL2, alpha=0.85, boxstyle="round"))

    # ── Panel 2: Penalty for over- / under-allocation vs v* ──────────────
    ax = fig.add_subplot(gs[0, 1]); _ax(ax, grid=True)
    ref_mu = 0.30  # example: Magma Ink
    v_star = 50 * ref_mu
    vv     = np.linspace(1, 80, 400)
    net_ev = vv / 100 * BUDGET * ref_mu - (vv / 100) ** 2 * BUDGET
    net_opt = v_star / 100 * BUDGET * ref_mu - (v_star / 100) ** 2 * BUDGET
    penalty = net_ev - net_opt   # always ≤ 0
    ax.plot(vv, penalty / 1000, color=SHORT_C, lw=2)
    ax.fill_between(vv, penalty / 1000, 0, alpha=0.20, color=SHORT_C)  # type: ignore[arg-type]
    ax.axvline(v_star, color=LONG_C, lw=1.5, ls="--", label=f"v*={v_star:.0f}%")
    ax.axhline(0, color="white", lw=0.7, ls=":")
    ax.set_xlabel("Allocation v (%)", color="white"); ax.set_ylabel("EV Penalty vs Optimal (k)", color="white")
    ax.set_title(f"Cost of Mis-Allocation  (μ={ref_mu:.0%})", color="white", fontsize=9)
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)
    ax.text(0.97, 0.05,
            f"Max EV = {_fmtk(net_opt)}\n±5% alloc penalty:\n"
            f"  v*-5={_fmtk(((v_star-5)/100*BUDGET*ref_mu-(v_star-5)**2/10000*BUDGET)-net_opt)}\n"
            f"  v*+5={_fmtk(((v_star+5)/100*BUDGET*ref_mu-(v_star+5)**2/10000*BUDGET)-net_opt)}",
            transform=ax.transAxes, color="#CCC", fontsize=7, ha="right", va="bottom",
            bbox=dict(facecolor=PANEL2, alpha=0.85, boxstyle="round"))

    # ── Panel 3: Total fees paid vs expected EV (pool scatter) ────────────
    ax = fig.add_subplot(gs[0, 2]); _ax(ax, grid=True)
    fees_all = np.array([sum(fee(abs(r["allocations"].get(k, 0))) for k in ASSETS) for r in fp])
    evs_all  = np.array([r["metrics"]["EV"] for r in fp])
    fee_pct  = fees_all / BUDGET * 100
    sharpes  = np.array([r["metrics"]["Sharpe"] for r in fp])
    sc = ax.scatter(fee_pct, evs_all / 1000, c=sharpes,
                    cmap="RdYlGn", s=10, alpha=0.55,
                    vmin=float(np.percentile(sharpes, 5)), vmax=float(np.percentile(sharpes, 95)))
    cb = plt.colorbar(sc, ax=ax); cb.set_label("Sharpe", color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    ax.axhline(0, color="white", lw=0.7, ls=":")
    ax.set_xlabel("Total Fees (% of Budget)", color="white")
    ax.set_ylabel("Expected Net PnL (k)", color="white")
    ax.set_title("Fee Drag vs Portfolio EV (Full-Eval Pool)", color="white", fontsize=9)
    # Mark analytical optimum
    x_ana, ev_ana = analytical_optimum()
    fee_ana = sum(fee(x_ana[i] * 100) for i in range(N)) / BUDGET * 100
    ax.scatter([fee_ana], [ev_ana / 1000], marker="*", s=200, color=ACC2, zorder=10,  # type: ignore[arg-type]
               label=f"KKT opt  EV={_fmtk(ev_ana)}")
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)

    # ── Panel 4: Lagrange multiplier interpretation ───────────────────────
    ax = fig.add_subplot(gs[1, 0]); _ax(ax, grid=True)
    n_active = np.arange(1, N + 1)
    sum_mu   = np.sort(EFF_MU)[::-1].cumsum()
    lambdas  = (sum_mu - 2.0) / n_active
    ax.plot(n_active, lambdas, color=ACC2, lw=2, marker="o", ms=6)
    ax.axhline(0, color="white", lw=0.7, ls=":")
    ax.fill_between(n_active, lambdas, 0, where=(lambdas > 0), alpha=0.15, color=SHORT_C,
                    label="λ>0: budget constraint binding")
    ax.fill_between(n_active, lambdas, 0, where=(lambdas <= 0), alpha=0.15, color=LONG_C,
                    label="λ<0: budget not exhausted")
    ax.set_xlabel("Number of assets (sorted by |μ|, descending)", color="white")
    ax.set_ylabel("Lagrange multiplier λ", color="white")
    ax.set_title("λ vs Assets Included  (KKT Constraint Shadow Price)", color="white", fontsize=9)
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)
    ax.set_xticks(n_active)
    ax.text(0.02, 0.97,
            "λ = (Σμ_i − 2) / N\na_i* = (μ_i − λ) / 2\nλ is cost of 1% more budget",
            transform=ax.transAxes, color=ACC2, fontsize=7, va="top",
            bbox=dict(facecolor=PANEL2, alpha=0.85, boxstyle="round"))

    # ── Panel 5: Long-only vs Long+Short EV comparison ────────────────────
    ax = fig.add_subplot(gs[1, 1]); _ax(ax, grid=True)
    # Long+short: all 9 assets
    evs_ls = []
    # Long-only: best k long assets
    long_mus  = np.sort([abs(PRODUCTS[k]["ev_return"]) for k in ASSETS
                         if PRODUCTS[k]["signal"] == "long"])[::-1]
    short_mus = np.sort([abs(PRODUCTS[k]["ev_return"]) for k in ASSETS
                         if PRODUCTS[k]["signal"] == "short"])[::-1]

    for strategy, mus, color, lbl in [
        ("long_only",  long_mus,                       ACC1,   "Long-only (best k longs)"),
        ("long_short", np.sort(EFF_MU)[::-1],           LONG_C, "Long+Short (best k assets)"),
    ]:
        evs_k = []
        for k in range(1, len(mus) + 1):
            mu_k  = mus[:k]
            lam_k = (mu_k.sum() - 2.0) / k
            x_k   = np.maximum((mu_k - lam_k) / 2.0, 0.0)
            evs_k.append(BUDGET * float((x_k * mu_k - x_k ** 2).sum()))
        ax.plot(range(1, len(mus) + 1), np.array(evs_k) / 1000,
                color=color, lw=2, marker="o", ms=5, label=lbl)
    ax.axhline(ev_ana / 1000, color=ACC2, lw=1.2, ls="--", label=f"L+S max={_fmtk(ev_ana)}")
    ax.set_xlabel("Number of positions", color="white"); ax.set_ylabel("EV (k SeaShells)", color="white")
    ax.set_title("Long-Only vs Long+Short EV (KKT Optimal)", color="white", fontsize=9)
    ax.legend(facecolor=PANEL, labelcolor="white", fontsize=7)
    ax.text(0.97, 0.05,
            f"L+S gain vs Long-only:\n{_fmtk(ev_ana - evs_k[4])} (+{(ev_ana/evs_k[4]-1)*100:.0f}%)",
            transform=ax.transAxes, color=ACC2, fontsize=8, ha="right",
            bbox=dict(facecolor=PANEL2, alpha=0.85, boxstyle="round"))

    # ── Panel 6: Analytical summary table ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(PANEL); ax.axis("off")
    x_ana, ev_ana = analytical_optimum()
    rows = []
    total_fee_ana = 0.0
    for i, k in enumerate(ASSETS):
        p      = PRODUCTS[k]
        mu     = EFF_MU[i]
        v_kkt  = x_ana[i] * 100
        v_sa   = 50 * mu
        gross  = v_kkt / 100 * BUDGET * mu
        f_kkt  = fee(v_kkt)
        total_fee_ana += f_kkt
        net    = gross - f_kkt
        rows.append([p["name"][:12],
                     "L" if p["signal"] == "long" else "S",
                     f"{mu*100:.0f}%", f"{v_sa:.0f}%", f"{v_kkt:.1f}%",
                     _fmtk(gross), _fmtk(f_kkt), _fmtk(net)])
    rows.append(["─ TOTAL ─", "", "", "", "",
                 _fmtk(sum(float(r[5].replace("k","").replace("+","")) * (1000 if "k" in r[5] else 1) for r in rows)),
                 _fmtk(total_fee_ana),
                 _fmtk(ev_ana)])
    _tbl(ax, rows, ["Product", "Dir", "|μ|", "v*SA", "v*KKT", "Gross", "Fee", "Net"], fontsize=5.8)
    ax.set_title("KKT Analytical Allocation Summary", color="white", fontsize=8, pad=5)

    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    x_ana, ev_ana = analytical_optimum()
    print(f"╔═══ IMC Prosperity Round 5  |  Long+Short  |  seed={SEED} ════════════════╗")
    print(f"║  Analytical KKT optimum:")
    for i, k in enumerate(ASSETS):
        p   = PRODUCTS[k]
        pct = x_ana[i] * 100
        if pct < 0.5: continue
        print(f"║    {'L' if p['signal']=='long' else 'S'}  {p['name']:<22s}  {pct:>5.1f}%")
    print(f"║  Theoretical max EV = {_fmtk(ev_ana)}  (KKT, continuous)")
    print(f"╚{'═'*68}╝\n")

    rng = np.random.default_rng(SEED)

    print(f"[1] Simulating {N_SCENARIOS:,} scenarios per product ...")
    ret_mat = simulate_all(N_SCENARIOS, rng)    # (N, N_SCENARIOS)

    print("\n[2] Multi-stage portfolio optimisation (signed L+S) ...")
    opt  = Optimizer(ret_mat, np.random.default_rng(SEED + 1))
    pool = opt.run()
    print(f"\n    Total pool: {len(pool):,} portfolios")

    print(f"\n[3] Generating {OUTPUT_PDF} ...")
    with pdf_backend.PdfPages(OUTPUT_PDF) as pdf:
        _page_cover(pdf, pool)
        _page_fee_analysis(pdf)
        _page_products_overview(pdf, ret_mat)

        # Product detail pages: 3 per page
        for start in range(0, N, 3):
            _page_product_detail(pdf, ASSETS[start:start + 3], ret_mat)

        _page_frontier(pdf, pool,
                       x_metric="Std", x_label="Std Dev",
                       color_metric="Sharpe", color_label="Sharpe",
                       title="Efficient Frontier — EV vs Std Dev  (Σ|alloc|=100 %, L+S)")
        _page_frontier(pdf, pool,
                       x_metric="Downside_Vol", x_label="Downside Vol",
                       color_metric="Sortino", color_label="Sortino",
                       title="Sortino Frontier — EV vs Downside Vol  (Σ|alloc|=100 %, L+S)")

        _page_fee_optimization_study(pdf, pool)
        _page_table(pdf, pool)
        _page_deep_dive(pdf, pool, ret_mat)

    # ── Console recommendation ─────────────────────────────────────────────
    fp   = _full_pool(pool)
    best = fp[int(np.argmax([r["metrics"]["Sharpe"] for r in fp]))]
    m, a = best["metrics"], best["allocations"]
    total_fee = sum(fee(abs(a.get(k, 0))) for k in ASSETS)

    print(f"\n╔═══ RECOMMENDED ALLOCATION  (Max Sharpe, Full-Eval) ═════════════════╗")
    for k in ASSETS:
        av  = a.get(k, 0)
        if av == 0: continue
        p   = PRODUCTS[k]
        bar = "█" * (abs(av) // 3)
        dir_s = "L " if av > 0 else "S "
        print(f"║  {dir_s} {p['name']:<22s} {av:>+4d}%  {bar}")
    print(f"╠{'═'*68}╣")
    print(f"║  Σ|alloc|={sum(abs(v) for v in a.values())}%   Fees={_fmtk(total_fee)}")
    print(f"║  EV={_fmtk(m['EV'])}   Sharpe={m['Sharpe']:.3f}   Sortino={m['Sortino']:.3f}")
    print(f"║  Win={m['Win_Rate']:.1%}   VaR95={_fmtk(m['VaR_95'])}   CVaR95={_fmtk(m['CVaR_95'])}")
    print(f"╚{'═'*68}╝")
    print(f"\n  PDF → {OUTPUT_PDF}")
    print(f"\n── Standalone optima (v* = 50·|μ|) ──")
    for i, k in enumerate(ASSETS):
        p    = PRODUCTS[k]
        v_sa = 50 * EFF_MU[i]
        v_kkt = x_ana[i] * 100
        print(f"  {'L' if p['signal']=='long' else 'S'}  {p['name']:<22s}"
              f"  standalone={v_sa:.1f}%  KKT={v_kkt:.1f}%")


if __name__ == "__main__":
    main()

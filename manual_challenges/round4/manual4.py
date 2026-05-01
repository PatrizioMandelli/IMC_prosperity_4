"""
IMC Prosperity 4 – Round 4  |  AETHER_CRYSTAL exotic options
v4: Exhaustive global Monte Carlo  →  Pareto efficient frontier
    • 3^12 coarse grid  +  500 K random uniform  +  3 × 10-iter CEM (150 K/iter)
    • Downside-volatility (lower semi-std) risk metric  |  Sortino objective
    • CRN Greeks: Delta, Gamma, Vega, Theta  |  parallel bump-and-reprice
    • Parallel payoff computation via ThreadPoolExecutor
    • Rich progress logging with elapsed / ETA per phase
    • Save / load search results (skip re-run with LOAD_PREVIOUS=True)
    • 6-page PDF: frontier scatter · table · per-asset distributions ·
                  Greeks · portfolio PnL distributions · greedy analysis
"""
from __future__ import annotations
import os, time, warnings
from dataclasses import dataclass, replace as dc_replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import norm, gaussian_kde
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap as _get_cmap

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────  CONSTANTS  ────────────────────────────────────
S0                  = 50.0
SIGMA               = 2.51
R                   = 0.0
TRADING_DAYS        = 252
STEPS_PER_DAY       = 4
STEPS_PER_YEAR      = 1008        # TRADING_DAYS × STEPS_PER_DAY
DT                  = 1.0 / STEPS_PER_YEAR
STEPS_2W            = 40
STEPS_3W            = 60
CONTRACT_SIZE       = 3_000

# ─────────────────────────  SEARCH HYPERPARAMETERS  ──────────────────────────
SEED                = 42

# Monte Carlo paths
N_PATHS_COARSE      = 80_000      # paths for fast portfolio search
N_PATHS_FULL        = 500_000     # paths for final precise evaluation
N_GREEK_PATHS       = 50_000      # paths for Greeks (CRN)

# Search phases (evaluated on coarse paths)
GRID_LEVELS         = 3           # levels per product in coarse grid (G^N_PRODS portfolios)
N_RANDOM_SAMPLE     = 500_000    # phase-2: uniform random portfolios
N_CEM_RESTARTS      = 1           # independent CEM restarts
N_CEM_ITER          = 10          # iterations per CEM restart
N_CEM_SAMPLE        = 100_000     # portfolios per CEM iteration
CEM_ELITE_FRAC      = 0.05        # top fraction kept as elite
CEM_MIN_STD         = 0.02        # min std as fraction of max_vol

# Re-evaluation and output
BATCH_SIZE          = 2048        # portfolio eval batch (higher = faster BLAS)
N_TOP_REEVAL        = 10_000      # top-Sortino candidates re-evaluated on full paths
MAX_SCATTER         = 10_000      # scatter dots cap (prevents matplotlib OOM)

# ─────────────────────────  MODE & FILE CONTROL  ──────────────────────────────
LOAD_PREVIOUS       = False       # True → skip search, load RESULTS_FILE
SAVE_RESULTS        = True        # save all_pos/e/dv after search
RESULTS_FILE        = "sim_results.npz"
CACHE_DIR           = "mc_cache"
PDF_OUT             = "round4-manual.pdf"

# ─────────────────────────  PRODUCT REGISTRY  ─────────────────────────────────
@dataclass
class Product:
    name:          str
    kind:          str
    bid:           float
    ask:           float
    max_vol:       int
    K:             float = 0.0
    steps:         int   = STEPS_3W
    chooser_steps: int   = STEPS_2W
    barrier:       float = 0.0
    binary_payout: float = 10.0

PRODUCTS: list[Product] = [
    Product("AC",        "spot",        49.975, 50.025, 200),
    Product("AC_50_C",   "call",        12.000, 12.050,  50, K=50, steps=60),
    Product("AC_50_P",   "put",         12.000, 12.050,  50, K=50, steps=60),
    Product("AC_35_P",   "put",          4.330,  4.350,  50, K=35, steps=60),
    Product("AC_40_P",   "put",          6.500,  6.550,  50, K=40, steps=60),
    Product("AC_45_P",   "put",          9.050,  9.100,  50, K=45, steps=60),
    Product("AC_60_C",   "call",         8.800,  8.850,  50, K=60, steps=60),
    Product("AC_50_P_2", "put",          9.700,  9.750,  50, K=50, steps=40),
    Product("AC_50_C_2", "call",         9.700,  9.750,  50, K=50, steps=40),
    Product("AC_50_CO",  "chooser",     22.200, 22.300,  50, K=50, steps=60, chooser_steps=40),
    Product("AC_40_BP",  "binary_put",   5.000,  5.100,  50, K=40, steps=60, binary_payout=10.0),
    Product("AC_45_KO",  "ko_put",       0.150,  0.175, 500, K=45, steps=60, barrier=35.0),
]
N_PRODS  = len(PRODUCTS)
MAX_VOLS = np.array([p.max_vol for p in PRODUCTS], dtype=np.int32)

# ──────────────────────────  PAYOFF FUNCTIONS  ────────────────────────────────
def pf_spot(p, s):               return p[:, s]
def pf_call(p, s, K):            return np.maximum(p[:, s] - K, 0.0)
def pf_put(p, s, K):             return np.maximum(K - p[:, s], 0.0)
def pf_chooser(p, s, cs, K):
    return np.where(p[:, cs] >= K,
                    np.maximum(p[:, s] - K, 0.),
                    np.maximum(K - p[:, s], 0.))
def pf_binary_put(p, s, K, payout):
    return np.where(p[:, s] < K, payout, 0.0)
def pf_ko_put(p, s, K, B):
    knocked = np.min(p[:, 1:s+1], axis=1) < B
    return np.where(knocked, 0., np.maximum(K - p[:, s], 0.))

def compute_payoff(prod: Product, paths: np.ndarray) -> np.ndarray:
    k = prod.kind; s = prod.steps
    if k == "spot":        return pf_spot(paths, s)
    if k == "call":        return pf_call(paths, s, prod.K)
    if k == "put":         return pf_put(paths, s, prod.K)
    if k == "chooser":     return pf_chooser(paths, s, prod.chooser_steps, prod.K)
    if k == "binary_put":  return pf_binary_put(paths, s, prod.K, prod.binary_payout)
    if k == "ko_put":      return pf_ko_put(paths, s, prod.K, prod.barrier)
    raise ValueError(k)

# ──────────────────────  PATH GENERATION + PAYOFF MATRIX  ─────────────────────
def _cache_path(n: int, seed: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return f"{CACHE_DIR}/paths_N{n}_S{seed}.npy"

def generate_paths(n_paths: int, seed: int = SEED) -> np.ndarray:
    fpath = _cache_path(n_paths, seed)
    if os.path.exists(fpath):
        print(f"    ↳ cache hit: {fpath}  ({n_paths:,} paths)")
        return np.load(fpath)
    t0 = time.time()
    print(f"    ↳ generating {n_paths:,} GBM paths …", end="", flush=True)
    rng     = np.random.default_rng(seed)
    Z       = rng.standard_normal((n_paths, STEPS_3W))
    log_ret = (R - 0.5*SIGMA**2)*DT + SIGMA*np.sqrt(DT)*Z
    lp      = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)], axis=1)
    paths   = (S0 * np.exp(lp)).astype(np.float32)
    np.save(fpath, paths)
    print(f"  done in {time.time()-t0:.1f}s")
    return paths

def build_payoff_matrix(paths: np.ndarray) -> np.ndarray:
    """(N_paths, N_prods) float32; computed in parallel per product."""
    with ThreadPoolExecutor() as ex:
        cols = list(ex.map(lambda p: compute_payoff(p, paths), PRODUCTS))
    return np.column_stack(cols).astype(np.float32)

def precompute_unit_nets(payoff_mat: np.ndarray):
    """
    buy_unit_T[i, path]  = (payoff_i[path] - ask_i) * CS   (pnl per 1 long unit)
    sell_unit_T[i, path] = (bid_i - payoff_i[path]) * CS   (pnl per 1 short unit)
    Returns two (N_PRODS, N_PATHS) contiguous float32 arrays.
    """
    asks = np.array([p.ask for p in PRODUCTS], dtype=np.float32)
    bids = np.array([p.bid for p in PRODUCTS], dtype=np.float32)
    CS   = np.float32(CONTRACT_SIZE)
    return (np.ascontiguousarray(((payoff_mat - asks) * CS).T, dtype=np.float32),
            np.ascontiguousarray(((bids - payoff_mat) * CS).T, dtype=np.float32))

# ─────────────────────────  PORTFOLIO EVALUATION  ─────────────────────────────
def eval_portfolios(pos_batch: np.ndarray, but: np.ndarray, sut: np.ndarray):
    pos_f = pos_batch.astype(np.float32, copy=False)
    pnl   = np.maximum(pos_f, 0.) @ but         # (B, N) — new array
    pnl  += np.maximum(-pos_f, 0.) @ sut        # in-place; avoids second large alloc
    e     = pnl.mean(axis=1, dtype=np.float64)
    pnl  -= e[:, None].astype(np.float32)       # center in-place
    np.minimum(pnl, 0., out=pnl)               # clip to negatives in-place
    pnl  *= pnl
    return e, np.sqrt(pnl.mean(axis=1, dtype=np.float64))

def batch_eval(positions: np.ndarray, but: np.ndarray, sut: np.ndarray,
               desc: str = "eval") -> tuple[np.ndarray, np.ndarray]:
    N = len(positions)
    e_all  = np.empty(N, np.float64)
    dv_all = np.empty(N, np.float64)
    for s in tqdm(range(0, N, BATCH_SIZE), desc=f"  {desc}", unit="batch", leave=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        sl = slice(s, min(s + BATCH_SIZE, N))
        e_all[sl], dv_all[sl] = eval_portfolios(positions[sl], but, sut)
    return e_all, dv_all

def portfolio_pnl_dist(pos: np.ndarray, but: np.ndarray, sut: np.ndarray) -> np.ndarray:
    """Full (N_PATHS,) PnL distribution for a single portfolio."""
    pf = pos.astype(np.float32).reshape(1, -1)
    return (np.maximum(pf, 0.) @ but + np.maximum(-pf, 0.) @ sut).ravel().astype(np.float64)

def _sortino(e, dv):
    return np.where(dv > 0, e / dv, np.where(e > 0, 1e9, -1e9))

# ──────────────────────────  SEARCH PHASES  ───────────────────────────────────
def coarse_grid_positions() -> np.ndarray:
    """GRID_LEVELS^N_PRODS portfolios: each product at evenly-spaced levels of max_vol."""
    G      = GRID_LEVELS
    total  = G**N_PRODS
    levels = np.linspace(-1.0, 1.0, G)
    idx    = np.stack(np.unravel_index(np.arange(total, dtype=np.int64), [G]*N_PRODS), axis=1)
    return np.round(levels[idx] * MAX_VOLS).astype(np.int32)

def random_sample_positions(n: int, rng_seed: int) -> np.ndarray:
    """60% full-range integer uniform, 40% discrete {-1,0,+1}×max_vol."""
    rng    = np.random.default_rng(rng_seed)
    n_full = int(n * 0.6)
    n_disc = n - n_full
    full = rng.integers(-MAX_VOLS, MAX_VOLS + 1, size=(n_full, N_PRODS), dtype=np.int32)
    disc = (rng.integers(-1, 2, size=(n_disc, N_PRODS), dtype=np.int8).astype(np.int32)
            * MAX_VOLS)
    return np.vstack([full, disc])

def cem_run(but: np.ndarray, sut: np.ndarray,
            init_means: np.ndarray | None = None,
            restart_id: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One CEM run: N_CEM_ITER iterations × N_CEM_SAMPLE portfolios.
    Returns (positions, e_pnl, down_vol) for all iterations stacked.
    """
    means    = (init_means.astype(np.float64)
                if init_means is not None else np.zeros(N_PRODS, np.float64))
    stds     = MAX_VOLS.astype(np.float64) / 2.0
    min_stds = MAX_VOLS.astype(np.float64) * CEM_MIN_STD
    n_elite  = max(1, int(N_CEM_SAMPLE * CEM_ELITE_FRAC))

    all_pos: list[np.ndarray] = []
    all_e:   list[np.ndarray] = []
    all_dv:  list[np.ndarray] = []

    for it in range(N_CEM_ITER):
        rng      = np.random.default_rng(SEED + 200 + restart_id*1000 + it)
        pos_cont = rng.normal(means, stds, size=(N_CEM_SAMPLE, N_PRODS))
        pos_int  = np.clip(np.round(pos_cont), -MAX_VOLS, MAX_VOLS).astype(np.int32)

        e, dv    = batch_eval(pos_int, but, sut,
                              f"CEM r{restart_id+1}/{N_CEM_RESTARTS} "
                              f"it{it+1}/{N_CEM_ITER}")
        sor      = _sortino(e, dv)
        elite_ix = np.argsort(sor)[-n_elite:]
        elite    = pos_int[elite_ix].astype(np.float64)
        means    = elite.mean(0)
        stds     = np.maximum(elite.std(0), min_stds)

        all_pos.append(pos_int); all_e.append(e); all_dv.append(dv)
        print(f"      elite Sortino [{sor[elite_ix].min():.3f}, "
              f"{sor[elite_ix].max():.3f}]  "
              f"means={np.round(means, 1)}")

    return np.vstack(all_pos), np.concatenate(all_e), np.concatenate(all_dv)

# ──────────────────────────  GREEKS (CRN)  ────────────────────────────────────
def compute_product_greeks(n_paths: int = N_GREEK_PATHS) -> list[dict]:
    """
    Delta, Gamma, Vega, Theta per product via bump-and-reprice with CRN.
    Bump computations run in parallel threads.
    """
    rng  = np.random.default_rng(SEED + 999)
    Z    = rng.standard_normal((n_paths, STEPS_3W))      # shared Z scores
    h_s  = 0.5                                            # spot bump
    h_v  = 0.01                                           # vol bump

    def make_paths(s0, sigma, Z_arr=Z):
        lr = (R - 0.5*sigma**2)*DT + sigma*np.sqrt(DT)*Z_arr
        lp = np.concatenate([np.zeros((len(Z_arr), 1)), np.cumsum(lr, 1)], 1)
        return (s0 * np.exp(lp)).astype(np.float32)

    # parallel path generation for all bump scenarios
    with ThreadPoolExecutor() as ex:
        f_base = ex.submit(make_paths, S0,      SIGMA)
        f_su   = ex.submit(make_paths, S0+h_s,  SIGMA)
        f_sd   = ex.submit(make_paths, S0-h_s,  SIGMA)
        f_vu   = ex.submit(make_paths, S0,      SIGMA+h_v)
        f_vd   = ex.submit(make_paths, S0,      SIGMA-h_v)
        base, su, sd, vu, vd = (f_base.result(), f_su.result(), f_sd.result(),
                                 f_vu.result(), f_vd.result())

    # shorter paths for theta (one trading day fewer steps)
    n_th = STEPS_3W - STEPS_PER_DAY
    Z_th = Z[:, STEPS_PER_DAY:]          # skip first day's innovations
    paths_th = make_paths(S0, SIGMA, Z_th[:, :n_th] if n_th > 0 else Z_th)

    results: list[dict] = []
    for prod in PRODUCTS:
        pf_b  = compute_payoff(prod, base).mean()
        def bump_payoff(p): return compute_payoff(prod, p).mean()
        with ThreadPoolExecutor() as ex:
            r_su = ex.submit(bump_payoff, su)
            r_sd = ex.submit(bump_payoff, sd)
            r_vu = ex.submit(bump_payoff, vu)
            r_vd = ex.submit(bump_payoff, vd)
            pf_su, pf_sd = r_su.result(), r_sd.result()
            pf_vu, pf_vd = r_vu.result(), r_vd.result()

        delta = (pf_su - pf_sd) / (2 * h_s)
        gamma = (pf_su - 2*pf_b + pf_sd) / (h_s**2)
        vega  = (pf_vu - pf_vd) / (2 * h_v)

        # theta: value change from 1 trading day passing (negative = decay)
        if prod.steps > STEPS_PER_DAY:
            prod_th = dc_replace(prod,
                                 steps=min(prod.steps - STEPS_PER_DAY, n_th),
                                 chooser_steps=min(prod.chooser_steps,
                                                   prod.steps - STEPS_PER_DAY - 1))
            pf_th   = compute_payoff(prod_th, paths_th).mean()
            theta   = pf_th - pf_b          # per trading day (r=0, no discounting)
        else:
            theta = 0.0

        results.append({"name": prod.name, "fair": pf_b,
                         "delta": delta, "gamma": gamma,
                         "vega": vega, "theta": theta})
    return results

# ─────────────────────────  PARETO FRONTIER  ──────────────────────────────────
def pareto_frontier(e_pnl: np.ndarray, down_vol: np.ndarray) -> np.ndarray:
    """
    Sweep by down_vol ascending; keep portfolios that strictly improve running-max
    E[PnL] → Pareto-optimal set on (E[PnL], down_vol).
    """
    ix = np.argsort(down_vol)
    e_s = e_pnl[ix]
    front: list[int] = []; max_e = -np.inf
    for j, idx in enumerate(ix):
        if e_s[j] > max_e:
            front.append(idx); max_e = e_s[j]
    return np.array(front)

def greedy_portfolio(payoff_mat: np.ndarray) -> np.ndarray:
    fair = payoff_mat.mean(0)
    pos  = np.zeros(N_PRODS, np.int32)
    for i, p in enumerate(PRODUCTS):
        if   fair[i] > p.ask: pos[i] =  p.max_vol
        elif fair[i] < p.bid: pos[i] = -p.max_vol
    return pos

# ──────────────────────  BLACK-SCHOLES REFERENCE  ────────────────────────────
def bs_call(S, K, T, sig, r=0.):
    if T <= 0: return max(S-K, 0.)
    d1 = (np.log(S/K) + (r+.5*sig**2)*T) / (sig*T**.5)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d1 - sig*T**.5)
def bs_put(S, K, T, sig, r=0.):
    if T <= 0: return max(K-S, 0.)
    d1 = (np.log(S/K) + (r+.5*sig**2)*T) / (sig*T**.5)
    d2 = d1 - sig*T**.5
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
def bs_chooser(S, K, T1, T2, sig, r=0.):
    return bs_call(S, K, T2, sig, r) + bs_put(S, K, T1, sig, r)
def bs_binary_put(S, K, T, sig, payout=10., r=0.):
    if T <= 0: return payout if S < K else 0.
    d2 = (np.log(S/K) + (r-.5*sig**2)*T) / (sig*T**.5)
    return payout*np.exp(-r*T)*norm.cdf(-d2)

# ─────────────────────────  SAVE / LOAD  ──────────────────────────────────────
def save_results(all_pos, all_e, all_dv, path=RESULTS_FILE):
    np.savez_compressed(path, all_pos=all_pos, all_e=all_e, all_dv=all_dv)
    print(f"  Results saved → {path}  ({os.path.getsize(path)//1024:,} KB)")

def load_results(path=RESULTS_FILE):
    d = np.load(path)
    return d["all_pos"], d["all_e"], d["all_dv"]

# ─────────────────────────  CONSOLE TABLE  ────────────────────────────────────
def print_frontier_table(positions, e_pnl, dv):
    names = [p.name for p in PRODUCTS]
    sor   = _sortino(e_pnl, dv)
    order = np.argsort(e_pnl)[::-1]
    W = 145
    print(f"\n{'='*W}\n  Efficient Frontier Portfolios  (sorted by E[PnL] desc)\n{'='*W}")
    hdr  = f"{'Rank':>4}  {'E[PnL]':>12}  {'DwnVol':>12}  {'Sortino':>9}  "
    hdr += "  ".join(f"{n:>9}" for n in names)
    print(hdr); print("-"*W)
    for rank, i in enumerate(order, 1):
        row  = f"{rank:>4}  {e_pnl[i]:>12,.1f}  {dv[i]:>12,.1f}  {sor[i]:>9.3f}  "
        row += "  ".join(f"{v:>9}" for v in positions[i])
        print(row)
    print("="*W)

# ───────────────────────────  PLOTTING  ───────────────────────────────────────
_CMAP  = _get_cmap("plasma")
_CMAP2 = _get_cmap("RdBu_r")

def _subset(n_total, max_pts=MAX_SCATTER, seed=0):
    if n_total <= max_pts: return np.arange(n_total)
    return np.random.default_rng(seed).choice(n_total, max_pts, replace=False)

def _style_table(tbl, cell_data, n_fixed_cols=0):
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="w", fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#f5f5f5")
        else:            cell.set_facecolor("#ffffff")
        if r > 0 and c >= n_fixed_cols:
            try:
                v = float(cell_data[r-1][c].replace(",", ""))
                if v > 0:   cell.set_facecolor("#d4efdf")
                elif v < 0: cell.set_facecolor("#fadbd8")
            except Exception: pass


# ── Page 1: Frontier scatter ──────────────────────────────────────────────────
def page_frontier_scatter(pdf, e_all, dv_all, front_idx, front_pos,
                          fe_full, fdv_full, g_e, g_dv):
    sor_all = _sortino(e_all, dv_all)
    vmin, vmax = np.percentile(sor_all, [5, 95])
    nc = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(16, 20))
    fig.suptitle("AETHER_CRYSTAL – Efficient Frontier  (Global Monte Carlo v4)",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 1, hspace=0.42, top=0.95, bottom=0.05,
                           left=0.09, right=0.95)

    # [0] Main risk-return scatter
    ax0 = fig.add_subplot(gs[0])
    sub = _subset(len(e_all))
    ax0.scatter(dv_all[sub], e_all[sub], c=sor_all[sub], cmap=_CMAP, norm=nc,
                s=3, alpha=0.3, linewidths=0, rasterized=True, label="Portfolios")
    ord_f = np.argsort(fdv_full)
    ax0.plot(fdv_full[ord_f], fe_full[ord_f], "w-", lw=2.8, zorder=4)
    ax0.plot(fdv_full[ord_f], fe_full[ord_f], "r-", lw=1.8, zorder=5, label="Pareto frontier")
    ax0.scatter(fdv_full, fe_full, s=35, c="red", zorder=6, edgecolors="w", lw=0.4)
    ax0.scatter(g_dv, g_e, marker="*", s=500, c="gold",
                edgecolors="k", lw=0.9, zorder=10, label="Greedy ★")
    fig.colorbar(ScalarMappable(norm=nc, cmap=_CMAP), ax=ax0, pad=0.01, label="Sortino")
    ax0.set(xlabel="Downside volatility  (lower semi-std)",
            ylabel="E[PnL]",
            title="Risk–Return Landscape  (colour = Sortino ratio)")
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.22)

    # [1] Sortino curve along frontier
    ax1 = fig.add_subplot(gs[1])
    sor_f = _sortino(fe_full[ord_f], fdv_full[ord_f])
    ax1.plot(fdv_full[ord_f], sor_f, "r-o", ms=4, lw=1.5)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    best_s = np.argmax(sor_f)
    ax1.scatter(fdv_full[ord_f][best_s], sor_f[best_s],
                marker="D", s=120, c="cyan", edgecolors="k", zorder=5,
                label=f"Max Sortino = {sor_f[best_s]:.3f}")
    ax1.set(xlabel="Downside volatility", ylabel="Sortino ratio",
            title="Sortino ratio along the efficient frontier")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.22)

    # [2] Position heatmap of frontier portfolios (left→right: low DV to high DV)
    ax2 = fig.add_subplot(gs[2])
    fp_norm = front_pos[ord_f] / MAX_VOLS[None, :]
    im = ax2.imshow(fp_norm.T, aspect="auto", cmap=_CMAP2,
                    vmin=-1, vmax=1, interpolation="nearest")
    ax2.set_yticks(range(N_PRODS))
    ax2.set_yticklabels([p.name for p in PRODUCTS], fontsize=7.5)
    ax2.set(xlabel="Frontier portfolio (low DV → high DV)",
            title="Frontier positions  (blue=long, red=short, white=flat)")
    fig.colorbar(im, ax=ax2, pad=0.01, label="pos / max_vol")

    pdf.savefig(fig, dpi=100); plt.close(fig)


# ── Page 2: Frontier table ────────────────────────────────────────────────────
def page_frontier_table(pdf, front_pos, fe_full, fdv_full):
    sor_f  = _sortino(fe_full, fdv_full)
    order  = np.argsort(fe_full)[::-1]
    n_rows = len(front_pos)
    names  = [p.name for p in PRODUCTS]
    hdrs   = ["Rank", "E[PnL]", "DwnVol", "Sortino"] + names

    fig_h = max(7, 0.30 * n_rows + 2.5)
    fig   = plt.figure(figsize=(max(18, 3 + 1.2*N_PRODS), fig_h))
    fig.suptitle("Efficient Frontier – Full Position Table  (sorted by E[PnL] desc)",
                 fontsize=12, fontweight="bold")
    ax = fig.add_subplot(111); ax.axis("off")

    rows = []
    for rank, i in enumerate(order, 1):
        rows.append([str(rank), f"{fe_full[i]:,.1f}", f"{fdv_full[i]:,.1f}",
                     f"{sor_f[i]:.3f}"] + [str(v) for v in front_pos[i]])

    cw = [0.04, 0.08, 0.08, 0.07] + [0.055]*N_PRODS
    cw = [w/sum(cw) for w in cw]
    tbl = ax.table(cellText=rows, colLabels=hdrs, cellLoc="center",
                   loc="center", colWidths=cw)
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.35)
    _style_table(tbl, rows, n_fixed_cols=4)

    pdf.savefig(fig, dpi=100, bbox_inches="tight"); plt.close(fig)


# ── Page 3: Per-asset payoff distributions ────────────────────────────────────
def page_asset_distributions(pdf, payoff_mat: np.ndarray):
    nrows, ncols = 6, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 22))
    fig.suptitle("Per-Asset Payoff Distributions  (MC simulated)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.965, bottom=0.04,
                        left=0.07, right=0.97)

    for i, (prod, ax) in enumerate(zip(PRODUCTS, axes.ravel())):
        pf  = payoff_mat[:, i].astype(np.float64)
        fair = pf.mean()
        std  = pf.std()
        skew = float(np.mean(((pf - fair)/std)**3)) if std > 0 else 0.
        kurt = float(np.mean(((pf - fair)/std)**4)) - 3 if std > 0 else 0.

        n_bins = min(150, max(30, int(len(pf)**0.45)))
        ax.hist(pf, bins=n_bins, density=True, color="#5b8cdb",
                alpha=0.65, edgecolor="none")

        # KDE overlay
        if std > 1e-6:
            xg = np.linspace(pf.min(), pf.max(), 400)
            try:
                ax.plot(xg, gaussian_kde(pf)(xg), "k-", lw=1.2, label="KDE")
            except Exception:
                pass

        ax.axvline(fair,    color="#27ae60", lw=1.8, ls="-",  label=f"Fair {fair:.3f}")
        ax.axvline(prod.bid, color="#e74c3c", lw=1.4, ls="--", label=f"Bid {prod.bid}")
        ax.axvline(prod.ask, color="#2980b9", lw=1.4, ls=":",  label=f"Ask {prod.ask}")

        eb  = fair - prod.ask
        es  = prod.bid - fair
        best_edge = eb if abs(eb) >= abs(es) else es
        direction = "BUY" if eb > es else "SELL"
        clr = "#27ae60" if best_edge > 0 else "#e74c3c"

        ax.set_title(f"{prod.name}  [fair={fair:.3f}  edge({direction})={best_edge:+.4f}]",
                     fontsize=8.5, fontweight="bold", color=clr)
        ax.set_xlabel("Payoff", fontsize=7.5)
        ax.set_ylabel("Density", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right", ncol=2)

        stats = (f"σ={std:.3f}  skew={skew:.2f}  kurt={kurt:.2f}\n"
                 f"min={pf.min():.2f}  max={pf.max():.2f}")
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.grid(True, alpha=0.2)

    for ax in axes.ravel()[N_PRODS:]:
        ax.set_visible(False)

    pdf.savefig(fig, dpi=100); plt.close(fig)


# ── Page 4: Greeks ───────────────────────────────────────────────────────────
def page_greeks(pdf, greeks: list[dict]):
    names  = [g["name"] for g in greeks]
    deltas = [g["delta"] for g in greeks]
    gammas = [g["gamma"] for g in greeks]
    vegas  = [g["vega"]  for g in greeks]
    thetas = [g["theta"] for g in greeks]
    fairs  = [g["fair"]  for g in greeks]

    fig = plt.figure(figsize=(16, 20))
    fig.suptitle("Per-Asset Greeks  (CRN bump-and-reprice, r=0, σ=251%)",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.38,
                           top=0.94, bottom=0.22, left=0.08, right=0.97)

    def _bar(ax, vals, title, color_pos="#2ecc71", color_neg="#e74c3c"):
        colors = [color_pos if v >= 0 else color_neg for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="k", lw=0.9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(True, axis="x", alpha=0.3)

    _bar(fig.add_subplot(gs[0, 0]), deltas, "Delta  (∂V/∂S, bump ±0.5)")
    _bar(fig.add_subplot(gs[0, 1]), gammas, "Gamma  (∂²V/∂S²)",
         "#3498db", "#e67e22")
    _bar(fig.add_subplot(gs[1, 0]), vegas,  "Vega  (∂V/∂σ, bump ±1%)",
         "#9b59b6", "#e74c3c")
    _bar(fig.add_subplot(gs[1, 1]), thetas, "Theta  (Δfair per trading day)")

    # Summary table (bottom)
    ax_t = fig.add_axes([0.04, 0.02, 0.92, 0.20])
    ax_t.axis("off")
    hdrs  = ["Product", "Fair", "Bid", "Ask", "Edge(B)", "Edge(S)",
             "Delta", "Gamma", "Vega", "Theta/day"]
    rows  = []
    for g, p in zip(greeks, PRODUCTS):
        rows.append([
            g["name"],
            f"{g['fair']:.4f}",
            f"{p.bid:.4f}", f"{p.ask:.4f}",
            f"{g['fair']-p.ask:+.4f}", f"{p.bid-g['fair']:+.4f}",
            f"{g['delta']:+.4f}", f"{g['gamma']:+.5f}",
            f"{g['vega']:+.4f}", f"{g['theta']:+.5f}",
        ])
    tbl = ax_t.table(cellText=rows, colLabels=hdrs, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.4)
    _style_table(tbl, rows, n_fixed_cols=4)

    pdf.savefig(fig, dpi=100); plt.close(fig)


# ── Page 5: Portfolio PnL distributions ──────────────────────────────────────
def page_pnl_distributions(pdf, front_pos, fe_full, fdv_full, g_pos,
                            but_f, sut_f):
    sor_f = _sortino(fe_full, fdv_full)

    # Pick representative frontier portfolios
    i_max_e   = np.argmax(fe_full)
    i_max_sor = np.argmax(sor_f)
    i_min_dv  = np.argmin(fdv_full)

    # deduplicate by index
    candidates_idx = list(dict.fromkeys([i_max_e, i_max_sor, i_min_dv]))
    labels_map = {
        i_max_e:   f"Max E[PnL]  E={fe_full[i_max_e]:,.0f}",
        i_max_sor: f"Max Sortino  S={sor_f[i_max_sor]:.3f}",
        i_min_dv:  f"Min DV  DV={fdv_full[i_min_dv]:,.0f}",
    }
    colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]

    n_panels = len(candidates_idx) + 1   # +1 for greedy
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5*n_panels))
    if n_panels == 1: axes = [axes]
    fig.suptitle("Portfolio PnL Distributions  (500 K MC paths)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.55, top=0.97, bottom=0.04, left=0.09, right=0.97)

    def _plot_dist(ax, pnl, label, color):
        e   = pnl.mean(); std = pnl.std()
        dv  = np.sqrt(np.mean(np.minimum(pnl - e, 0.)**2))
        sor = e / dv if dv > 0 else 0.
        p5  = np.percentile(pnl, 5)
        p95 = np.percentile(pnl, 95)
        n_bins = min(200, max(40, int(len(pnl)**0.45)))
        ax.hist(pnl, bins=n_bins, density=True, color=color,
                alpha=0.55, edgecolor="none")
        if std > 1e-3:
            xg = np.linspace(p5 - 2*std, p95 + 2*std, 500)
            try:
                kde = gaussian_kde(pnl, bw_method=0.15)
                ax.plot(xg, kde(xg), "-", color=color, lw=2)
            except Exception:
                pass
        ax.axvline(0,  color="black",   lw=1.0, ls="--", alpha=0.6)
        ax.axvline(e,  color="#f39c12", lw=2.0, ls="-",  label=f"E[PnL]={e:,.0f}")
        ax.axvline(p5, color="gray",    lw=1.2, ls=":",  label=f"5th pct={p5:,.0f}")
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                         pnl.min(), 0, alpha=0.08, color="red")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("PnL (settlement)"); ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper right")
        stats = (f"E={e:,.0f}  σ={std:,.0f}  DV={dv:,.0f}  "
                 f"Sortino={sor:.3f}\n5th={p5:,.0f}  95th={p95:,.0f}  "
                 f"min={pnl.min():,.0f}")
        ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=8,
                va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.grid(True, alpha=0.2)

    for k, (i, col) in enumerate(zip(candidates_idx, colors)):
        pnl = portfolio_pnl_dist(front_pos[i], but_f, sut_f)
        _plot_dist(axes[k], pnl, labels_map[i], col)

    # Greedy
    g_pnl = portfolio_pnl_dist(g_pos, but_f, sut_f)
    _plot_dist(axes[-1], g_pnl, "Greedy portfolio  (per-product local optimum)",
               "#f39c12")

    # Fix fill_betweenx (called before ylim is finalised): redo
    for ax in axes:
        ax.figure.canvas.draw()   # force layout

    pdf.savefig(fig, dpi=100); plt.close(fig)


# ── Page 6: Greedy analysis ───────────────────────────────────────────────────
def page_greedy_analysis(pdf, payoff_mat, g_pos, g_e, g_dv,
                         e_all, dv_all, fe_full, fdv_full, front_pos):
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle("Greedy Baseline Analysis", fontsize=15, fontweight="bold", y=0.99)
    gs  = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.38,
                            top=0.94, bottom=0.06, left=0.08, right=0.97)

    fair  = payoff_mat.mean(0)
    asks  = np.array([p.ask for p in PRODUCTS])
    bids  = np.array([p.bid for p in PRODUCTS])
    names = [p.name for p in PRODUCTS]

    # [0,0] Edge bars
    ax00 = fig.add_subplot(gs[0, 0])
    eb   = fair - asks; es = bids - fair
    best = np.where(np.abs(eb) >= np.abs(es), eb, es)
    c_e  = ["#27ae60" if v > 0 else "#e74c3c" if v < 0 else "#95a5a6" for v in best]
    ax00.barh(names, best, color=c_e)
    ax00.axvline(0, color="k", lw=0.9)
    ax00.set(title="Per-product best edge", xlabel="Fair − price")
    ax00.grid(True, axis="x", alpha=0.3)

    # [0,1] Position bars
    ax01 = fig.add_subplot(gs[0, 1])
    c_p  = ["#27ae60" if v > 0 else "#e74c3c" if v < 0 else "#95a5a6" for v in g_pos]
    ax01.barh(names, g_pos, color=c_p)
    ax01.axvline(0, color="k", lw=0.9)
    ax01.set(title="Greedy portfolio positions", xlabel="Position (contracts)")
    ax01.grid(True, axis="x", alpha=0.3)

    # [1,:] Scatter + frontier + greedy ★ + max Sortino ◆
    ax1  = fig.add_subplot(gs[1, :])
    sor_all = _sortino(e_all, dv_all)
    vmin, vmax = np.percentile(sor_all, [5, 95])
    nc   = Normalize(vmin=vmin, vmax=vmax)
    sub  = _subset(len(e_all), max_pts=2000)
    ax1.scatter(dv_all[sub], e_all[sub], c=sor_all[sub], cmap=_CMAP, norm=nc,
                s=5, alpha=0.28, linewidths=0, rasterized=True)
    ord_f = np.argsort(fdv_full)
    ax1.plot(fdv_full[ord_f], fe_full[ord_f], "r-", lw=2, zorder=4, label="Frontier")
    ax1.scatter(g_dv, g_e, marker="*", s=600, c="gold",
                edgecolors="k", lw=1, zorder=10, label="Greedy ★")
    bsi  = np.argmax(_sortino(fe_full, fdv_full))
    ax1.scatter(fdv_full[bsi], fe_full[bsi], marker="D", s=200,
                c="cyan", edgecolors="k", lw=1, zorder=10, label="Max Sortino ◆")
    ax1.set(xlabel="Downside volatility", ylabel="E[PnL]",
            title="Greedy ★ vs efficient frontier")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.22)

    # [2,:] Per-product contribution in greedy
    ax2  = fig.add_subplot(gs[2, :])
    per_e = np.zeros(N_PRODS)
    for i, (p, v) in enumerate(zip(PRODUCTS, g_pos)):
        if v == 0: continue
        m = payoff_mat[:, i].mean()
        per_e[i] = (m - p.ask if v > 0 else p.bid - m) * abs(v) * CONTRACT_SIZE
    ax2.bar(names, per_e,
            color=["#27ae60" if x > 0 else "#e74c3c" for x in per_e])
    ax2.axhline(0, color="k", lw=0.9)
    ax2.set(title=f"Per-product E[PnL] contribution in greedy  "
                  f"(total={g_e:,.0f}  DV={g_dv:,.0f}  "
                  f"Sortino={_sortino(g_e, g_dv):.3f})",
            ylabel="E[PnL]")
    ax2.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    pdf.savefig(fig, dpi=100); plt.close(fig)


# ─────────────────────────────  MAIN  ─────────────────────────────────────────
def _eta(n_batches, sec_per_batch=0.12):
    return n_batches * sec_per_batch

def main():
    t_start = time.time()
    sep = "="*72

    print(f"\n{sep}")
    print("  IMC Prosperity 4  |  Round 4  |  Global Frontier Search  (v4)")
    print(f"{sep}")

    # ── Upfront work estimate ────────────────────────────────────────────────
    total_coarse = GRID_LEVELS**N_PRODS
    total_random = N_RANDOM_SAMPLE
    total_cem    = N_CEM_RESTARTS * N_CEM_ITER * N_CEM_SAMPLE
    total_port   = total_coarse + total_random + total_cem
    est_sec      = _eta(total_port / BATCH_SIZE)
    print(f"\n  Search plan (coarse paths = {N_PATHS_COARSE:,}  full paths = {N_PATHS_FULL:,})")
    print(f"  ├─ Coarse grid  : {GRID_LEVELS}^{N_PRODS} = {total_coarse:>10,} portfolios")
    print(f"  ├─ Random sample: {total_random:>22,} portfolios")
    print(f"  ├─ CEM          : {N_CEM_RESTARTS}×{N_CEM_ITER}×{N_CEM_SAMPLE:,} = "
          f"{total_cem:>10,} portfolios")
    print(f"  └─ Total        : {total_port:>22,} portfolios")
    print(f"  Estimated search time ≈ {est_sec/60:.0f}–{est_sec/60*2:.0f} min")
    if LOAD_PREVIOUS:
        print(f"\n  *** LOAD_PREVIOUS=True → will skip search and load {RESULTS_FILE}")

    # ── 1. Paths ──────────────────────────────────────────────────────────────
    t1 = time.time()
    print(f"\n{'─'*50}")
    print(f"[1/8] MC Paths")
    paths_coarse = generate_paths(N_PATHS_COARSE, seed=SEED)
    paths_full   = generate_paths(N_PATHS_FULL,   seed=SEED)
    print(f"  done in {time.time()-t1:.1f}s")

    # ── 2. Payoff matrices ────────────────────────────────────────────────────
    t2 = time.time()
    print(f"\n{'─'*50}")
    print(f"[2/8] Payoff matrices  (parallel per product)")
    pm_coarse = build_payoff_matrix(paths_coarse)
    pm_full   = build_payoff_matrix(paths_full)
    but_c, sut_c = precompute_unit_nets(pm_coarse)
    but_f, sut_f = precompute_unit_nets(pm_full)
    print(f"  done in {time.time()-t2:.1f}s")

    # ── 3. Fair values ────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"[3/8] Fair values & BS reference")
    fair_mc = pm_full.mean(0)
    T2_yr   = STEPS_2W / STEPS_PER_YEAR
    print(f"\n  {'Product':<15} {'MC fair':>10} {'BS ref':>10} "
          f"{'Bid':>8} {'Ask':>8} {'Edge(B)':>9} {'Edge(S)':>9}")
    print("  " + "─"*80)
    for i, p in enumerate(PRODUCTS):
        T = p.steps / STEPS_PER_YEAR; Tc = p.chooser_steps / STEPS_PER_YEAR
        if   p.kind == "call":        bs = bs_call(S0, p.K, T, SIGMA)
        elif p.kind == "put":         bs = bs_put(S0, p.K, T, SIGMA)
        elif p.kind == "chooser":     bs = bs_chooser(S0, p.K, Tc, T2_yr, SIGMA)
        elif p.kind == "binary_put":  bs = bs_binary_put(S0, p.K, T, SIGMA, p.binary_payout)
        else:                          bs = float("nan")
        mc = fair_mc[i]
        print(f"  {p.name:<15} {mc:>10.4f} {bs:>10.4f} {p.bid:>8.4f} {p.ask:>8.4f} "
              f"{mc-p.ask:>9.4f} {p.bid-mc:>9.4f}")

    # ── Portfolio search (skip if LOAD_PREVIOUS) ──────────────────────────────
    if LOAD_PREVIOUS and os.path.exists(RESULTS_FILE):
        print(f"\n{'─'*50}")
        print(f"[4-6/8] Loading previous results from {RESULTS_FILE} …")
        all_pos, all_e, all_dv = load_results(RESULTS_FILE)
        print(f"  Loaded {len(all_pos):,} portfolios")
    else:
        if LOAD_PREVIOUS:
            print(f"  (LOAD_PREVIOUS=True but {RESULTS_FILE} not found; running search)")

        # ── 4. Coarse grid ────────────────────────────────────────────────────
        t4 = time.time()
        print(f"\n{'─'*50}")
        print(f"[4/8] Coarse grid  {GRID_LEVELS}^{N_PRODS} = {total_coarse:,}")
        coarse_pos       = coarse_grid_positions()
        e_c, dv_c        = batch_eval(coarse_pos, but_c, sut_c,
                                      f"Coarse 3^{N_PRODS}")
        print(f"  ✓ {total_coarse:,} portfolios in {time.time()-t4:.1f}s  "
              f"best Sortino={_sortino(e_c,dv_c).max():.3f}")

        # ── 5. Random uniform sample ──────────────────────────────────────────
        t5 = time.time()
        print(f"\n{'─'*50}")
        print(f"[5/8] Random uniform  ({N_RANDOM_SAMPLE:,} portfolios)")
        rand_pos = random_sample_positions(N_RANDOM_SAMPLE, SEED + 50)
        e_r, dv_r = batch_eval(rand_pos, but_c, sut_c, "Random uniform")
        print(f"  ✓ {N_RANDOM_SAMPLE:,} portfolios in {time.time()-t5:.1f}s  "
              f"best Sortino={_sortino(e_r,dv_r).max():.3f}")

        # ── 6. CEM restarts ───────────────────────────────────────────────────
        t6 = time.time()
        print(f"\n{'─'*50}")
        print(f"[6/8] CEM  ({N_CEM_RESTARTS} restarts × {N_CEM_ITER} iters × "
              f"{N_CEM_SAMPLE:,}/iter)")
        all_cem_pos: list[np.ndarray] = []
        all_cem_e:   list[np.ndarray] = []
        all_cem_dv:  list[np.ndarray] = []

        # CEM restart initializations: flat, from best coarse, from best random
        merged_e   = np.concatenate([e_c, e_r])
        merged_dv  = np.concatenate([dv_c, dv_r])
        merged_pos = np.vstack([coarse_pos, rand_pos])
        best_so_far_idx = np.argmax(_sortino(merged_e, merged_dv))
        best_init = merged_pos[best_so_far_idx].astype(np.float64)

        inits = [None, best_init, best_init * 0.5][:N_CEM_RESTARTS]
        for r_id, init in enumerate(inits):
            print(f"\n  ── CEM restart {r_id+1}/{N_CEM_RESTARTS} "
                  f"{'(flat start)' if init is None else '(from best)'}  ──")
            pos, e, dv = cem_run(but_c, sut_c, init_means=init, restart_id=r_id)
            all_cem_pos.append(pos); all_cem_e.append(e); all_cem_dv.append(dv)

        cem_pos = np.vstack(all_cem_pos)
        e_cem   = np.concatenate(all_cem_e)
        dv_cem  = np.concatenate(all_cem_dv)
        print(f"\n  ✓ CEM done in {time.time()-t6:.1f}s  "
              f"best Sortino={_sortino(e_cem,dv_cem).max():.3f}")

        # ── Merge + dedup ─────────────────────────────────────────────────────
        all_pos = np.vstack([coarse_pos, rand_pos, cem_pos])
        all_e   = np.concatenate([e_c, e_r, e_cem])
        all_dv  = np.concatenate([dv_c, dv_r, dv_cem])
        print(f"\n  Deduplicating {len(all_pos):,} portfolios …", end="", flush=True)
        _, uniq = np.unique(all_pos, axis=0, return_index=True)
        all_pos, all_e, all_dv = all_pos[uniq], all_e[uniq], all_dv[uniq]
        print(f"  → {len(all_pos):,} unique")

        if SAVE_RESULTS:
            save_results(all_pos, all_e, all_dv)

    # ── 7. Full-path re-evaluation of top candidates ──────────────────────────
    t7 = time.time()
    print(f"\n{'─'*50}")
    print(f"[7/8] Full-path re-eval of top {N_TOP_REEVAL:,} by Sortino")
    sor_all = _sortino(all_e, all_dv)
    top_idx = np.argsort(sor_all)[-N_TOP_REEVAL:]
    e_top, dv_top = batch_eval(all_pos[top_idx], but_f, sut_f,
                               f"Full-path top-{N_TOP_REEVAL}")
    all_e[top_idx] = e_top; all_dv[top_idx] = dv_top
    print(f"  ✓ done in {time.time()-t7:.1f}s")

    # ── 8. Pareto frontier + greedy + Greeks ──────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"[8/8] Pareto frontier + greedy + Greeks")

    front_idx = pareto_frontier(all_e, all_dv)
    front_pos = all_pos[front_idx]
    print(f"  Pareto frontier: {len(front_idx)} portfolios")

    fe_full, fdv_full = batch_eval(front_pos, but_f, sut_f, "Frontier full-path eval")
    # write precise values back so scatter cloud is consistent with frontier line
    all_e[front_idx]  = fe_full
    all_dv[front_idx] = fdv_full

    g_pos = greedy_portfolio(pm_full)
    g_arr_e, g_arr_dv = eval_portfolios(g_pos[None, :], but_f, sut_f)
    g_e = g_arr_e[0]; g_dv = g_arr_dv[0]
    print(f"  Greedy:  E[PnL]={g_e:,.1f}  DwnVol={g_dv:,.1f}  "
          f"Sortino={_sortino(g_e,g_dv):.3f}")
    print(f"  Greedy positions: "
          + ", ".join(f"{p.name}={v}" for p, v in zip(PRODUCTS, g_pos) if v))

    print(f"\n  Computing Greeks with CRN ({N_GREEK_PATHS:,} paths, parallel) …",
          end="", flush=True)
    t_gr = time.time()
    greeks = compute_product_greeks(N_GREEK_PATHS)
    print(f"  done in {time.time()-t_gr:.1f}s")

    # Console table
    print_frontier_table(front_pos, fe_full, fdv_full)

    # ── PDF output ────────────────────────────────────────────────────────────
    print(f"\nWriting 6-page PDF → {PDF_OUT} …")
    with PdfPages(PDF_OUT) as pdf:
        print("  page 1: frontier scatter …")
        page_frontier_scatter(pdf, all_e, all_dv, front_idx, front_pos,
                              fe_full, fdv_full, g_e, g_dv)
        print("  page 2: frontier table …")
        page_frontier_table(pdf, front_pos, fe_full, fdv_full)
        print("  page 3: per-asset distributions …")
        page_asset_distributions(pdf, pm_full)
        print("  page 4: Greeks …")
        page_greeks(pdf, greeks)
        print("  page 5: portfolio PnL distributions …")
        page_pnl_distributions(pdf, front_pos, fe_full, fdv_full,
                               g_pos, but_f, sut_f)
        print("  page 6: greedy analysis …")
        page_greedy_analysis(pdf, pm_full, g_pos, g_e, g_dv,
                             all_e, all_dv, fe_full, fdv_full, front_pos)

    t_tot = time.time() - t_start
    print(f"\n  PDF saved → {PDF_OUT}")
    print(f"\n{sep}")
    print(f"  Total elapsed: {t_tot/60:.1f} min  ({t_tot:.0f}s)")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()

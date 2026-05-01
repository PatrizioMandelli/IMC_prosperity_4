"""
# Manual Trading Challenge:

## Game Description
You trade against counterparties with reserve prices r drawn uniformly from
{670, 675, 680, ..., 920} (51 values, step 5). Fair value = 920 SeaShells.

For each counterparty with reserve r, given your bids (b1, b2):
  • b1 > r          → trade at b1:  profit = 920 - b1
  • b2 > r > b1:
      – b2 > μ₂     → trade at b2 (full):  profit = 920 - b2
      – b2 ≤ μ₂     → penalised profit:  (920 - b2) · [(920 - μ₂)/(920 - b2)]³
                                        = (920 - μ₂)³ / (920 - b2)²
    where μ₂ = mean second bid across ALL players (externality).

## Game-Theory Framework

### Strategy Space
  S = {(b1, b2) | b1, b2 ∈ {670,675,...,920}}    |S| = 51² = 2601

### Expected Payoff (analytical)
  U(b1, b2; μ₂) = (1/51) · Σ_{r∈R} π(b1, b2, r, μ₂)

  The dependence on μ₂ makes this a mean-field game: each player is a price-taker
  with respect to the population's second-bid distribution.

### Nash Equilibrium Condition
  In a symmetric Nash equilibrium σ*, every strategy in the support maximises U(·; μ₂*),
  where μ₂* = E_{σ*}[b2]. The second-bid mechanism creates a coordination game:
  if all players bid high on b2, the threshold rises and penalties disappear.
  If all players bid low, the threshold falls and high b2 bids become profitable —
  classic strategic complementarity.

### Quantal Response Equilibrium (QRE)  [McKelvey & Palfrey 1995]
  Replaces best-response with the logit map:
    σ(s) = exp(λ · U(s; σ)) / Z(σ),   Z = partition function

  λ=0 → uniform (infinite noise),  λ→∞ → Nash best-response

  Damped fixed-point iteration:  σ_{t+1} = (1-α)σ_t + α · Softmax(λ · U(σ_t))

### Fictitious Play  [Brown 1951, Robinson 1951]
  Each agent best-responds (with ε-noise) to the empirical distribution of past play:
    BR_ε(ĥ_t) = argmax_s [U(s; ĥ_t) + εᵢ],   εᵢ ∼ N(0, ξ²)
    ĥ_{t+1}  = (t·ĥ_t + δ_{BR_ε}) / (t+1)    (running average)

  The ε-noise implements bounded rationality and breaks limit cycles.

### Monte Carlo Replicator Dynamics  [Taylor & Jonker 1978]
  Simulates N finite players. Each iteration:
    1. Sample strategies {sᵢ} from σ  (stochastic population)
    2. Realized μ₂ = average b2 in the sample  (finite-sample noise)
    3. Compute analytical payoffs at realized μ₂
    4. Replicator update:
         σ_{t+1}(s) ∝ σ_t(s) · exp(U(s; μ₂_realized) / T)
       where T is temperature (rationality noise).

  The finite-population sampling introduces genuine randomness, modelling
  uncertainty about how many other players there are and what they bid.
  The replicator equation d σ/dt = σ(s)[U(s) - Ū] is the continuous limit.
"""



import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.markers import MarkerStyle

# ============================================================
# PARAMETERS
# ============================================================

FAIR_VALUE = 920
RES_MIN = 670
STEP = 5

RESERVES = np.arange(RES_MIN, FAIR_VALUE + STEP, STEP, dtype=float)  # 51 values
BID_GRID = np.arange(RES_MIN, FAIR_VALUE + STEP, STEP, dtype=float)  # 51 values
N_R = len(RESERVES)  # 51
N_B = len(BID_GRID)  # 51

STRATEGIES = [(b1, b2) for b1 in BID_GRID for b2 in BID_GRID]
N_S = len(STRATEGIES)  # 2601
b1_arr = np.array([s[0] for s in STRATEGIES], dtype=float)
b2_arr = np.array([s[1] for s in STRATEGIES], dtype=float)

# ---- Unified noise / rationality parameter ----
#
# NOISE_LEVEL η ∈ [0, 1] is the single knob controlling rationality across
# all three methods.  It maps to a rationality temperature τ via a log-uniform
# schedule so that each unit step in η doubles τ:
#
#   τ(η) = τ_min · (τ_max / τ_min)^η           [log-uniform]
#
# This is the *inverse rationality* of the population.  The three methods then
# derive their specific noise parameters coherently from τ:
#
#   QRE:  λ = 1/τ
#         σ(s) ∝ exp(λ·U(s))  →  as τ→0 (η→0): pure Nash BR
#                               →  as τ→∞ (η→1): uniform randomisation
#
#   MC:   T = τ
#         σ_{t+1}(s) ∝ σ_t(s)·exp(U(s)/T)  →  same Boltzmann form as QRE
#
#   FP:   ξ = τ · √(π/2)
#         BR_ε = argmax [U(s) + N(0,ξ²)]
#         Probit-logit matching: P(choose s*|ΔU) = Φ(ΔU / (ξ√2))
#                                             ≈ sigmoid(λ·ΔU)  with  λ = √(π/2)/ξ = 1/τ  ✓
#
# All three methods therefore have the same effective rationality temperature τ.

NOISE_LEVEL  = 0.10   # η — change this to tune all methods simultaneously
TAU_MIN      = 0.1    # τ at η=0  (near-Nash, SeaShells)
TAU_MAX      = 50.0   # τ at η=1  (near-random, SeaShells)
_LOG_RATIO   = np.log(TAU_MAX / TAU_MIN)   # ln(500) ≈ 6.21

def _tau(eta: float = NOISE_LEVEL) -> float:
    """Rationality temperature for noise level η."""
    return TAU_MIN * np.exp(eta * _LOG_RATIO)

# Derived per-method parameters (coherent with NOISE_LEVEL)
QRE_LAMBDA     = 1.0 / _tau()                      # logit rationality
MC_TEMPERATURE = _tau()                             # Boltzmann temperature
FP_NOISE       = _tau() * np.sqrt(np.pi / 2)       # Gaussian std (probit-logit match)

# ---- Other hyper-parameters ----
QRE_DAMPING  = 0.25  # step size α for damped QRE iteration
MC_N_PLAYERS = 1000  # finite population size for MC simulation
N_ITER       = 500   # max iterations per method
PATIENCE     = 50    # early-stop: consecutive iters below tolerance
CONV_TOL     = 0.05  # convergence tolerance (SeaShells)
RANDOM_SEED  = 42

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# PAYOFF COMPUTATION 
# ============================================================


def payoff_matrix(avg_b2: float) -> np.ndarray:
    """
    U(b1, b2; μ₂) — expected profit for every strategy (N_S,).

    Iterates over the 51 reserve values; each iteration is a vectorised
    operation over all 2601 strategies.  O(N_R · N_S) = O(132k).

    Penalty branch (b2 ≤ μ₂):
        effective_profit = (920 - μ₂)³ / (920 - b2)²
    Note: when b2 = 920, spread = 0 anyway so the branch contributes 0.
    """
    payoffs = np.zeros(N_S)
    p_r = 1.0 / N_R

    for r in RESERVES:
        m1 = b1_arr > r  # first-bid wins
        m2 = (~m1) & (b2_arr > r)  # second-bid eligible
        m2_full = m2 & (b2_arr > avg_b2)  # above threshold → full
        m2_pen = m2 & (b2_arr <= avg_b2)  # below threshold → penalised

        payoffs[m1] += p_r * (FAIR_VALUE - b1_arr[m1])
        payoffs[m2_full] += p_r * (FAIR_VALUE - b2_arr[m2_full])

        if np.any(m2_pen):
            denom = FAIR_VALUE - b2_arr[m2_pen]
            safe_denom = np.where(denom > 1e-9, denom, 1e-9)
            payoffs[m2_pen] += p_r * (FAIR_VALUE - avg_b2) ** 3 / safe_denom**2

    return payoffs


# ============================================================
# HELPER: patience tracker
# ============================================================


class PatienceTracker:
    def __init__(self, patience: int, tol: float):
        self.patience = patience
        self.tol = tol
        self._count = 0
        self._prev = (0.0, 0.0)

    def step(self, eb1: float, eb2: float) -> bool:
        """Returns True if converged."""
        if abs(eb1 - self._prev[0]) < self.tol and abs(eb2 - self._prev[1]) < self.tol:
            self._count += 1
        else:
            self._count = 0
        self._prev = (eb1, eb2)
        return self._count >= self.patience


# ============================================================
# METHOD 1 — QUANTAL RESPONSE EQUILIBRIUM (QRE)
# ============================================================


def run_qre(
    n_iter=N_ITER,
    lam=QRE_LAMBDA,
    alpha=QRE_DAMPING,
    patience=PATIENCE,
    tol=CONV_TOL,
    noise_level=None,
    verbose=True,
):
    """
    Damped logit-QRE fixed-point iteration.

      σ_{t+1} = (1 - α)·σ_t  +  α · Softmax(λ · U(·; μ₂(σ_t)))

    Damping (α < 1) prevents oscillation around the fixed point.
    At convergence, σ* is a logit-QRE: every player's mixed strategy is
    proportional to the exponential of their expected payoff.

    noise_level: if given, overrides lam via the unified τ(η) schedule.
    """
    if noise_level is not None:
        lam = 1.0 / _tau(noise_level)
    sigma = np.ones(N_S) / N_S
    hist = {"avg_b1": [], "avg_b2": [], "best_b1": [], "best_b2": []}
    pt = PatienceTracker(patience, tol)

    for t in range(n_iter):
        avg_b2 = float(sigma @ b2_arr)
        U = payoff_matrix(avg_b2)
        U -= U.max()  # shift for numerical stability (log-sum-exp trick)

        qr = np.exp(lam * U)
        qr /= qr.sum()
        sigma = (1 - alpha) * sigma + alpha * qr

        eb1 = float(sigma @ b1_arr)
        eb2 = float(sigma @ b2_arr)
        bi = int(np.argmax(sigma))
        bb1, bb2 = STRATEGIES[bi]
        hist["avg_b1"].append(eb1)
        hist["avg_b2"].append(eb2)
        hist["best_b1"].append(bb1)
        hist["best_b2"].append(bb2)

        if verbose:
            print(
                f"[QRE  t={t:3d}]  E[b1]={eb1:6.1f}  E[b2]={eb2:6.1f}  "
                f"mode=({bb1},{bb2})"
            )

        if pt.step(eb1, eb2):
            if verbose:
                print(f"  → QRE converged at t={t}  (patience={patience})")
            break

    return sigma, hist


# ============================================================
# METHOD 2 — FICTITIOUS PLAY  (perturbed best response)
# ============================================================


def run_fictitious_play(
    n_iter=N_ITER,
    noise_scale=FP_NOISE,
    patience=PATIENCE,
    tol=CONV_TOL,
    noise_level=None,
    verbose=True,
):
    """
    Fictitious play with Gaussian payoff perturbations (ε-rationality).

    At each step the agent best-responds to the empirical frequency of past
    play, with i.i.d. Gaussian noise added to each strategy's payoff
    before argmax.  The perturbation prevents deterministic cycling and
    models heterogeneous rationality levels across participants.

    Empirical frequency update (Brown's procedure):
        ĥ_{t+1} = t/(t+1) · ĥ_t  +  1/(t+1) · δ_{BR_ε(ĥ_t)}

    noise_level: if given, overrides noise_scale via ξ = τ(η)·√(π/2).
    """
    if noise_level is not None:
        noise_scale = _tau(noise_level) * np.sqrt(np.pi / 2)
    hist_freq = np.ones(N_S) / N_S
    hist = {"avg_b1": [], "avg_b2": [], "best_b1": [], "best_b2": []}
    pt = PatienceTracker(patience, tol)

    for t in range(n_iter):
        avg_b2 = float(hist_freq @ b2_arr)
        U = payoff_matrix(avg_b2)

        noise = rng.normal(0.0, noise_scale, size=N_S)
        br_idx = int(np.argmax(U + noise))

        delta = np.zeros(N_S)
        delta[br_idx] = 1.0
        hist_freq = t / (t + 1) * hist_freq + 1 / (t + 1) * delta

        eb1 = float(hist_freq @ b1_arr)
        eb2 = float(hist_freq @ b2_arr)
        bb1, bb2 = STRATEGIES[br_idx]
        hist["avg_b1"].append(eb1)
        hist["avg_b2"].append(eb2)
        hist["best_b1"].append(bb1)
        hist["best_b2"].append(bb2)

        if verbose:
            print(
                f"[FP   t={t:3d}]  E[b1]={eb1:6.1f}  E[b2]={eb2:6.1f}  "
                f"BR=({bb1},{bb2})"
            )

        if pt.step(eb1, eb2):
            if verbose:
                print(f"  → Fictitious Play converged at t={t}  (patience={patience})")
            break

    return hist_freq, hist


# ============================================================
# METHOD 3 — MONTE CARLO REPLICATOR DYNAMICS
# ============================================================


def run_monte_carlo(
    n_iter=N_ITER,
    n_players=MC_N_PLAYERS,
    temperature=MC_TEMPERATURE,
    patience=PATIENCE,
    tol=CONV_TOL,
    noise_level=None,
    verbose=True,
):
    """
    Finite-population evolutionary simulation.

    Each iteration:
      1. Sample n_players strategies stochastically from current σ.
      2. Compute REALIZED μ₂ = mean b2 of the sampled population.
         → Introduces genuine randomness: μ₂ fluctuates each round,
           just like real competition where player count is unknown.
      3. Compute analytical payoffs U(·; μ₂_realized).
      4. Replicator dynamics update (discrete-time, multiplicative):
             log σ_{t+1} ∝ log σ_t  +  U / T
         which is the Gibbs sampler form of:
             σ_{t+1}(s) = σ_t(s)·exp(U(s)/T) / Σ_{s'} σ_t(s')·exp(U(s')/T)

    T (temperature) controls selection pressure:
      T→0: winner-takes-all (only best strategy survives)
      T→∞: neutral drift (strategies equally likely)

    noise_level: if given, overrides temperature via T = τ(η).
    """
    if noise_level is not None:
        temperature = _tau(noise_level)
    sigma = np.ones(N_S) / N_S
    hist = {
        "avg_b1": [],
        "avg_b2": [],
        "best_b1": [],
        "best_b2": [],
        "realized_avg_b2": [],
    }
    pt = PatienceTracker(patience, tol)

    for t in range(n_iter):
        # Step 1 — sample finite population
        player_idx = rng.choice(N_S, size=n_players, p=sigma)
        realized_avg_b2 = float(b2_arr[player_idx].mean())

        # Step 2 — payoffs at realized μ₂
        U = payoff_matrix(realized_avg_b2)

        # Step 3 — replicator update (numerically stable via log-space)
        log_sigma = np.log(sigma + 1e-300) + U / temperature
        log_sigma -= log_sigma.max()
        sigma = np.exp(log_sigma)
        sigma /= sigma.sum()

        eb1 = float(sigma @ b1_arr)
        eb2 = float(sigma @ b2_arr)
        bi = int(np.argmax(sigma))
        bb1, bb2 = STRATEGIES[bi]
        hist["avg_b1"].append(eb1)
        hist["avg_b2"].append(eb2)
        hist["best_b1"].append(bb1)
        hist["best_b2"].append(bb2)
        hist["realized_avg_b2"].append(realized_avg_b2)

        if verbose:
            print(
                f"[MC   t={t:3d}]  E[b1]={eb1:6.1f}  E[b2]={eb2:6.1f}  "
                f"realized_μ₂={realized_avg_b2:6.1f}"
            )

        if pt.step(eb1, eb2):
            if verbose:
                print(f"  → Monte Carlo converged at t={t}  (patience={patience})")
            break

    return sigma, hist


# ============================================================
# SENSITIVITY: best response as a function of assumed μ₂
# ============================================================


def best_response_sweep(mu2_grid=None):
    if mu2_grid is None:
        mu2_grid = np.arange(750, 920, 5, dtype=float)
    rows = []
    for mu2 in mu2_grid:
        U = payoff_matrix(float(mu2))
        bidx = int(np.argmax(U))
        b1o, b2o = STRATEGIES[bidx]
        rows.append((mu2, b1o, b2o, U[bidx]))
    return rows


# ============================================================
# ROBUSTNESS SWEEP — vary NOISE_LEVEL, track equilibrium bids
# ============================================================

def robustness_sweep(
    eta_grid=None,
    mc_runs: int = 3,
    sweep_n_iter: int = 100,
    sweep_patience: int = 15,
):
    """
    Run all three methods across a grid of noise levels η and record
    the equilibrium E[b1], E[b2] and mode bids at each level.

    Reveals *robustness*: bid recommendations that are stable across a
    wide range of η are robust to uncertainty about opponent rationality.

    Parameters
    ----------
    eta_grid   : array of η values in [0,1], default linspace(0.05, 0.95, 15)
    mc_runs    : number of independent MC runs per η (averages stochasticity)
    sweep_n_iter / sweep_patience : reduced limits for speed

    Returns
    -------
    dict keyed by method name, each value a dict of lists:
      'eta', 'tau', 'lambda', 'eb1', 'eb2', 'mode_b1', 'mode_b2'
    """
    if eta_grid is None:
        eta_grid = np.linspace(0.05, 0.95, 15)

    kw = dict(n_iter=sweep_n_iter, patience=sweep_patience, verbose=False)
    res = {m: dict(eta=[], tau=[], lam=[], eb1=[], eb2=[], mode_b1=[], mode_b2=[])
           for m in ("QRE", "FP", "MC")}

    print(f"\n[Robustness sweep: {len(eta_grid)} noise levels × 3 methods × {mc_runs} MC runs]")
    for k, eta in enumerate(eta_grid):
        tau   = _tau(eta)
        lam_e = 1.0 / tau
        print(f"  η={eta:.2f}  τ={tau:.2f}  λ={lam_e:.3f}", end="  ... ", flush=True)

        # QRE (deterministic given η)
        sg, _ = run_qre(noise_level=eta, **kw)
        bi    = int(np.argmax(sg))
        res["QRE"]["eta"].append(eta);  res["QRE"]["tau"].append(tau)
        res["QRE"]["lam"].append(lam_e)
        res["QRE"]["eb1"].append(float(sg @ b1_arr))
        res["QRE"]["eb2"].append(float(sg @ b2_arr))
        res["QRE"]["mode_b1"].append(int(STRATEGIES[bi][0]))
        res["QRE"]["mode_b2"].append(int(STRATEGIES[bi][1]))

        # Fictitious Play
        sf, _ = run_fictitious_play(noise_level=eta, **kw)
        bi    = int(np.argmax(sf))
        res["FP"]["eta"].append(eta);   res["FP"]["tau"].append(tau)
        res["FP"]["lam"].append(lam_e)
        res["FP"]["eb1"].append(float(sf @ b1_arr))
        res["FP"]["eb2"].append(float(sf @ b2_arr))
        res["FP"]["mode_b1"].append(int(STRATEGIES[bi][0]))
        res["FP"]["mode_b2"].append(int(STRATEGIES[bi][1]))

        # Monte Carlo — average mc_runs independent seeds
        mc_eb1, mc_eb2, mc_mb1, mc_mb2 = [], [], [], []
        for _ in range(mc_runs):
            sm, _ = run_monte_carlo(noise_level=eta, **kw)
            bi    = int(np.argmax(sm))
            mc_eb1.append(float(sm @ b1_arr)); mc_eb2.append(float(sm @ b2_arr))
            mc_mb1.append(int(STRATEGIES[bi][0])); mc_mb2.append(int(STRATEGIES[bi][1]))
        res["MC"]["eta"].append(eta);   res["MC"]["tau"].append(tau)
        res["MC"]["lam"].append(lam_e)
        res["MC"]["eb1"].append(float(np.mean(mc_eb1)))
        res["MC"]["eb2"].append(float(np.mean(mc_eb2)))
        res["MC"]["mode_b1"].append(int(np.median(mc_mb1)))
        res["MC"]["mode_b2"].append(int(np.median(mc_mb2)))

        print("done")

    return res


# ============================================================
# RUN ALL SIMULATIONS
# ============================================================

print("=" * 62)
print("  IMC PROSPERITY — CELESTIAL GARDENERS' GUILD")
print("  Game-Theoretic Bid Optimisation")
print("=" * 62)

print(f"\nStrategy grid: {N_B}×{N_B} = {N_S} pure strategies")
print(
    f"Reserve dist:  Uniform{{{RES_MIN},{RES_MIN+STEP},...,{FAIR_VALUE}}}  (N={N_R})\n"
)

print("─" * 62)
print("METHOD 1 — QRE  (λ={}, α={})".format(QRE_LAMBDA, QRE_DAMPING))
print("─" * 62)
sigma_qre, hist_qre = run_qre(verbose=True)

print("\n" + "─" * 62)
print("METHOD 2 — FICTITIOUS PLAY  (noise={})".format(FP_NOISE))
print("─" * 62)
sigma_fp, hist_fp = run_fictitious_play(verbose=True)

print("\n" + "─" * 62)
print(
    "METHOD 3 — MONTE CARLO REPLICATOR  (N={}, T={})".format(
        MC_N_PLAYERS, MC_TEMPERATURE
    )
)
print("─" * 62)
sigma_mc, hist_mc = run_monte_carlo(verbose=True)


# ============================================================
# FINAL SUMMARY
# ============================================================


def summarise(name, sigma):
    eb1 = float(sigma @ b1_arr)
    eb2 = float(sigma @ b2_arr)
    bi = int(np.argmax(sigma))
    b1o, b2o = STRATEGIES[bi]
    U = payoff_matrix(eb2)
    ep = U[bi]
    print(f"\n  [{name}]")
    print(f"    Equilibrium E[b1] = {eb1:.1f}    E[b2] = {eb2:.1f}")
    print(f"    Mode strategy:     b1 = {b1o}    b2 = {b2o}")
    print(f"    E[profit at mode] = {ep:.2f} SeaShells")
    return b1o, b2o, ep


print("\n" + "=" * 62)
print("  FINAL RECOMMENDATIONS")
print("=" * 62)
recs = []
for nm, sg in [
    ("QRE", sigma_qre),
    ("Fictitious Play", sigma_fp),
    ("Monte Carlo", sigma_mc),
]:
    recs.append(summarise(nm, sg))

# Consensus bid: median of the three mode recommendations
b1_votes = [r[0] for r in recs]
b2_votes = [r[1] for r in recs]
print(
    f"\n   CONSENSUS BID:  b1 = {int(np.median(b1_votes))}    b2 = {int(np.median(b2_votes))}"
)

print("\n" + "─" * 62)
print("  BEST RESPONSE SENSITIVITY  (pure BR vs assumed μ₂)")
print("─" * 62)
print(f"{'μ₂':>8}  {'b1_opt':>8}  {'b2_opt':>8}  {'E[profit]':>12}")
for mu2, b1o, b2o, ep in best_response_sweep():
    print(f"{mu2:8.0f}  {int(b1o):8d}  {int(b2o):8d}  {ep:12.2f}")

print("\n" + "─" * 62)
print(f"  NOISE LEVEL: η={NOISE_LEVEL}  τ={_tau():.2f}  λ={QRE_LAMBDA:.3f}  "
      f"T={MC_TEMPERATURE:.2f}  ξ={FP_NOISE:.2f}")
print("─" * 62)
rob = robustness_sweep()


# ============================================================
# PLOTS — multi-page PDF
# ============================================================
from matplotlib.backends.backend_pdf import PdfPages

PDF_PATH      = "round3-manual.pdf"
THRESHOLD_FRAC = 0.005   # cells below 0.5% of peak σ rendered black

methods_plot = [
    ("QRE",             hist_qre,  sigma_qre),
    ("Fictitious Play", hist_fp,   sigma_fp),
    ("Monte Carlo",     hist_mc,   sigma_mc),
]
MCOLS = ["steelblue", "darkorange", "mediumseagreen"]


def ptitle(fig, text):
    fig.text(0.5, 0.98, text, ha="center", va="top",
             fontsize=12, fontweight="bold")


def masked_heatmap(ax, sigma, title, star_color="cyan"):
    """
    Heatmap of σ(b1,b2) with black masking for near-zero probability.
    Cells below THRESHOLD_FRAC × peak are rendered solid black by
    setting vmin > 0 and cmap.set_under('black').
    """
    heat  = sigma.reshape(N_B, N_B)
    vmax  = heat.max()
    vmin  = vmax * THRESHOLD_FRAC
    cmap  = plt.get_cmap("inferno").copy()
    cmap.set_under("black")
    im = ax.imshow(heat, origin="lower", aspect="auto",
                   extent=[RES_MIN, FAIR_VALUE, RES_MIN, FAIR_VALUE],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    bi = int(np.argmax(sigma))
    b1o, b2o = STRATEGIES[bi]
    ax.scatter([b2o], [b1o], color=star_color, s=140, zorder=6,
               marker=MarkerStyle("x"), label=f" b1={int(b1o)}, b2={int(b2o)}")
    ax.set_xlabel("b2 (SeaShells)", fontsize=8)
    ax.set_ylabel("b1 (SeaShells)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    return im, int(b1o), int(b2o)


with PdfPages(PDF_PATH) as pdf:

    # ── PAGE 1: CONVERGENCE (one panel per method) ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ptitle(fig, "Page 1 — Convergence of E[b1] and E[b2] per Method")
    for ax, (name, h, _), c in zip(axes, methods_plot, MCOLS):
        ax.plot(h["avg_b1"], lw=2, label="E[b1]", color="steelblue")
        ax.plot(h["avg_b2"], lw=2, label="E[b2]", color="darkorange")
        if "realized_avg_b2" in h:
            ax.plot(h["realized_avg_b2"], lw=1, alpha=0.4, ls="--",
                    color="gray", label="realized μ₂")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Bid (SeaShells)")
        ax.set_ylim(RES_MIN - 5, FAIR_VALUE + 5)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 2: CROSS-METHOD CONVERGENCE (all on same axes) ────────────
    fig, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    ptitle(fig, "Page 2 — Cross-Method Comparison of Convergence")
    for (name, h, _), c in zip(methods_plot, MCOLS):
        ax2a.plot(h["avg_b1"], lw=2, label=name, color=c)
        ax2b.plot(h["avg_b2"], lw=2, label=name, color=c)
    ax2a.set_title("E[b1] — all methods"); ax2a.set_xlabel("Iteration")
    ax2a.set_ylabel("E[b1] (SeaShells)"); ax2a.grid(True, alpha=0.3); ax2a.legend()
    ax2b.set_title("E[b2] — all methods"); ax2b.set_xlabel("Iteration")
    ax2b.set_ylabel("E[b2] (SeaShells)"); ax2b.grid(True, alpha=0.3); ax2b.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 3: PER-METHOD HEATMAPS (black-masked) + CONSENSUS ─────────
    fig, axes3 = plt.subplots(1, 4, figsize=(19, 5))
    ptitle(fig, "Page 3 — Strategy Distribution σ(b1,b2)  [black = near-zero probability]")
    for ax, (name, _, sigma) in zip(axes3[:3], methods_plot):
        im, _, _ = masked_heatmap(ax, sigma, name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    sigma_cons = (sigma_qre + sigma_fp + sigma_mc) / 3.0
    sigma_cons /= sigma_cons.sum()
    im_c, b1c, b2c = masked_heatmap(axes3[3], sigma_cons, "Consensus (mean of 3)")
    plt.colorbar(im_c, ax=axes3[3], fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)

    # shared lists used by multiple pages from here on
    sigma_list = [sigma_qre, sigma_fp, sigma_mc, sigma_cons]
    name_list  = ["QRE", "Fictitious Play", "Monte Carlo", "Consensus"]
    consensus_mu2 = float(sigma_cons @ b2_arr)

    # ── PAGE 4: PAYOFF LANDSCAPE per method (4 panels) ─────────────────
    # For each method we compute U(b1,b2) at THAT method's own equilibrium μ₂,
    # so the heatmap reflects the self-consistent payoff surface each solver found.
    _b2_lin = np.linspace(RES_MIN, FAIR_VALUE, N_B)
    _b1_lin = np.linspace(RES_MIN, FAIR_VALUE, N_B)

    fig, axes_p4 = plt.subplots(1, 4, figsize=(20, 5))
    ptitle(fig, "Page 4 — Payoff Landscape  U(b1,b2) at each method's equilibrium μ₂  [★ = best response]")
    for ax_p4, nm, sigma in zip(axes_p4, name_list, sigma_list):
        mu2_p  = float(sigma @ b2_arr)
        U_p    = payoff_matrix(mu2_p)
        heat_p = U_p.reshape(N_B, N_B)
        im_p   = ax_p4.imshow(heat_p, origin="lower", aspect="auto",
                              extent=[RES_MIN, FAIR_VALUE, RES_MIN, FAIR_VALUE],
                              cmap="viridis")
        cs_p   = ax_p4.contour(_b2_lin, _b1_lin, heat_p,
                               levels=8, colors="white", linewidths=0.6, alpha=0.55)
        ax_p4.clabel(cs_p, inline=True, fontsize=6, fmt="%.0f")
        bi_p   = int(np.argmax(U_p))
        b1op, b2op = STRATEGIES[bi_p]
        ax_p4.scatter([b2op], [b1op], color="red", s=160, zorder=7,
                      marker="*", label=f"★ ({int(b1op)},{int(b2op)})")
        ax_p4.set_xlabel("b2"); ax_p4.set_ylabel("b1")
        ax_p4.set_title(f"{nm}\nμ₂={mu2_p:.0f}", fontsize=10, fontweight="bold")
        ax_p4.legend(fontsize=8)
        fig.colorbar(im_p, ax=ax_p4, fraction=0.046, pad=0.04, label="E[profit]")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 5: PAYOFF LANDSCAPE summary at consensus μ₂ ───────────────
    U_cons  = payoff_matrix(consensus_mu2)
    heat_U  = U_cons.reshape(N_B, N_B)   # [b1_idx, b2_idx]

    fig = plt.figure(figsize=(17, 6))
    ptitle(fig, f"Page 5 — Payoff Landscape  U(b1,b2) at consensus μ₂={consensus_mu2:.0f}  (1D slices)")
    gs4 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    ax4a = fig.add_subplot(gs4[0])
    im4 = ax4a.imshow(heat_U, origin="lower", aspect="auto",
                      extent=[RES_MIN, FAIR_VALUE, RES_MIN, FAIR_VALUE],
                      cmap="viridis")
    b2_grid_lin = np.linspace(RES_MIN, FAIR_VALUE, N_B)
    b1_grid_lin = np.linspace(RES_MIN, FAIR_VALUE, N_B)
    cs = ax4a.contour(b2_grid_lin, b1_grid_lin, heat_U,
                      levels=10, colors="white", linewidths=0.6, alpha=0.6)
    ax4a.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    bi_c = int(np.argmax(U_cons))
    b1oc, b2oc = STRATEGIES[bi_c]
    ax4a.scatter([b2oc], [b1oc], color="red", s=150, zorder=7,
                 marker=MarkerStyle("x"), label=f" BR ({int(b1oc)},{int(b2oc)})")
    ax4a.set_xlabel("b2"); ax4a.set_ylabel("b1")
    ax4a.set_title("U(b1,b2) + iso-profit contours", fontsize=9)
    ax4a.legend(fontsize=8)
    fig.colorbar(im4, ax=ax4a, fraction=0.046, pad=0.04, label="E[profit]")

    ax4b = fig.add_subplot(gs4[1])
    best_b2_per_b1 = heat_U.max(axis=1)   # max over b2 columns → per b1
    ax4b.plot(BID_GRID, best_b2_per_b1, lw=2, color="teal")
    ax4b.fill_between(BID_GRID, best_b2_per_b1, alpha=0.2, color="teal")
    ax4b.set_xlabel("b1 (SeaShells)"); ax4b.set_ylabel("max_{b2} U(b1,b2)")
    ax4b.set_title("Best achievable profit\nfor each first bid b1", fontsize=9)
    ax4b.grid(True, alpha=0.3)

    ax4c = fig.add_subplot(gs4[2])
    best_b1_per_b2 = heat_U.max(axis=0)   # max over b1 rows → per b2
    ax4c.plot(BID_GRID, best_b1_per_b2, lw=2, color="coral")
    ax4c.fill_between(BID_GRID, best_b1_per_b2, alpha=0.2, color="coral")
    ax4c.set_xlabel("b2 (SeaShells)"); ax4c.set_ylabel("max_{b1} U(b1,b2)")
    ax4c.set_title("Best achievable profit\nfor each second bid b2", fontsize=9)
    ax4c.grid(True, alpha=0.3)

    fig.subplots_adjust(top=0.88, wspace=0.38)
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 6: MARGINAL DISTRIBUTIONS p(b1) and p(b2) ─────────────────
    fig, axes5 = plt.subplots(2, 4, figsize=(18, 9))
    ptitle(fig, "Page 6 — Marginal Strategy Distributions  p(b1) and p(b2)")

    for col_i, (nm, sigma) in enumerate(zip(name_list, sigma_list)):
        heat    = sigma.reshape(N_B, N_B)
        p_b1    = heat.sum(axis=1)   # marginal over b2 → p(b1)
        p_b2    = heat.sum(axis=0)   # marginal over b1 → p(b2)
        mode_b1 = int(BID_GRID[np.argmax(p_b1)])
        mode_b2 = int(BID_GRID[np.argmax(p_b2)])

        ax_t = axes5[0, col_i]
        ax_b = axes5[1, col_i]

        ax_t.bar(BID_GRID, p_b1, width=STEP * 0.85, color="steelblue", alpha=0.85)
        ax_t.axvline(mode_b1, color="red", ls="--", lw=1.5, label=f"mode={mode_b1}")
        ax_t.set_title(f"{nm}: p(b1)", fontsize=10, fontweight="bold")
        ax_t.set_xlabel("b1"); ax_t.set_ylabel("Probability")
        ax_t.grid(True, alpha=0.3, axis="y"); ax_t.legend(fontsize=8)

        ax_b.bar(BID_GRID, p_b2, width=STEP * 0.85, color="darkorange", alpha=0.85)
        ax_b.axvline(mode_b2, color="red", ls="--", lw=1.5, label=f"mode={mode_b2}")
        ax_b.set_title(f"{nm}: p(b2)", fontsize=10, fontweight="bold")
        ax_b.set_xlabel("b2"); ax_b.set_ylabel("Probability")
        ax_b.grid(True, alpha=0.3, axis="y"); ax_b.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 7: BEST RESPONSE SENSITIVITY vs assumed μ₂ ────────────────
    mu2_sweep = np.arange(750, 921, 5, dtype=float)
    br_rows   = best_response_sweep(mu2_sweep)
    mu2_v  = np.array([r[0] for r in br_rows])
    b1_br  = np.array([int(r[1]) for r in br_rows])
    b2_br  = np.array([int(r[2]) for r in br_rows])
    ep_br  = np.array([r[3] for r in br_rows])

    fig, (ax6a, ax6b, ax6c) = plt.subplots(1, 3, figsize=(16, 5))
    ptitle(fig, "Page 7 — Best-Response Sensitivity  (pure BR vs assumed μ₂)")

    ax6a.step(mu2_v, b1_br, where="mid", lw=2, color="steelblue", label="b1_opt")
    ax6a.step(mu2_v, b2_br, where="mid", lw=2, color="darkorange", label="b2_opt")
    ax6a.axvline(consensus_mu2, color="gray", ls=":", lw=1.5,
                 label=f"consensus μ₂={consensus_mu2:.0f}")
    ax6a.set_xlabel("Assumed μ₂"); ax6a.set_ylabel("Optimal bid")
    ax6a.set_title("Optimal (b1*, b2*) as function of μ₂", fontsize=10)
    ax6a.legend(fontsize=9); ax6a.grid(True, alpha=0.3)

    ax6b.plot(mu2_v, ep_br, lw=2, color="mediumseagreen")
    ax6b.fill_between(mu2_v, ep_br, alpha=0.2, color="mediumseagreen")
    ax6b.set_xlabel("Assumed μ₂"); ax6b.set_ylabel("E[profit] (SeaShells)")
    ax6b.set_title("Max achievable E[profit] vs μ₂", fontsize=10)
    ax6b.grid(True, alpha=0.3)

    sc = ax6c.scatter(b2_br, ep_br, c=mu2_v, cmap="plasma", s=50, zorder=5)
    plt.colorbar(sc, ax=ax6c, label="μ₂")
    ax6c.set_xlabel("b2_opt"); ax6c.set_ylabel("E[profit]")
    ax6c.set_title("b2* vs profit  (coloured by μ₂)", fontsize=10)
    ax6c.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 7: PENALTY MECHANICS ───────────────────────────────────────
    b2_fine = np.linspace(671, 919, 400)
    mu2_refs = [780, 820, 860, 890, 910]

    fig, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))
    ptitle(fig, "Page 8 — Penalty Mechanics  [(920−μ₂)/(920−b2)]³")

    for mu2_r in mu2_refs:
        pen_factor  = (FAIR_VALUE - mu2_r)**3 / (FAIR_VALUE - b2_fine)**3
        pen_profit  = (FAIR_VALUE - mu2_r)**3 / (FAIR_VALUE - b2_fine)**2
        full_profit = FAIR_VALUE - b2_fine
        ax7a.plot(b2_fine, pen_factor, lw=2, label=f"μ₂={mu2_r}")
        ax7b.plot(b2_fine, pen_profit, lw=2, label=f"penalised μ₂={mu2_r}")

    ax7a.axhline(1.0, color="black", ls="--", lw=1.5, label="no penalty (b2>μ₂)")
    ax7a.set_xlabel("b2"); ax7a.set_ylabel("Penalty factor [(920−μ₂)/(920−b2)]³")
    ax7a.set_title("Fraction of full profit retained\nwhen b2 ≤ μ₂", fontsize=9)
    ax7a.legend(fontsize=8); ax7a.grid(True, alpha=0.3); ax7a.set_ylim(0, 1.4)

    ax7b.plot(b2_fine, FAIR_VALUE - b2_fine, "k--", lw=2, label="full profit (b2>μ₂)")
    ax7b.set_xlabel("b2"); ax7b.set_ylabel("Effective profit (SeaShells)")
    ax7b.set_title("Penalised vs full profit in SeaShells", fontsize=9)
    ax7b.legend(fontsize=8); ax7b.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 8: CUMULATIVE DISTRIBUTION of b1 and b2 ───────────────────
    fig, axes8 = plt.subplots(2, 4, figsize=(18, 9))
    ptitle(fig, "Page 9 — Cumulative Strategy Distributions  CDF(b1) and CDF(b2)")

    for col_i, (nm, sigma) in enumerate(zip(name_list, sigma_list)):
        heat  = sigma.reshape(N_B, N_B)
        p_b1  = heat.sum(axis=1)
        p_b2  = heat.sum(axis=0)
        cdf_b1 = np.cumsum(p_b1)
        cdf_b2 = np.cumsum(p_b2)

        ax_t = axes8[0, col_i]
        ax_b = axes8[1, col_i]

        ax_t.step(BID_GRID, cdf_b1, where="post", lw=2, color="steelblue")
        ax_t.fill_between(BID_GRID, cdf_b1, step="post", alpha=0.15, color="steelblue")
        ax_t.set_title(f"{nm}: CDF(b1)", fontsize=10, fontweight="bold")
        ax_t.set_xlabel("b1"); ax_t.set_ylabel("Cumulative probability")
        ax_t.grid(True, alpha=0.3); ax_t.set_ylim(0, 1.05)

        ax_b.step(BID_GRID, cdf_b2, where="post", lw=2, color="darkorange")
        ax_b.fill_between(BID_GRID, cdf_b2, step="post", alpha=0.15, color="darkorange")
        ax_b.set_title(f"{nm}: CDF(b2)", fontsize=10, fontweight="bold")
        ax_b.set_xlabel("b2"); ax_b.set_ylabel("Cumulative probability")
        ax_b.grid(True, alpha=0.3); ax_b.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 10: ROBUSTNESS ANALYSIS ────────────────────────────────────
    # Shows how equilibrium bids shift as NOISE_LEVEL (η) varies.
    # Robust strategies are flat lines — insensitive to opponent rationality.
    #
    # τ(η) mapping is shown in bottom-right so the reader can convert
    # noise level to the physical rationality temperature in SeaShells.

    _rob_cols = {"QRE": "steelblue", "FP": "darkorange", "MC": "mediumseagreen"}
    _eta_ref  = NOISE_LEVEL   # current setting (vertical reference line)

    fig, axes_r = plt.subplots(2, 3, figsize=(17, 9))
    ptitle(fig, f"Page 10 — Robustness of Equilibrium Bids vs Noise Level η  "
                f"(current η={_eta_ref:.2f}, τ={_tau():.2f})")

    ax_b1e, ax_b2e, ax_tau = axes_r[0]   # top row: E[b1], E[b2], τ mapping
    ax_b1m, ax_b2m, ax_agr = axes_r[1]   # bottom row: mode b1, mode b2, agreement

    for mname, col in _rob_cols.items():
        eta_v  = np.array(rob[mname]["eta"])
        eb1_v  = np.array(rob[mname]["eb1"])
        eb2_v  = np.array(rob[mname]["eb2"])
        mb1_v  = np.array(rob[mname]["mode_b1"])
        mb2_v  = np.array(rob[mname]["mode_b2"])

        ax_b1e.plot(eta_v, eb1_v, lw=2, color=col, label=mname)
        ax_b2e.plot(eta_v, eb2_v, lw=2, color=col, label=mname)
        ax_b1m.step(eta_v, mb1_v, where="mid", lw=2, color=col, label=mname)
        ax_b2m.step(eta_v, mb2_v, where="mid", lw=2, color=col, label=mname)

    for ax in (ax_b1e, ax_b2e, ax_b1m, ax_b2m):
        ax.axvline(_eta_ref, color="red", ls=":", lw=1.5, label=f"η={_eta_ref:.2f}")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax_b1e.set_title("E[b1] vs η", fontsize=10, fontweight="bold")
    ax_b1e.set_xlabel("Noise level η"); ax_b1e.set_ylabel("E[b1] (SeaShells)")
    ax_b2e.set_title("E[b2] vs η", fontsize=10, fontweight="bold")
    ax_b2e.set_xlabel("Noise level η"); ax_b2e.set_ylabel("E[b2] (SeaShells)")
    ax_b1m.set_title("Mode b1 vs η", fontsize=10, fontweight="bold")
    ax_b1m.set_xlabel("Noise level η"); ax_b1m.set_ylabel("Mode b1 (SeaShells)")
    ax_b2m.set_title("Mode b2 vs η", fontsize=10, fontweight="bold")
    ax_b2m.set_xlabel("Noise level η"); ax_b2m.set_ylabel("Mode b2 (SeaShells)")

    # τ(η) mapping — log-scale y so the exponential map is a straight line
    eta_fine = np.linspace(0, 1, 200)
    tau_fine = TAU_MIN * np.exp(eta_fine * _LOG_RATIO)
    ax_tau.semilogy(eta_fine, tau_fine, lw=2, color="purple")
    ax_tau.axvline(_eta_ref, color="red", ls=":", lw=1.5)
    ax_tau.axhline(_tau(), color="red", ls=":", lw=1.5,
                   label=f"τ={_tau():.2f} → λ={QRE_LAMBDA:.3f}")
    ax_tau.set_xlabel("Noise level η"); ax_tau.set_ylabel("Temperature τ (SeaShells, log)")
    ax_tau.set_title("τ(η) = τ_min·(τ_max/τ_min)^η", fontsize=9, fontweight="bold")
    ax_tau.legend(fontsize=8); ax_tau.grid(True, alpha=0.3, which="both")

    # Method agreement on b2 (inter-method std — low = robust consensus)
    eta_v = np.array(rob["QRE"]["eta"])
    b2_matrix = np.stack([np.array(rob[m]["eb2"]) for m in _rob_cols], axis=0)
    b2_std = b2_matrix.std(axis=0)
    ax_agr.fill_between(eta_v, 0, b2_std, alpha=0.35, color="crimson", label="std(E[b2])")
    ax_agr.plot(eta_v, b2_std, lw=2, color="crimson")
    ax_agr.axvline(_eta_ref, color="red", ls=":", lw=1.5)
    ax_agr.set_xlabel("Noise level η")
    ax_agr.set_ylabel("Std across methods (SeaShells)")
    ax_agr.set_title("Inter-method disagreement on E[b2]\n(low = robust consensus)", fontsize=9)
    ax_agr.legend(fontsize=8); ax_agr.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig); plt.close(fig)

    # ── PAGE 11: SUMMARY DASHBOARD ─────────────────────────────────────

    fig, ax9 = plt.subplots(figsize=(16, 7))
    ptitle(fig, "Page 11 — Summary Dashboard & Final Recommendations")
    ax9.axis("off")

    rows_summary = [["Method", "E[b1]", "E[b2]", "Mode b1", "Mode b2", "E[profit]"]]
    for nm, sigma in zip(name_list, sigma_list):
        eb1 = float(sigma @ b1_arr)
        eb2 = float(sigma @ b2_arr)
        bi  = int(np.argmax(sigma))
        b1o_s, b2o_s = STRATEGIES[bi]
        ep  = payoff_matrix(eb2)[bi]
        rows_summary.append([nm, f"{eb1:.1f}", f"{eb2:.1f}",
                              str(int(b1o_s)), str(int(b2o_s)), f"{ep:.2f}"])

    tbl = ax9.table(cellText=rows_summary[1:], colLabels=rows_summary[0],
                    bbox=[0.0, 0.25, 1.0, 0.65],   # [left, bottom, width, height] in axes coords
                    cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(13)
    tbl.scale(1.0, 2.4)

    # colour header row
    for j in range(len(rows_summary[0])):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # highlight consensus row
    for j in range(len(rows_summary[0])):
        tbl[(4, j)].set_facecolor("#f0f4c3")

    fig.text(0.5, 0.14,
             f"★  CONSENSUS BID:  b1 = {b1c}    b2 = {b2c}    "
             f"(consensus μ₂ = {consensus_mu2:.0f})",
             ha="center", fontsize=15, fontweight="bold", color="#c0392b")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"\n PDF saved → {PDF_PATH}  (11 pages)")


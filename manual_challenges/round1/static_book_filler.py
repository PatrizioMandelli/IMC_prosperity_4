"""
Bid strategy tester - single-price auction (Prosperity)
Tie-break: max (higher clearing price)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def find_clearing(bids, asks, tie='max'):
    prices = sorted(set(p for p, _, _ in bids + asks))
    best_v, best_p = -1, []
    for p in prices:
        d = sum(v for bp, v, _ in bids if bp >= p)
        s = sum(v for ap, v, _ in asks if ap <= p)
        vol = min(d, s)
        if vol > best_v:
            best_v, best_p = vol, [p]
        elif vol == best_v:
            best_p.append(p)
    if best_v <= 0:
        return None, 0
    if tie == 'min':   return min(best_p), best_v
    if tie == 'max':   return max(best_p), best_v
    return (min(best_p) + max(best_p)) / 2, best_v


def simulate_bid(bids, asks, my_price, my_volume, sell_price, fee, tie='max'):
    b = [(p, v, 'other') for p, v in bids] + [(my_price, my_volume, 'me')]
    a = [(p, v, 'other') for p, v in asks]

    clearing, total_exec = find_clearing(b, a, tie)
    if clearing is None:
        return {'clearing': None, 'fill': 0, 'profit': 0, 'ppu': 0}

    b.sort(key=lambda x: (-x[0], x[2] == 'me'))
    eligible = [o for o in b if o[0] >= clearing]

    remaining, my_fill = total_exec, 0
    for price, volume, owner in eligible:
        if remaining <= 0:
            break
        exec_vol = min(volume, remaining)
        if owner == 'me':
            my_fill = exec_vol
        remaining -= exec_vol

    ppu = sell_price - clearing - fee
    return {'clearing': clearing, 'fill': my_fill,
            'profit': my_fill * ppu, 'ppu': ppu}


# ── CONFIG ────────────────────────────────────────────────────────

ASSETS = [
    {
        'name': 'DRYLAND FLAX',
        'bids': [(30, 30000), (29, 5000), (28, 12000), (27, 28000)],
        'asks': [(28, 40000), (31, 20000), (32, 20000), (33, 30000)],
        'sell_price': 30,
        'fee': 0.0,
        'prices_to_test': [27, 28, 29, 30, 31, 32, 33],
    },
    {
        'name': 'EMBER MUSHROOM',
        'bids': [(20, 43000), (19, 17000), (18, 6000), (17, 5000),
                 (16, 10000), (15, 5000), (14, 10000), (13, 7000)],
        'asks': [(12, 20000), (13, 25000), (14, 35000), (15, 6000),
                 (16, 5000), (18, 10000), (19, 12000)],
        'sell_price': 20,
        'fee': 0.10,
        'prices_to_test': [13, 14, 15, 16, 17, 18, 19, 20],
    },
]

TIE_RULE = 'max'
VOLUMES = np.linspace(100, 100000, 100000).astype(int)


# ── ANALISI ───────────────────────────────────────────────────────

def analyze_asset(asset, tie_rule, volumes):
    bids, asks = asset['bids'], asset['asks']
    sp, fee = asset['sell_price'], asset['fee']
    prices = asset['prices_to_test']

    b0 = [(p, v, 'other') for p, v in bids]
    a0 = [(p, v, 'other') for p, v in asks]
    c0, v0 = find_clearing(b0, a0, tie_rule)

    results = {}
    best = {'profit': -np.inf}
    for p in prices:
        profits = np.zeros(len(volumes))
        for i, v in enumerate(volumes):
            r = simulate_bid(bids, asks, p, int(v), sp, fee, tie_rule)
            profits[i] = r['profit']
            if r['profit'] > best['profit']:
                best = {**r, 'price': p, 'volume': int(v)}
        results[p] = profits

    all_points = []
    for p in prices:
        for i, v in enumerate(volumes):
            if results[p][i] > 0:
                all_points.append((p, int(v), results[p][i]))
    all_points.sort(key=lambda x: -x[2])

    robustness = {}
    for rule in ['min', 'mid', 'max']:
        r = simulate_bid(bids, asks, best['price'], best['volume'], sp, fee, rule)
        robustness[rule] = r['profit']

    return {'name': asset['name'], 'baseline': (c0, v0), 'sp': sp, 'fee': fee,
            'results': results, 'prices': prices, 'best': best,
            'top10': all_points[:10], 'robustness': robustness,
            'bids': bids, 'asks': asks}


def print_analysis(a):
    print(f"\n{'═'*60}\n  {a['name']}\n{'═'*60}")
    c0, v0 = a['baseline']
    print(f"Baseline senza di te: clearing={c0}, volume={v0:,}")
    print(f"sell_price={a['sp']}, fee={a['fee']}")
    print(f"\n── TOP 10 (tie='{TIE_RULE}') ──")
    print(f"{'price':<8}{'volume':<10}{'profit':>12}")
    for p, v, pr in a['top10']:
        print(f"{p:<8}{v:<10,}{pr:>12,.1f}")
    b = a['best']
    print(f"\n── OTTIMO ──")
    print(f"bid @{b['price']} vol {b['volume']:,}")
    print(f"clearing={b['clearing']}  fill={b['fill']:,}  "
          f"margine={b['ppu']:.2f}  profit={b['profit']:,.1f}")
    print(f"Robustezza: min={a['robustness']['min']:,.0f}  "
          f"mid={a['robustness']['mid']:,.0f}  "
          f"max={a['robustness']['max']:,.0f}")


# ── PLOT ──────────────────────────────────────────────────────────

def plot_asset(ax1, ax2, a, volumes, tie_rule):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(a['prices'])))

    ax1.set_facecolor('#1a1a2e')
    for spine in ax1.spines.values():
        spine.set_color('#444')
    ax1.tick_params(colors='#ccc')
    ax1.axhline(0, color='#555', ls='--', lw=0.8)
    for i, p in enumerate(a['prices']):
        ax1.plot(volumes, a['results'][p], color=colors[i], label=f'@{p}', lw=1.6)
    ax1.set_title(f"{a['name']} — Profit vs Volume (tie={tie_rule})",
                  color='white', fontsize=11)
    ax1.set_xlabel('Volume', color='#aaa')
    ax1.set_ylabel('Profit', color='#aaa')
    ax1.legend(fontsize=7, facecolor='#111', labelcolor='white', ncol=2)
    ax1.grid(True, color='#2a2a2a', lw=0.4)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax1.scatter([a['best']['volume']], [a['best']['profit']],
                color='red', s=80, zorder=5)

    ax2.set_facecolor('#1a1a2e')
    vol_subset = np.linspace(1000, 100000, 5000).astype(int)
    hmap = np.zeros((len(a['prices']), len(vol_subset)))
    for i, p in enumerate(a['prices']):
        for j, v in enumerate(vol_subset):
            r = simulate_bid(a['bids'], a['asks'], p, int(v),
                             a['sp'], a['fee'], tie_rule)
            hmap[i, j] = r['profit']
    im = ax2.imshow(hmap, aspect='auto', cmap='RdYlGn', origin='lower')
    ax2.set_xticks(range(len(vol_subset)))
    ax2.set_xticklabels([f'{v//1000}k' for v in vol_subset],
                        color='#ccc', fontsize=7, rotation=45)
    ax2.set_yticks(range(len(a['prices'])))
    ax2.set_yticklabels(a['prices'], color='#ccc')
    ax2.set_title(f"{a['name']} — Heatmap", color='white', fontsize=11)
    ax2.set_xlabel('Volume', color='#aaa')
    ax2.set_ylabel('Bid Price', color='#aaa')
    for i in range(len(a['prices'])):
        for j in range(len(vol_subset)):
            val = hmap[i, j]
            if abs(val) > 100:
                ax2.text(j, i, f'{int(val/1000)}k', ha='center', va='center',
                         fontsize=6, color='black')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')


# ── MAIN ──────────────────────────────────────────────────────────

analyses = [analyze_asset(a, TIE_RULE, VOLUMES) for a in ASSETS]
for a in analyses:
    print_analysis(a)

fig, axes = plt.subplots(len(ASSETS), 2, figsize=(16, 6 * len(ASSETS)))
fig.patch.set_facecolor('#0f0f0f')
if len(ASSETS) == 1:
    axes = [axes]
for i, a in enumerate(analyses):
    plot_asset(axes[i][0], axes[i][1], a, VOLUMES, TIE_RULE)
plt.tight_layout()
plt.savefig('bid_sweep_all.png', dpi=130, facecolor='#0f0f0f', bbox_inches='tight')
print(f"\n→ Plot salvato: bid_sweep_all.png")

print(f"\n{'═'*60}\n  RIEPILOGO\n{'═'*60}")
total_min = 0
for a in analyses:
    b = a['best']
    print(f"  {a['name']}: bid @{b['price']} vol {b['volume']:,}  "
          f"→ min={a['robustness']['min']:,.0f}  "
          f"mid={a['robustness']['mid']:,.0f}  "
          f"max={a['robustness']['max']:,.0f}")
    total_min += min(a['robustness'].values())
print(f"\n  Floor totale: {total_min:,.0f}")
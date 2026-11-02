"""Proximity analysis: distance to nearest same-direction training condition
vs behavioral error, mean across seeds.

Distance: min Euclidean in 2D start space, within same mag × direction.
Z-score: x z-scored within magnitude (pooling across directions).
Error: mean normalized error across seeds per condition.
Metric: Pearson r between z-scored distance and z-scored error.

Multi-seed:
  python src/rnn/analysis/plot_proximity.py --version_dir rnn/models/model_c/fullrank
"""
import sys, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sstats
from scipy.spatial.distance import cdist

from src.rnn.analysis.model_loader import load_and_predict

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = ['Helvetica', 'DejaVu Sans']

SEEDS = [0, 2, 4, 5, 7, 9, 12, 16, 17, 19]
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def _get_seed_data(version_dir, seeds=SEEDS):
    version_dir = Path(version_dir)
    results = []
    for seed in seeds:
        model_dir = version_dir / f's{seed}'
        if not (model_dir / 'model.pt').exists():
            continue
        try:
            labels, pred, grounded, internals, task = load_and_predict(model_dir)
        except Exception as e:
            print(f'  skip s{seed}: {e}')
            continue
        cond_4d = np.array([[c[0][0], c[0][1], c[1][0], c[1][1]] for c in labels])
        targets = np.array([[c[2][0], c[2][1]] for c in labels])
        mags = np.linalg.norm(cond_4d[:, 2:4], axis=1)
        results.append((seed, cond_4d, targets, pred, grounded, mags))
    return results


def compute_proximity_mean_seeds(version_dir, seeds=SEEDS):
    """Compute proximity with mean error across seeds.

    Returns (zx, zy, r, p) where zx/zy are z-scored pooled arrays,
    r is Pearson r, p is its p-value.
    """
    seed_data = _get_seed_data(version_dir, seeds=seeds)
    if not seed_data:
        return np.array([]), np.array([]), np.nan, np.nan

    _, c4d_ref, tgt_ref, _, grounded_ref, mags_ref = seed_data[0]
    hold = grounded_ref == 0
    train = grounded_ref == 1

    all_errs = []
    for seed, c4d, tgt, pred, grounded, mags in seed_data:
        errs = np.linalg.norm(pred - tgt, axis=1) / np.maximum(mags, 1e-6)
        all_errs.append(errs[hold])
    mean_errs = np.stack(all_errs).mean(axis=0)

    hold_idx_map = np.where(hold)[0]

    mag_pool_x, mag_pool_y = defaultdict(list), defaultdict(list)
    for um in np.unique(np.round(mags_ref, 4)):
        mm = np.abs(mags_ref - um) < 0.01
        for d in DIRS:
            v = np.array(d) * um
            dm = (np.abs(c4d_ref[:, 2] - v[0]) < 0.01) & \
                 (np.abs(c4d_ref[:, 3] - v[1]) < 0.01)
            tm = train & mm & dm
            hm = hold & mm & dm
            if tm.sum() == 0 or hm.sum() == 0:
                continue
            hold_global = np.where(hm)[0]
            hold_local = np.array([np.where(hold_idx_map == gi)[0][0]
                                   for gi in hold_global])
            x = cdist(c4d_ref[hm, :2], c4d_ref[tm, :2]).min(axis=1)
            y = mean_errs[hold_local]
            mag_pool_x[um].append(x)
            mag_pool_y[um].append(y)

    zx, zy, rx, ry = [], [], [], []
    for um in sorted(mag_pool_x.keys()):
        xa = np.concatenate(mag_pool_x[um])
        ya = np.concatenate(mag_pool_y[um])
        if xa.std() < 1e-8 or ya.std() < 1e-8:
            continue
        zx.append((xa - xa.mean()) / xa.std())
        zy.append((ya - ya.mean()) / ya.std())
        rx.append(xa)
        ry.append(ya)

    if not zx:
        return np.array([]), np.array([]), np.nan, np.nan

    px = np.concatenate(zx)
    py = np.concatenate(zy)
    r, p = sstats.pearsonr(px, py)
    raw_x = np.concatenate(rx)
    raw_y = np.concatenate(ry)
    return px, py, r, p, raw_x, raw_y


def plot_proximity_multi(version_dir, out_dir, seeds=SEEDS, ylim=None,
                          clip_z=None):
    """Proximity scatter with mean-across-seeds error."""
    result = compute_proximity_mean_seeds(version_dir, seeds)
    if len(result[0]) == 0:
        print('No valid data')
        return
    px, py, r, p = result[:4]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fontsize = 10

    if clip_z is not None:
        keep = (np.abs(px) <= clip_z) & (np.abs(py) <= clip_z)
        px = px[keep]
        py = py[keep]

    star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
    print(f'  r={r:.4f}, p={p:.2e}, n={len(px)}')

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.scatter(px, py, color=(0.5, 0.5, 0.5), s=4, edgecolors='none',
               alpha=0.25)
    sns.regplot(x=px, y=py, ax=ax, color='k', scatter=False,
                line_kws=dict(linewidth=1.8))
    if ax.collections and len(ax.collections) > 1:
        ax.collections[-1].set_alpha(0.25)

    ax.text(0.03, 0.97, f'r={r:.2f}{star}',
            transform=ax.transAxes, fontsize=fontsize, va='top',
            fontfamily='Helvetica')
    ax.set_xlabel('Dist. to train (z)', fontsize=fontsize, fontfamily='Helvetica')
    ax.set_ylabel('Error (behav., z)', fontsize=fontsize, fontfamily='Helvetica')
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax.set_xticks([0, 1])
    ax.set_xlim(xl[0] - 0.2, xl[1] + 0.2)
    ax.set_yticks([0, 1])
    ax.set_ylim(yl[0] - 0.2, yl[1] + 0.2)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fontsize)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontfamily('Helvetica')

    fig.tight_layout()
    save = f'{out_dir}/proximity_zscore.pdf'
    fig.savefig(save, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seed list, e.g. "0,1,2,3"')
    args = parser.parse_args()

    out_dir = args.out_dir or args.version_dir
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS
    plot_proximity_multi(args.version_dir, out_dir, seeds=seeds)

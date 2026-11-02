"""PCA pseudo-population: 2x2 (top-down residual + full PCA) x (train vs holdout).

Row 1: top-down view — subtract per-panel mean time course, PCA on residual,
       start-centered, first 1/3 of flow. Shows pure directional divergence.
Row 2: full PCA (PC1 vs PC2 with ramp). Shows common integration axis.

For v1-v3: pseudo-population across seeds, fixed conditions.
For v4: single seed, per-seed best train / worst holdout.

Usage:
    python plot_pca_train_test.py
"""
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PROJ = Path(__file__).resolve().parents[5]   # CT_MJ_2026: vendored root (was the lab Analysis root)
sys.path.insert(0, str(PROJ))

from src.rnn.analysis.model_loader import load_and_predict

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

DIR_LABELS = ['Right', 'Up', 'Left', 'Down']
DIR_COLORS = ['#ffb900', '#00ff2b', '#005eff', '#b600ff']
SEEDS = [0, 2, 4, 5, 7, 9, 12, 16, 17, 19]

SELECTED_TRAIN = [
    ((-3, 1), (6, 0), (3, 1)),
    ((-2, -3), (0, 6), (-2, 3)),
    ((3, -3), (-6, 0), (-3, -3)),
    ((-1, 3), (0, -6), (-1, -3)),
]
SELECTED_HOLDOUT = [
    ((-3, -3), (6, 0), (3, -3)),
    ((-3, -3), (0, 6), (-3, 3)),
    ((3, 0), (-6, 0), (-3, 0)),
    ((-3, 3), (0, -6), (-3, -3)),
]


def extract_segments(h_int, labels, task, selected):
    dt = task.dt
    ball_speed = task.BALL_SPEED
    flow_on = int(np.round(task.FLOW_ONSET_TIME / dt))
    segments = []
    for sel in selected:
        for i, cond in enumerate(labels):
            if cond == sel:
                vec = np.array(cond[1])
                flow_off = int(np.round((task.FLOW_ONSET_TIME + np.linalg.norm(vec) / ball_speed) / dt))
                segments.append(h_int[i, flow_on:flow_off, :])
                break
        else:
            return None
    return segments


def _add_axis_label(ax, label_x, label_y):
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax_len = (xlim[1] - xlim[0]) * 0.12
    ox = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    oy = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([ox, ox + ax_len], [oy, oy], 'k-', lw=1, clip_on=False)
    ax.plot([ox, ox], [oy, oy + ax_len], 'k-', lw=1, clip_on=False)
    ax.text(ox + ax_len, oy - (ylim[1] - ylim[0]) * 0.03,
            label_x, ha='left', va='top', fontsize=7, fontfamily='Helvetica')
    ax.text(ox - (xlim[1] - xlim[0]) * 0.03, oy + ax_len,
            label_y, ha='right', va='bottom', fontsize=7, fontfamily='Helvetica')


def _plot_2x2(train_segs_list, hold_segs_list, out_path, train_title='Training',
              hold_title='Holdout', flip_pc1=False):
    """Plot 2x2: top-down residual (row 0) + full PCA (row 1).

    train_segs_list, hold_segs_list: each is [seg_right, seg_up, seg_left, seg_down]
    where each seg is (n_t, n_neurons).
    flip_pc1: if True, negate PC1 so trajectories go upward.
    """
    n_t = min(s.shape[0] for s in train_segs_list + hold_segs_list)
    train_segs_list = [s[:n_t] for s in train_segs_list]
    hold_segs_list = [s[:n_t] for s in hold_segs_list]
    n_t_third = n_t // 3

    train_stack = np.stack(train_segs_list)
    hold_stack = np.stack(hold_segs_list)

    train_resid = train_stack - train_stack.mean(axis=0, keepdims=True)
    hold_resid = hold_stack - hold_stack.mean(axis=0, keepdims=True)

    all_resid = np.concatenate([train_resid.reshape(-1, train_resid.shape[2]),
                                 hold_resid.reshape(-1, hold_resid.shape[2])], axis=0)
    pca_resid = PCA(n_components=2)
    pca_resid.fit(all_resid)
    ev_r1 = pca_resid.explained_variance_ratio_[0] * 100
    ev_r2 = pca_resid.explained_variance_ratio_[1] * 100

    all_raw = np.concatenate([train_stack.reshape(-1, train_stack.shape[2]),
                               hold_stack.reshape(-1, hold_stack.shape[2])], axis=0)
    pca_full = PCA(n_components=2)
    pca_full.fit(all_raw)
    ev_f1 = pca_full.explained_variance_ratio_[0] * 100
    ev_f2 = pca_full.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(2, 2, figsize=(6, 4.5),
                             gridspec_kw={'height_ratios': [1, 2]})

    for col, (resid_segs, raw_segs, title) in enumerate([
        (train_resid, train_stack, train_title),
        (hold_resid, hold_stack, hold_title),
    ]):
        ax = axes[0, col]
        for i in range(4):
            proj = pca_resid.transform(resid_segs[i])
            proj = proj - proj[0:1]
            ax.plot(-proj[:n_t_third, 0], -proj[:n_t_third, 1],
                    color=DIR_COLORS[i], lw=2, alpha=0.8,
                    label=DIR_LABELS[i] if col == 0 else None)
        ax.scatter(0, 0, color=(0.5, 0.5, 0.5), s=240, zorder=5,
                   edgecolor='white', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10, fontfamily='Helvetica')

        ax = axes[1, col]
        n_t_two_thirds = n_t * 2 // 3
        sign = -1 if flip_pc1 else 1
        for i in range(4):
            proj = pca_full.transform(raw_segs[i])
            ax.plot(proj[:n_t_two_thirds, 1], sign * proj[:n_t_two_thirds, 0],
                    color=DIR_COLORS[i], lw=2, alpha=0.8)
            ax.scatter(proj[0, 1], sign * proj[0, 0], color='gray', s=120, zorder=5,
                       marker='o', edgecolors='k', linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for row in range(2):
        xl = [axes[row, c].get_xlim() for c in range(2)]
        yl = [axes[row, c].get_ylim() for c in range(2)]
        xmin = min(x[0] for x in xl); xmax = max(x[1] for x in xl)
        ymin = min(y[0] for y in yl); ymax = max(y[1] for y in yl)
        for c in range(2):
            axes[row, c].set_xlim(xmin, xmax)
            axes[row, c].set_ylim(ymin, ymax)

    _add_axis_label(axes[0, 0], f'rPC1 ({ev_r1:.0f}%)', f'rPC2 ({ev_r2:.0f}%)')
    _add_axis_label(axes[1, 0], f'PC2 ({ev_f2:.0f}%)', f'PC1 ({ev_f1:.0f}%)')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    print(f'  Residual: rPC1={ev_r1:.1f}%, rPC2={ev_r2:.1f}%')
    print(f'  Full: PC1={ev_f1:.1f}%, PC2={ev_f2:.1f}%')


def run(version_dir, out_path, flip_pc1=False):
    """Pseudo-population PCA for v1-v3 (fixed conditions across seeds)."""
    version_dir = Path(version_dir)
    train_segs = [[] for _ in range(4)]
    hold_segs = [[] for _ in range(4)]

    for seed in SEEDS:
        model_dir = version_dir / f's{seed}'
        if not (model_dir / 'model.pt').exists():
            continue
        try:
            labels, pred, grounded, internals, task = load_and_predict(model_dir)
        except Exception as e:
            print(f'  skip s{seed}: {e}')
            continue
        h_int = internals['h_integrator'].numpy()

        ts = extract_segments(h_int, labels, task, SELECTED_TRAIN)
        hs = extract_segments(h_int, labels, task, SELECTED_HOLDOUT)
        if ts is None or hs is None:
            continue
        for d in range(4):
            train_segs[d].append(ts[d])
            hold_segs[d].append(hs[d])

    n_seeds = len(train_segs[0])
    if n_seeds == 0:
        print('No valid seeds')
        return

    pseudo_train = [np.concatenate(s, axis=1) for s in train_segs]
    pseudo_hold = [np.concatenate(s, axis=1) for s in hold_segs]

    _plot_2x2(pseudo_train, pseudo_hold, out_path, flip_pc1=flip_pc1)
    print(f'  {n_seeds} seeds, {pseudo_train[0].shape[1]} pseudo-neurons')


def run_single_seed(model_dir, out_path, seed_label=''):
    """Single-seed PCA for v4: pick best train / worst holdout per direction."""
    model_dir = Path(model_dir)
    labels, pred, grounded, internals, task = load_and_predict(model_dir)
    h_int = internals['h_integrator'].numpy()
    dt = task.dt
    ball_speed = task.BALL_SPEED
    flow_on = int(np.round(task.FLOW_ONSET_TIME / dt))

    c4d = np.array([[c[0][0], c[0][1], c[1][0], c[1][1]] for c in labels])
    tgt = np.array([[c[2][0], c[2][1]] for c in labels])
    mags = np.linalg.norm(c4d[:, 2:4], axis=1)
    errs = np.linalg.norm(pred - tgt, axis=1) / np.maximum(mags, 1e-6)
    dirs_list = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    train_segs, hold_segs = [], []
    for d in dirs_list:
        vec = (d[0] * 6, d[1] * 6)
        flow_off = int(np.round((task.FLOW_ONSET_TIME + 6.0 / ball_speed) / dt))

        tr = [(i, errs[i]) for i in range(len(labels))
              if labels[i][1] == vec and grounded[i] == 1]
        best_i = min(tr, key=lambda x: x[1])[0]
        train_segs.append(h_int[best_i, flow_on:flow_off, :])

        ho = [(i, errs[i]) for i in range(len(labels))
              if labels[i][1] == vec and grounded[i] == 0]
        worst_i = max(ho, key=lambda x: x[1])[0]
        hold_segs.append(h_int[worst_i, flow_on:flow_off, :])

    _plot_2x2(train_segs, hold_segs, out_path,
              train_title=f'Training (best)', hold_title=f'Holdout (worst)')

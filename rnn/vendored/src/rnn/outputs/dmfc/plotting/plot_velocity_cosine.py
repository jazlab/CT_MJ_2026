"""Velocity cosine + common vs residual analysis across rank sweeps.

Row 1: Pairwise cosine similarity of velocity vectors across 4 flow
       directions, projected into top 3 PCs.
       1D integration → cos≈1, 4 independent → cos≈-1/3.

Row 2: ||residual|| / ||common|| ratio over time.
       common = ||mean trajectory across 4 directions||
       residual = mean_dir ||h_dir(t) - mean(t)||
       Low ratio = common dynamics dominate (1D integration).

Used by Supplementary Fig. S4.1 (fig_supp_velocity_cosine.py).
"""
import sys
from pathlib import Path
from itertools import combinations

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

SEEDS = list(range(10))
DIRS_4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
MAG = 6


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return np.dot(a, b) / (na * nb)


def compute_velocity_cosine(model_dir):
    """Compute mean pairwise cosine of velocity vectors across 4 directions.

    PCA is fit on hidden states, then velocity is computed in PCA space.
    Returns array of shape (n_flow_steps-1,) or None on failure.
    """
    model_dir = Path(model_dir)
    if not (model_dir / 'model.pt').exists():
        return None
    try:
        labels, pred, grounded, internals, task = load_and_predict(model_dir)
    except Exception as e:
        print(f'  skip {model_dir.name}: {e}')
        return None

    h_int = internals['h_integrator'].numpy()
    dt = task.dt
    flow_on = int(np.round(task.FLOW_ONSET_TIME / dt))
    flow_off = int(np.round((task.FLOW_ONSET_TIME + MAG / task.BALL_SPEED) / dt))

    dir_trajs = []
    for d in DIRS_4:
        vec = (d[0] * MAG, d[1] * MAG)
        found = False
        for i, cond in enumerate(labels):
            if cond[1] == vec and grounded[i] == 1:
                dir_trajs.append(h_int[i, flow_on:flow_off, :])
                found = True
                break
        if not found:
            return None

    n_t = min(s.shape[0] for s in dir_trajs)
    dir_trajs = [s[:n_t] for s in dir_trajs]

    all_h = np.concatenate(dir_trajs, axis=0)
    pca = PCA(n_components=3)
    pca.fit(all_h)

    h_pc = [pca.transform(t) for t in dir_trajs]
    vel_pc = [h[1:] - h[:-1] for h in h_pc]

    n_steps = vel_pc[0].shape[0]
    cos_curve = np.full(n_steps, np.nan)
    for t in range(n_steps):
        pairs = []
        for i, j in combinations(range(4), 2):
            pairs.append(_cosine_sim(vel_pc[i][t], vel_pc[j][t]))
        cos_curve[t] = np.nanmean(pairs)

    dt = task.dt
    t_axis = np.arange(n_steps) * dt
    return cos_curve, t_axis


def compute_common_residual(model_dir):
    """Compute ||common|| and ||residual|| over time for 4 directions.

    Returns (common_norm, residual_norm, time_axis) or None on failure.
    common_norm: ||mean across 4 directions|| at each timestep
    residual_norm: mean over 4 dirs of ||h_dir(t) - mean(t)||
    """
    model_dir = Path(model_dir)
    if not (model_dir / 'model.pt').exists():
        return None
    try:
        labels, pred, grounded, internals, task = load_and_predict(model_dir)
    except Exception as e:
        print(f'  skip {model_dir.name}: {e}')
        return None

    h_int = internals['h_integrator'].numpy()
    dt = task.dt
    flow_on = int(np.round(task.FLOW_ONSET_TIME / dt))
    flow_off = int(np.round((task.FLOW_ONSET_TIME + MAG / task.BALL_SPEED) / dt))

    dir_trajs = []
    for d in DIRS_4:
        vec = (d[0] * MAG, d[1] * MAG)
        found = False
        for i, cond in enumerate(labels):
            if cond[1] == vec and grounded[i] == 1:
                dir_trajs.append(h_int[i, flow_on:flow_off, :])
                found = True
                break
        if not found:
            return None

    n_t = min(s.shape[0] for s in dir_trajs)
    dir_trajs = np.stack([s[:n_t] for s in dir_trajs])

    mean_traj = dir_trajs.mean(axis=0)
    common_norm = np.linalg.norm(mean_traj, axis=1)
    residuals = dir_trajs - mean_traj[None, :, :]
    residual_norm = np.mean([np.linalg.norm(residuals[d], axis=1) for d in range(4)], axis=0)

    t_axis = np.arange(n_t) * dt
    return common_norm, residual_norm, t_axis


def _aggregate_seeds(version_dir, ranks, seeds, compute_fn):
    """Aggregate a per-seed compute function across ranks.

    compute_fn(model_dir) should return a tuple of arrays or None.
    Returns dict: rank_dir -> (mean_arrays, sem_arrays, time_axis).
    """
    results = {}
    for rank_dir in ranks:
        base = Path(version_dir) / rank_dir
        all_results = []
        for seed in seeds:
            r = compute_fn(base / f's{seed}')
            if r is not None:
                all_results.append(r)
        if not all_results:
            continue
        n_t = min(r[-1].shape[0] for r in all_results)
        n_arrays = len(all_results[0]) - 1
        stacked = [np.array([r[k][:n_t] for r in all_results]) for k in range(n_arrays)]
        means = [s.mean(axis=0) for s in stacked]
        sems = [s.std(axis=0) / np.sqrt(len(all_results)) for s in stacked]
        t_axis = all_results[0][-1][:n_t]
        results[rank_dir] = (means, sems, t_axis, len(all_results))
    return results


def _aggregate_single(version_dir, seeds, compute_fn):
    """Aggregate compute_fn across seeds for a flat directory."""
    all_results = []
    for seed in seeds:
        r = compute_fn(Path(version_dir) / f's{seed}')
        if r is not None:
            all_results.append(r)
    if not all_results:
        return None
    n_t = min(r[-1].shape[0] for r in all_results)
    n_arrays = len(all_results[0]) - 1
    stacked = [np.array([r[k][:n_t] for r in all_results]) for k in range(n_arrays)]
    means = [s.mean(axis=0) for s in stacked]
    sems = [s.std(axis=0) / np.sqrt(len(all_results)) for s in stacked]
    t_axis = all_results[0][-1][:n_t]
    return means, sems, t_axis, len(all_results)


RANK_COLORS = {
    'r1': '#1f77b4',
    'r2': '#ff7f0e',
    'r4': '#2ca02c',
    'r16': '#d62728',
    'r32': '#9467bd',
    'rfull': '#8c564b',
}
RANK_LABELS = {
    'r1': 'r=1', 'r2': 'r=2', 'r4': 'r=4',
    'r16': 'r=16', 'r32': 'r=32', 'rfull': 'r=32',
}


def _plot_row(axes, conditions, compute_fn, ylabel, ref_lines=None,
              ylim=None, yticks=None):
    """Plot one row of the figure across 4 conditions."""
    fontsize = 10
    for ax_idx, (title, base_dir, ranks) in enumerate(conditions):
        ax = axes[ax_idx]
        if ranks is not None:
            data = _aggregate_seeds(base_dir, ranks, SEEDS, compute_fn)
            for rk in ranks:
                if rk not in data:
                    continue
                means, sems, t, n = data[rk]
                ax.plot(t, means[0], color=RANK_COLORS[rk], lw=1.5)
                ax.fill_between(t, means[0] - sems[0], means[0] + sems[0],
                                color=RANK_COLORS[rk], alpha=0.2)
                print(f'  {rk}: {n} seeds, final={means[0][-1]:.3f}')
        else:
            result = _aggregate_single(base_dir, SEEDS, compute_fn)
            if result is not None:
                means, sems, t, n = result
                ax.plot(t, means[0], color=RANK_COLORS['rfull'], lw=1.5)
                ax.fill_between(t, means[0] - sems[0], means[0] + sems[0],
                                color=RANK_COLORS['rfull'], alpha=0.2)
                print(f'  {n} seeds, final={means[0][-1]:.3f}')

        if ref_lines:
            for rl in ref_lines:
                ax.axhline(rl, color='gray', ls='--', lw=0.8)
        if ylim:
            ax.set_ylim(ylim)
        if yticks:
            ax.set_yticks(yticks)
        ax.set_xlim(0, 2.0)
        ax.set_xlabel('Time (s)', fontsize=fontsize, fontfamily='Helvetica')
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize - 1)
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontfamily('Helvetica')

    axes[0].set_ylabel(ylabel, fontsize=fontsize, fontfamily='Helvetica')

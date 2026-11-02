"""Neural trajectory deviation vs behavioral error (Fig. 4, S4.3, S5.3).

For each seed × holdout condition:
  1. Find same-mag, same-direction training condition with smallest error (template)
  2. Start-remove integrator trajectories (subtract flow-onset state)
  3. Neural deviation = ||h_holdout(t) - h_template(t)|| over flow period
  4. Behavioral error = ||pred - target|| / mag

Pool across seeds, z-score within magnitude, scatter + r(t).
"""
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sstats

PROJ = Path(__file__).resolve().parents[5]   # CT_MJ_2026: vendored root (was the lab Analysis root)
sys.path.insert(0, str(PROJ))

from src.rnn.analysis.model_loader import load_and_predict

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

PUB = PROJ / 'src/rnn/outputs/pub_figures'
SEEDS = [0, 2, 4, 5, 7, 9, 12, 16, 17, 19]
COLOR = (0.5, 0.5, 0.5)


def get_direction(vec):
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return None
    return tuple(np.round(vec / n, 4))


def run(version_dir, out_name, xlim=(-0.1, 1), ylim=(0, 2), seeds=None,
        mean_across_seeds=False, free_ylim=False, ylim_max=None):
    if seeds is None:
        seeds = SEEDS
    from collections import defaultdict
    cond_devs = defaultdict(list)
    cond_errs = defaultdict(list)
    cond_mags = {}
    all_neural_devs = []
    all_behav_err = []
    all_mags = []
    all_times = None

    for seed in seeds:
        model_dir = version_dir / f's{seed}'
        if not (model_dir / 'model.pt').exists():
            continue
        try:
            labels, pred, grounded, internals, task = load_and_predict(model_dir)
        except Exception as e:
            print(f'  skip s{seed}: {e}')
            continue

        h_int = internals['h_integrator'].numpy()
        dt = task.dt
        flow_on = int(np.round(task.FLOW_ONSET_TIME / dt))

        cond_4d = np.array([[c[0][0], c[0][1], c[1][0], c[1][1]] for c in labels])
        targets = np.array([[c[2][0], c[2][1]] for c in labels])
        mags = np.linalg.norm(cond_4d[:, 2:4], axis=1)
        errs = np.linalg.norm(pred - targets, axis=1)
        norm_errs = errs / np.maximum(mags, 1e-6)

        n_conds, n_t_total, n_h = h_int.shape
        n_flow = n_t_total - flow_on

        if all_times is None:
            all_times = np.arange(n_flow) * dt

        train_mask = grounded == 1
        hold_mask = grounded == 0
        ball_speed = task.BALL_SPEED

        for i_hold in np.where(hold_mask)[0]:
            vec_h = cond_4d[i_hold, 2:4]
            mag_h = mags[i_hold]
            dir_h = get_direction(vec_h)
            if dir_h is None or mag_h < 1e-6:
                continue

            best_template = None
            best_err = np.inf
            for i_train in np.where(train_mask)[0]:
                vec_t = cond_4d[i_train, 2:4]
                mag_t = mags[i_train]
                dir_t = get_direction(vec_t)
                if dir_t == dir_h and abs(mag_t - mag_h) < 0.01:
                    if norm_errs[i_train] < best_err:
                        best_err = norm_errs[i_train]
                        best_template = i_train

            if best_template is None:
                continue

            flow_off_idx = int(np.round(mag_h / ball_speed / dt))
            flow_off_idx = min(flow_off_idx, n_flow - 1)

            h_hold = h_int[i_hold, flow_on:flow_on + n_flow, :]
            h_tmpl = h_int[best_template, flow_on:flow_on + n_flow, :]
            sr_hold = h_hold - h_hold[0:1, :]
            sr_tmpl = h_tmpl - h_tmpl[0:1, :]
            dev = np.linalg.norm(sr_hold - sr_tmpl, axis=1)

            dev_val = dev[flow_off_idx]
            err_val = norm_errs[i_hold]
            mag_val = np.round(mag_h, 2)

            if mean_across_seeds:
                cond_key = (tuple(cond_4d[i_hold]), tuple(targets[i_hold]))
                cond_devs[cond_key].append(dev_val)
                cond_errs[cond_key].append(err_val)
                cond_mags[cond_key] = mag_val
            else:
                all_neural_devs.append(dev_val)
                all_behav_err.append(err_val)
                all_mags.append(mag_val)

        print(f'  s{seed}: {hold_mask.sum()} holdout conditions')

    if mean_across_seeds:
        keys = list(cond_devs.keys())
        x_raw = np.array([np.mean(cond_devs[k]) for k in keys])
        y_raw = np.array([np.mean(cond_errs[k]) for k in keys])
        m_raw = np.array([cond_mags[k] for k in keys])
    else:
        x_raw = np.array(all_neural_devs)
        y_raw = np.array(all_behav_err)
        m_raw = np.array(all_mags)

    # Z-score x within magnitude, keep y raw
    x, y = [], []
    for um in np.unique(np.round(m_raw, 2)):
        mask = np.abs(m_raw - um) < 0.01
        xm, ym = x_raw[mask], y_raw[mask]
        if xm.std() > 1e-4:
            x.append((xm - xm.mean()) / xm.std())
        else:
            x.append(np.zeros_like(xm))
        y.append(ym)
    x = np.concatenate(x)
    y = np.concatenate(y)

    print(f'\nTotal points: {len(x)}')
    print(f'Neural dev (z): mean={x.mean():.6f}, max={x.max():.6f}')
    print(f'Behav err: mean={y.mean():.4f}, max={y.max():.4f}')

    fontsize = 10
    fig, ax = plt.subplots(figsize=(2, 2))
    m = np.isfinite(x) & np.isfinite(y)
    rng = np.random.default_rng(0)
    has_neural_var = x_raw.max() > 1e-4
    x_jit = rng.normal(0, 0.05, size=m.sum())
    y_jit = rng.normal(0, 0.005, size=m.sum())
    ax.scatter(x[m] + x_jit, y[m] + y_jit, color=COLOR, s=4, edgecolors='none', alpha=0.25)
    if has_neural_var and x[m].std() > 1e-6:
        sns.regplot(x=x[m], y=y[m], ax=ax, color=COLOR, scatter=False,
                    line_kws=dict(linewidth=1.5))
        if ax.collections:
            ax.collections[-1].set_alpha(0.3)
        r_val, p_val = sstats.pearsonr(x[m], y[m])
        star = '***' if p_val < .001 else '**' if p_val < .01 else '*' if p_val < .05 else ''
        ax.text(0.05, 0.95, f'r={r_val:.2f}{star}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=fontsize - 1, fontfamily='Helvetica')
    if has_neural_var:
        ax.set_xlim((-3, 3))
        ax.set_xticks([0, 1])
    else:
        ax.set_xlim((-0.5, 5.5))
        ax.set_xticks([0, 1, 5])
    ymax = y[m].max()
    ytick_top = max(int(ymax), 1)
    if ylim_max is not None:
        ax.set_ylim((-0.1, ylim_max))
    else:
        ax.set_ylim((-0.1, ytick_top + 0.5))
    ax.set_yticks([0, ytick_top])
    ax.set_xlabel('Error (neural, z)', fontsize=fontsize, fontfamily='Helvetica')
    ax.set_ylabel('Error (behav.)', fontsize=fontsize, fontfamily='Helvetica')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fontsize - 1)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontfamily('Helvetica')
    fig.tight_layout()
    scatter_path = PUB / out_name
    fig.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {scatter_path}')

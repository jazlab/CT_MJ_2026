"""Causal perturbation analysis: perturb hidden state along common vs residual
axis and measure signed length change in the readout output.

Common axis: direction of mean trajectory across 4 cardinal directions (mag=6).
Residual axis: direction-specific deviation from mean at flow offset.

Usage:
    python src/rnn/analysis/plot_perturbation.py --version_dir rnn/models/model_c/fullrank
"""
import sys, argparse, numpy as np, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.rnn.analysis.model_loader import load_and_predict, load_model
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Helvetica'

DIRS_4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
MAG = 6


def compute_signed_slopes(model_dir, seeds, eps=0.2):
    """Compute per-seed mean signed slope of output length w.r.t. common/residual perturbation."""
    centers = make_retinotopic_centers(7)
    seed_common = []
    seed_resid = []

    for seed in seeds:
        mdir = Path(model_dir) / f's{seed}'
        if not (mdir / 'model.pt').exists():
            continue
        try:
            model, cfg, s1 = load_model(mdir)
            labels, pred, grounded, internals, task = load_and_predict(mdir)
        except:
            continue

        h_full = internals['h_integrator']
        h_np = h_full.numpy()
        dt = task.dt
        flow_on = int(np.round(task.FLOW_ONSET_TIME / dt))
        flow_off = int(np.round((task.FLOW_ONSET_TIME + MAG / task.BALL_SPEED) / dt))

        dir_trajs = []
        for d in DIRS_4:
            vec = (d[0] * MAG, d[1] * MAG)
            for i, c in enumerate(labels):
                if c[1] == vec and grounded[i] == 1:
                    dir_trajs.append(h_np[i, flow_on:flow_off, :])
                    break
        if len(dir_trajs) != 4:
            continue

        n_t = min(s.shape[0] for s in dir_trajs)
        dir_stack = np.stack([s[:n_t] for s in dir_trajs])
        mean_traj = dir_stack.mean(axis=0)
        common_vel = mean_traj[-1] - mean_traj[0]
        common_axis = common_vel / np.linalg.norm(common_vel)
        resid = dir_stack[0, -1] - mean_traj[-1]
        if np.linalg.norm(resid) < 1e-8:
            continue
        resid_axis = resid / np.linalg.norm(resid)

        common_axis_t = torch.tensor(common_axis, dtype=torch.float32)
        resid_axis_t = torch.tensor(resid_axis, dtype=torch.float32)

        hold_idx = [i for i in range(len(labels)) if grounded[i] == 0]

        c_slopes = []
        r_slopes = []
        for idx in hold_idx:
            cond = labels[idx]
            vec = np.array(cond[1])
            fo = int(np.round((task.FLOW_ONSET_TIME + np.linalg.norm(vec) / task.BALL_SPEED) / dt))
            h_snap = h_full[idx, fo, :].clone()
            start = np.array(cond[0], dtype=np.float32)
            retino = torch.tensor(retinotopic_encode(start.reshape(1, 2), centers, 1.0)[0],
                                  dtype=torch.float32)

            def _predict(h_pert):
                if hasattr(model, 'readout_mlp'):
                    return model.readout_mlp(torch.cat([h_pert, retino]).unsqueeze(0)).squeeze().numpy()
                else:
                    disp = model.disp_mlp(h_pert.unsqueeze(0)).squeeze()
                    inp = torch.cat([retino, disp])
                    return model.w_target(inp.unsqueeze(0)).squeeze().numpy()

            with torch.no_grad():
                len_cp = np.linalg.norm(_predict(h_snap + eps * common_axis_t) - start)
                len_cn = np.linalg.norm(_predict(h_snap - eps * common_axis_t) - start)
                len_rp = np.linalg.norm(_predict(h_snap + eps * resid_axis_t) - start)
                len_rn = np.linalg.norm(_predict(h_snap - eps * resid_axis_t) - start)

            c_slopes.append((len_cp - len_cn) / (2 * eps))
            r_slopes.append((len_rp - len_rn) / (2 * eps))

        seed_common.append(np.mean(c_slopes))
        seed_resid.append(np.mean(r_slopes))

    return np.array(seed_common), np.array(seed_resid)


def plot_perturbation_bar(version_dirs, labels, seeds, out_path):
    """Bar plot of signed slopes for one or more models."""
    fontsize = 10
    n = len(version_dirs)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.5))
    if n == 1:
        axes = [axes]

    for ax, vdir, label in zip(axes, version_dirs, labels):
        print(f'  Computing {label}...')
        c_vals, r_vals = compute_signed_slopes(vdir, seeds)

        x = np.arange(2)
        means = [c_vals.mean(), r_vals.mean()]
        sems = [c_vals.std() / np.sqrt(len(c_vals)), r_vals.std() / np.sqrt(len(r_vals))]

        ax.bar(x, means, width=0.6, color=['steelblue', 'coral'],
               yerr=sems, capsize=4)

        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.1, 0.1, len(c_vals))
        ax.scatter(np.zeros(len(c_vals)) + jitter, c_vals, color='k', s=15, alpha=0.5, zorder=5)
        ax.scatter(np.ones(len(r_vals)) + jitter, r_vals, color='k', s=15, alpha=0.5, zorder=5)

        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['Common', 'Residual'], fontsize=fontsize - 1)
        ax.set_title(label, fontsize=fontsize, fontfamily='Helvetica')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize - 1)
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontfamily('Helvetica')

    axes[0].set_ylabel('Length sensitivity\n(signed slope)', fontsize=fontsize, fontfamily='Helvetica')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version_dir', type=str, nargs='+', required=True)
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--seeds', type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else list(range(10))
    labels = args.labels or [Path(d).name for d in args.version_dir]

    plot_perturbation_bar(args.version_dir, labels, seeds, args.out)

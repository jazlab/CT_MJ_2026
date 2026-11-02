"""Fig. 4d PCA (neural-diff) for model d.

Model d (rnn/models/model_d): Embed(f=1) + pretrained, frozen full-rank RNN(32) → [h; embedded Start] → deep MLP(6x128)
→ target. The input embedding mixes Flow and Start before factorizing.

PCA: per seed, best training vs holdout with max start-removed neural diff (mag=6). Uses seed 3.
"""
import sys, shutil
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[5]   # CT_MJ_2026: vendored root (was the lab Analysis root)
sys.path.insert(0, str(PROJ))

from src.rnn.analysis.model_loader import load_and_predict
from src.rnn.outputs.dmfc.plotting.plot_pca import _plot_2x2
from src.rnn.analysis.plot_proximity import plot_proximity_multi
from src.rnn.outputs.pub_figures.scripts.fig_neural_vs_behavior import run as run_nvb

PUB = PROJ / 'src/rnn/outputs/pub_figures'
SEEDS = list(range(10))
PCA_SEED = 3

DIR_VECS = [(6, 0), (0, 6), (-6, 0), (0, -6)]


def run_pca_neuraldiff(model_dir, seed, out_path, inset_ms=300):
    """Single-seed PCA: best training vs holdout with max start-removed neural diff per direction."""
    sd = Path(model_dir) / f's{seed}'
    labels, pred, grounded, internals, task = load_and_predict(sd)
    h_int = internals['h_integrator'].numpy()
    dt = task.dt
    ball_speed = task.BALL_SPEED
    flow_on = int(np.round(task.FLOW_ONSET_TIME / dt))
    flow_off = int(np.round((task.FLOW_ONSET_TIME + 6.0 / ball_speed) / dt))

    cond_4d = np.array([[c[0][0], c[0][1], c[1][0], c[1][1]] for c in labels])
    targets = np.array([[c[2][0], c[2][1]] for c in labels])
    mags = np.linalg.norm(cond_4d[:, 2:4], axis=1)
    errs = np.linalg.norm(pred - targets, axis=1) / np.maximum(mags, 1e-6)

    train_segs, hold_segs = [], []
    for dvec in DIR_VECS:
        # Best training (smallest error)
        tr = [(i, errs[i]) for i in range(len(labels))
              if labels[i][1] == dvec and grounded[i] == 1]
        best_i = min(tr, key=lambda x: x[1])[0]
        h_tmpl = h_int[best_i, flow_on:flow_off, :]
        sr_tmpl = h_tmpl - h_tmpl[0:1, :]
        train_segs.append(h_tmpl)

        # Holdout with largest start-removed neural diff from template
        ho = [i for i in range(len(labels))
              if labels[i][1] == dvec and grounded[i] == 0]
        best_hold = None
        best_diff = -1
        for i_h in ho:
            h_h = h_int[i_h, flow_on:flow_off, :]
            sr_h = h_h - h_h[0:1, :]
            diff = np.linalg.norm(sr_h - sr_tmpl, axis=1).mean()
            if diff > best_diff:
                best_diff = diff
                best_hold = i_h
        hold_segs.append(h_int[best_hold, flow_on:flow_off, :])

    _plot_2x2(train_segs, hold_segs, out_path,
              train_title='Training (best)', hold_title='Holdout (max neural diff)',
              inset_ms=inset_ms)

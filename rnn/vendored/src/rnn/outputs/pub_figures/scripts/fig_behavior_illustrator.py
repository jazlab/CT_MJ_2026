"""Behavior illustrator: specific train/test conditions, mag=1 up.

Same style as fig_v3_behavior.py (black background, arrow plot).
Plots 3 training starts + 3 holdout starts for up direction, mag=1.

Train starts: (-2,-3), (-1,-2), (2,0)
Test starts:  (-1,-1), (2,2), (2,-2)

v1_deep: seed 19
v2: seed 19 (largest gap on these conditions)
v3: seed 0
v4_deep flinear: seed 0

Output: pub_figures/v{1,2,3,4}_behavior_up_mag1.pdf
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJ = Path(__file__).resolve().parents[5]   # CT_MJ_2026: vendored root (was the lab Analysis root)
sys.path.insert(0, str(PROJ))

from src.rnn.analysis.model_loader import load_and_predict

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COLOR_TRAIN = (255/255, 99/255, 71/255)
COLOR_HOLDOUT = (70/255, 130/255, 180/255)
BG_COLOR = (0, 0, 0)

TRAIN_STARTS = [(-2, -3), (-1, -2), (2, 0)]
TEST_STARTS = [(-1, -1), (2, 2), (2, -2)]
DIRECTION = (0, 1)
MAG = 1


def plot_arrow(ax, start, end, vector_norm, color, width=3):
    sac_vector = end - start
    r = np.linalg.norm(sac_vector)
    if r < 1e-6:
        return
    theta = np.arctan2(sac_vector[1], sac_vector[0])
    head_width = 0.4 / 3 * width * 1.5
    head_length = 0.15 / 3 * width * 1.5
    linewidth = 1.5 / 3 * width * 1.5
    ax.arrow(start[0], start[1], sac_vector[0], sac_vector[1], alpha=1,
             head_width=head_width, head_length=head_length,
             linewidth=linewidth, overhang=0.75, color=color,
             length_includes_head=True, zorder=10)


def run(model_dir, out_path, train_starts=None, test_starts=None):
    if train_starts is None:
        train_starts = TRAIN_STARTS
    if test_starts is None:
        test_starts = TEST_STARTS
    labels, pred, grounded, internals, task = load_and_predict(model_dir)
    vector = (DIRECTION[0] * MAG, DIRECTION[1] * MAG)

    width = 3
    fig, ax = plt.subplots(1, 1, figsize=(width, width), clip_on=False)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim([-3.2, 3.2])
    ax.set_ylim([-3.2, 3.2])
    ax.set_clip_on(False)
    ax.set_xticks(np.arange(-3, 3.1, 1))
    ax.set_yticks(np.arange(-3, 3.1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    ax.set_aspect('equal')
    ax.grid(linewidth=width * 0.3, alpha=0.7, zorder=-200)
    ax.add_patch(Circle((0, 0), 0.3, edgecolor=(1, 0, 0), facecolor=(1, 0, 0),
                         linewidth=2, zorder=-100))
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, cond in enumerate(labels):
        if cond[1] != vector:
            continue
        start = np.array(cond[0], dtype=float)
        if tuple(cond[0]) in train_starts and grounded[i] == 1:
            plot_arrow(ax, start, pred[i], MAG, COLOR_TRAIN, width)
        elif tuple(cond[0]) in test_starts and grounded[i] == 0:
            plot_arrow(ax, start, pred[i], MAG, COLOR_HOLDOUT, width)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {out_path}')

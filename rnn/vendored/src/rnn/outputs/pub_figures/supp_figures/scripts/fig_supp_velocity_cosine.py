"""Supplementary Fig. S4.1: velocity cosine across rank sweeps.

2 columns: model a (linear readout) vs model c (FFN readout)
Each shows the rank sweep: rank1, rank2, rank4, rank16, fullrank (= rank 32)
Models: rnn/models/model_a/ and rnn/models/model_c/
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import cool

PROJ = Path(__file__).resolve().parents[6]   # CT_MJ_2026: vendored root (was the lab Analysis root)
sys.path.insert(0, str(PROJ))

from src.rnn.outputs.dmfc.plotting.plot_velocity_cosine import (
    SEEDS, _aggregate_seeds, compute_velocity_cosine,
)

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

MODELS = PROJ.parent / 'models'   # rnn/models

RANKS = ['rank1', 'rank2', 'rank4', 'rank16', 'fullrank']
RANK_LABELS = {'rank1': 'r=1', 'rank2': 'r=2', 'rank4': 'r=4', 'rank16': 'r=16', 'fullrank': 'r=32'}
RANK_COLORS = {rk: cool(i / (len(RANKS) - 1)) for i, rk in enumerate(RANKS)}


def plot_2panel(out_path):
    fontsize = 10

    conditions = [
        ('Model a: linear readout', MODELS / 'model_a', RANKS),
        ('Model c: FFN readout', MODELS / 'model_c', RANKS),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)

    for ax_idx, (title, base_dir, ranks) in enumerate(conditions):
        ax = axes[ax_idx]
        ax.set_title(title, fontsize=fontsize, fontfamily='Helvetica')
        print(f'\n=== {title} ===')

        data = _aggregate_seeds(base_dir, ranks, SEEDS, compute_velocity_cosine)
        for rk in ranks:
            if rk not in data:
                continue
            means, sems, t, n = data[rk]
            ax.plot(t, means[0], color=RANK_COLORS[rk], lw=1.5,
                    label=RANK_LABELS[rk])
            ax.fill_between(t, means[0] - sems[0], means[0] + sems[0],
                            color=RANK_COLORS[rk], alpha=0.2)
            print(f'  {rk}: {n} seeds, final={means[0][-1]:.3f}')

        ax.axhline(1.0, color='gray', ls='--', lw=0.8)
        ax.axhline(-1/3, color='gray', ls='--', lw=0.8)
        ax.set_xlim(0, 2.0)
        ax.set_ylim(-0.6, 1.15)
        ax.set_xlabel('Time (s)', fontsize=fontsize, fontfamily='Helvetica')
        ax.set_xticks([0, 2.0])
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize - 1)
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontfamily('Helvetica')

    axes[0].set_ylabel('Cos sim.', fontsize=fontsize,
                        fontfamily='Helvetica')

    handles = [plt.Line2D([0], [0], color=RANK_COLORS[rk], lw=1.5)
               for rk in RANKS]
    labels = [RANK_LABELS[rk] for rk in RANKS]
    axes[-1].legend(handles, labels, fontsize=fontsize - 2, frameon=False,
                    loc='lower right')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_path}')

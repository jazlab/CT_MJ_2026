"""Fig. 5g: 2D sweep heatmap — independence (α) × alignment (β) → holdout error (output OUT_DIR/Figure5g.pdf).

Sweep architecture (start_mix blending):
  readout_start = α·start_mix_from_RNN + (1-α)·clean_start_2d
  combine([readout_start; disp], flatness=β)

α values (original): [0, 0.25, 0.75, 0.96, 1.0]
  plotted as 1-α: [1.0, 0.75, 0.25, 0.04, 0.0]
β values: [linear, 4, 1, 0.5, 0.125]

Models: rnn/models/sweep/alpha{α}_beta{β}/s{0,1,2}
Cache: cache/Figure5g_sweep.json (mean holdout error per cell; --recompute rebuilds it from the models)

Usage:
    python Figure5g_sweep_heatmap.py [--recompute] [OUT_DIR]
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

HERE = Path(os.path.dirname(os.path.abspath(__file__)))


# Cached sweep results: (alpha, beta) -> mean holdout error (cache/Figure5g_sweep.json; rebuilt from the sweep
# models with --recompute)
def recompute():
    """mean over the 3 seeds of the holdout normalized error |prediction - target| / |Flow| of each sweep model
    (rnn/models/sweep/alpha{alpha}_beta{beta}/s*; identical to the cache)"""
    sys.path.insert(0, str(HERE.parent / 'rnn'))
    import numpy as np
    import figures as F
    from src.rnn.analysis.model_loader import load_and_predict
    res = []
    for d in sorted((F.MODELS / 'sweep').iterdir()):
        a, b = d.name[len('alpha'):].split('_beta')
        errs = []
        for sd in sorted(d.glob('s*')):
            labels, pred, grounded, internals, task = load_and_predict(sd)
            tgt = np.array([[c[2][0], c[2][1]] for c in labels])
            mags = np.linalg.norm(np.array([[c[1][0], c[1][1]] for c in labels]), axis=1)
            e = np.linalg.norm(pred - tgt, axis=1) / np.maximum(mags, 1e-6)
            errs.append(e[grounded == 0].mean())
        res.append([float(a), b, float(np.mean(errs))])
    json.dump(res, open(HERE / 'cache' / 'Figure5g_sweep.json', 'w'), indent=0)


if '--recompute' in sys.argv:
    sys.argv.remove('--recompute')
    recompute()
results = {(a, b): v for a, b, v in json.load(open(HERE / 'cache' / 'Figure5g_sweep.json'))}

# Original α values to plot, reversed so independent is on the right
ALPHAS_ORIG = [1.0, 0.96, 0.75, 0.25, 0.0]
ALPHAS_LABEL = [1 - a for a in ALPHAS_ORIG]  # [0.0, 0.04, 0.25, 0.75, 1.0]

# β values bottom to top: nonlinear → aligned
BETAS = ['0.125', '0.5', '1.0', '4.0', 'linear']
BETA_LABELS = ['0.125', '0.5', '1', '4', 'Linear']

mat = np.zeros((len(BETAS), len(ALPHAS_ORIG)))
for i, beta in enumerate(BETAS):
    for j, alpha in enumerate(ALPHAS_ORIG):
        mat[i, j] = results.get((alpha, beta), np.nan)

fontsize = 10
fig, ax = plt.subplots(figsize=(3.2, 3.2))

im = ax.imshow(mat, cmap='magma_r', aspect='equal', origin='lower',
               norm=LogNorm(vmin=0.005, vmax=0.8),
               interpolation='bicubic')

ax.set_xticks([0, len(ALPHAS_ORIG) - 1])
ax.set_xticklabels(['0', '1'], fontsize=fontsize - 2, fontfamily='Helvetica')
ax.set_yticks(range(len(BETAS)))
ax.set_yticklabels(BETA_LABELS, fontsize=fontsize - 2, fontfamily='Helvetica')

ax.set_xlabel(r'Independence (1-$\alpha$)', fontsize=fontsize)
ax.set_ylabel(r'Alignment ($\beta$)', fontsize=fontsize)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.08)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Holdout error', fontsize=fontsize - 1, fontfamily='Helvetica')
cbar.ax.tick_params(labelsize=fontsize - 2)
for tl in cbar.ax.get_yticklabels():
    tl.set_fontfamily('Helvetica')

fig.tight_layout()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'out'
OUT.mkdir(parents=True, exist_ok=True)
save = OUT / 'Figure5g.pdf'
fig.savefig(save, bbox_inches='tight')
plt.close()
print(f'Saved: {save}')

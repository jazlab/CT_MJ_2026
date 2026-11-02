"""Shared entry points for the RNN figure panels (Figs. 4, 5g; S4.1-S4.3, S5.3).

The RNN code is vendored from the lab code base under rnn/vendored/src/rnn/ (paths and module names adapted), and
the trained checkpoints are in rnn/models/<model>/s<seed>/ (see rnn/README.md). These helpers put the vendored tree
on sys.path and call the original plotting functions with the original arguments, writing to OUT_DIR under
figure-panel names. Each model is run forward on all conditions (seconds per seed); no training.
"""
import sys, shutil, functools, contextlib
from pathlib import Path

RNN = Path(__file__).resolve().parent
VENDORED = RNN / 'vendored'
MODELS = RNN / 'models'
if str(VENDORED) not in sys.path:
    sys.path.insert(0, str(VENDORED))

REGPLOT_SEED = 0   # seed for the bootstrap confidence bands of the regression lines (unseeded in the lab code)


@contextlib.contextmanager
def _seeded_regplot(seed=REGPLOT_SEED):
    """make seaborn.regplot's bootstrap band reproducible without editing the vendored plotting code"""
    import seaborn as sns
    orig = sns.regplot
    sns.regplot = functools.partial(orig, seed=seed)
    try:
        yield
    finally:
        sns.regplot = orig


# Behavior insets (fig_behavior_illustrator.py): Flow up, magnitude 1
TRAIN_STARTS = [(-2, 2), (1, -3), (2, 0)]
TEST_STARTS = [(-1, 2), (2, -3), (-2, -2)]


def pca_pseudo(model_dir, out, **kw):
    """pseudo-population PCA, training vs holdout (plotting/plot_pca_train_test.run)"""
    from src.rnn.outputs.dmfc.plotting.plot_pca_train_test import run
    run(model_dir, out, **kw)


def pca_population(model_dir, out, inset_ms=300):
    """pseudo-population PCA (plotting/plot_pca.run)"""
    from src.rnn.outputs.dmfc.plotting.plot_pca import run
    run(model_dir, out, inset_ms=inset_ms)


def pca_single_seed(seed_dir, out, inset_ms=300):
    """single-seed PCA (plotting/plot_pca.run_single_seed)"""
    from src.rnn.outputs.dmfc.plotting.plot_pca import run_single_seed
    run_single_seed(seed_dir, out, inset_ms=inset_ms)


def pca_neuraldiff(model_dir, seed, out, joint=False):
    """single-seed PCA of the best training vs most deviating holdout condition per direction
    (pub_figures/scripts/fig_pca_neuraldiff.run_pca_neuraldiff; fig_supp_mixed_input.run_pca_neuraldiff for the
    mixed-input model)"""
    if joint:
        from src.rnn.outputs.pub_figures.supp_figures.scripts.fig_supp_mixed_input import run_pca_neuraldiff
    else:
        from src.rnn.outputs.pub_figures.scripts.fig_pca_neuraldiff import run_pca_neuraldiff
    run_pca_neuraldiff(model_dir, seed, out)


def proximity(model_dir, out, seeds):
    """distance to the nearest training condition vs behavioral error (analysis/plot_proximity.plot_proximity_multi)"""
    from src.rnn.analysis.plot_proximity import plot_proximity_multi
    out = Path(out)
    with _seeded_regplot():
        plot_proximity_multi(model_dir, str(out.parent), seeds=seeds)
    shutil.move(str(out.parent / 'proximity_zscore.pdf'), str(out))


def neural_behavior(model_dir, out, **kw):
    """neural deviation from the template vs behavioral error (pub_figures/scripts/fig_neural_vs_behavior.run)"""
    import src.rnn.outputs.pub_figures.scripts.fig_neural_vs_behavior as NVB
    out = Path(out)
    NVB.PUB = out.parent
    NVB.sstats = _PrintPearson(NVB.sstats)
    with _seeded_regplot():
        NVB.run(model_dir, out.name, **kw)


class _PrintPearson:
    """scipy.stats proxy that prints the exact Pearson r, p and n the plotting code annotates"""
    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        return getattr(self._mod, name)

    def pearsonr(self, x, y):
        res = self._mod.pearsonr(x, y)
        print(f'  neural vs behavior: r={res[0]:.4f}, p={res[1]:.3g}, n={len(x)}')
        return res


def behavior_inset(seed_dir, out):
    """example saccades, training and holdout Starts (pub_figures/scripts/fig_behavior_illustrator.run)"""
    from src.rnn.outputs.pub_figures.scripts.fig_behavior_illustrator import run
    run(seed_dir, out, train_starts=TRAIN_STARTS, test_starts=TEST_STARTS)


def out_dir(script_file):
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(script_file).resolve().parent / 'out'
    d.mkdir(parents=True, exist_ok=True)
    return d

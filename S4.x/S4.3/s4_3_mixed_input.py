"""Supplementary Fig. S4.3: Timing Module with mixed input ([Flow; Start] into the frozen RNN) + FFN readout
(rnn/models/model_mixed_input, 10 seeds): PCA (seed 4, best training vs most deviating holdout), neural deviation vs
behavioral error and proximity (10 seeds pooled). Plotting: rnn/vendored/src/rnn/outputs/pub_figures/supp_figures/
scripts/fig_supp_mixed_input.py.

Usage:
    python s4_3_mixed_input.py [OUT_DIR]
Output: OUT_DIR/S4.3_{pca,neural_behavior,proximity}.pdf (default ./out)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rnn'))
import figures as F

if __name__ == '__main__':
    out = F.out_dir(__file__)
    d = F.MODELS / 'model_mixed_input'
    F.pca_neuraldiff(d, 4, out / 'S4.3_pca.pdf', joint=True)
    F.proximity(d, out / 'S4.3_proximity.pdf', list(range(10)))
    F.neural_behavior(d, out / 'S4.3_neural_behavior.pdf', seeds=list(range(10)))

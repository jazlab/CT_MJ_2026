"""Supplementary Fig. S5.3: a, full generalization model (Timing Module + modular readout, factorized input;
rnn/models/model_modular_readout); b, the same with a nonlinear input embedding (rnn/models/model_embedded_modular,
embedding flatness 0.125 and 0.5). Panels as in the figure:
  a  PCA: pseudo-population (inset 300 ms); proximity and neural vs behavior: seed 7; inset: seed 3
  b  PCA: flatness 0.125, seed 6 (single seed); neural vs behavior: flatness 0.125 (10 seeds, seed-mean);
     proximity: flatness 0.5 (10 seeds); inset: flatness 0.125, seed 7

Usage:
    python s5_3_models.py [OUT_DIR]
Output: OUT_DIR/S5.3{a,b}_{pca,neural_behavior,proximity,inset}.pdf (default ./out)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rnn'))
import figures as F

if __name__ == '__main__':
    out = F.out_dir(__file__)
    a = F.MODELS / 'model_modular_readout'
    b = F.MODELS / 'model_embedded_modular'
    F.pca_population(a, out / 'S5.3a_pca.pdf', inset_ms=300)
    F.proximity(a, out / 'S5.3a_proximity.pdf', [7])
    F.neural_behavior(a, out / 'S5.3a_neural_behavior.pdf', seeds=[7])
    F.behavior_inset(a / 's3', out / 'S5.3a_inset.pdf')
    F.pca_single_seed(b / 'flatness0.125/s6', out / 'S5.3b_pca.pdf', inset_ms=300)
    F.neural_behavior(b / 'flatness0.125', out / 'S5.3b_neural_behavior.pdf', seeds=list(range(10)),
                      mean_across_seeds=True)
    F.proximity(b / 'flatness0.5', out / 'S5.3b_proximity.pdf', list(range(10)))
    F.behavior_inset(b / 'flatness0.125/s7', out / 'S5.3b_inset.pdf')

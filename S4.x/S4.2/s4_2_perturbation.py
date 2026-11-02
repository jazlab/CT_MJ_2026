"""Supplementary Fig. S4.2: causal perturbation of the integrator state along the common and residual axes (signed
slope of the readout's displacement length; 10 seeds). Plotting: rnn/vendored/src/rnn/analysis/plot_perturbation.py,
one panel per model:
  a  Timing Module + linear readout      rnn/models/model_b               ('Model b (linear readout)')
  b  Timing Module + FFN readout         rnn/models/model_c/fullrank      ('Model c (FFN readout)')
  c  Timing Module + modular readout     rnn/models/model_modular_readout ('Modular readout')

Usage:
    python s4_2_perturbation.py [OUT_DIR]
Output: OUT_DIR/S4.2{a,b,c}.pdf (default ./out)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rnn'))
import figures as F

if __name__ == '__main__':
    from src.rnn.analysis.plot_perturbation import plot_perturbation_bar
    out = F.out_dir(__file__)
    for panel, d, label in [('a', 'model_b', 'Model b (linear readout)'),
                            ('b', 'model_c/fullrank', 'Model c (FFN readout)'),
                            ('c', 'model_modular_readout', 'Modular readout')]:
        plot_perturbation_bar([str(F.MODELS / d)], [label], list(range(10)), str(out / f'S4.2{panel}.pdf'))

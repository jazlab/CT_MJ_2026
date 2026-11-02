"""Supplementary Fig. S4.1: cosine similarity of the integrator velocity across Flow directions over time, rank
sweep (r = 1, 2, 4, 16, 32), a: model a, trainable RNN + linear readout (rnn/models/model_a), b: model c, pretrained
+ frozen Timing Module + FFN readout (rnn/models/model_c). Plotting: rnn/vendored/src/rnn/outputs/pub_figures/
supp_figures/scripts/fig_supp_velocity_cosine.py.

Usage:
    python s4_1_velocity_cosine.py [OUT_DIR]
Output: OUT_DIR/S4.1.pdf (default ./out)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rnn'))
import figures as F

if __name__ == '__main__':
    from src.rnn.outputs.pub_figures.supp_figures.scripts.fig_supp_velocity_cosine import plot_2panel
    plot_2panel(F.out_dir(__file__) / 'S4.1.pdf')

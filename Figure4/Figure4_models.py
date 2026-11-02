"""Fig. 4: four RNN architectures (rows a-d) x three features (PCA of integrator states; neural deviation vs
behavioral error; proximity effect), plus the behavior insets (example saccades, Flow up, magnitude 1).

Runs the trained checkpoints (rnn/models/) through the plotting code in rnn/vendored/ (via rnn/figures.py). Rows:
  a  Flow -> full-rank RNN -> linear([h; Start])                          rnn/models/model_a/fullrank (20 seeds)
  b  Flow -> pretrained + frozen Timing Module -> linear                   rnn/models/model_b (10 seeds)
  c  Flow -> Timing Module -> FFN readout (6 x 128)                        rnn/models/model_c/fullrank (10 seeds)
  d  nonlinear input embedding(Flow, Start) -> Timing Module -> FFN        rnn/models/model_d (10 seeds)
Behavior insets (one seed per model, as in the figure): a s16, b s0, c s1, d s3.

Usage:
    python Figure4_models.py [OUT_DIR]
Output: OUT_DIR/Figure4{a,b,c,d}_{pca,neural_behavior,proximity,inset}.pdf (default ./out)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rnn'))
import figures as F

if __name__ == '__main__':
    out = F.out_dir(__file__)
    M = F.MODELS
    rows = [('a', M / 'model_a/fullrank', list(range(20)), dict(seeds=[0]), M / 'model_a/fullrank/s16', {}),
            ('b', M / 'model_b', list(range(10)), dict(seeds=list(range(10)), mean_across_seeds=True),
             M / 'model_b/s0', {}),
            ('c', M / 'model_c/fullrank', list(range(10)), dict(seeds=list(range(10)), mean_across_seeds=True),
             M / 'model_c/fullrank/s1', dict(flip_pc1=True)),
            ('d', M / 'model_d', list(range(10)), dict(seeds=list(range(10)), mean_across_seeds=True),
             M / 'model_d/s3', None)]
    for row, mdir, seeds, nvb_kw, inset_dir, pca_kw in rows:
        print(f'=== Fig. 4{row}: {mdir.relative_to(M)}')
        if pca_kw is None:
            F.pca_neuraldiff(mdir, 3, out / f'Figure4{row}_pca.pdf')          # PCA seed 3
        else:
            F.pca_pseudo(mdir, out / f'Figure4{row}_pca.pdf', **pca_kw)
        F.neural_behavior(mdir, out / f'Figure4{row}_neural_behavior.pdf', **nvb_kw)
        F.proximity(mdir, out / f'Figure4{row}_proximity.pdf', seeds)
        F.behavior_inset(inset_dir, out / f'Figure4{row}_inset.pdf')

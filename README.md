# Tang & Jazayeri 2026: code for the RNN model figures

Reproduces Figs. 4 and 5g and Supplementary Figs. S4.1-S4.3 and S5.3 from the trained RNN models in `rnn/models/`.
The monkey figures (Figs. 1-3, S1-S3) and the human psychophysics panels (Figs. 5c-e, S5.1-S5.2) will be released
after peer review.

```
pip install -r requirements.txt
cd Figure4 && python Figure4_models.py
cd ../Figure5 && python Figure5g_sweep_heatmap.py
cd ../S4.x/S4.1 && python s4_1_velocity_cosine.py
cd ../S4.2 && python s4_2_perturbation.py
cd ../S4.3 && python s4_3_mixed_input.py
cd ../../S5.x/S5.3 && python s5_3_models.py
```
Each script writes its panels to `out/` in its own folder; the `out/` folders contain the panels as generated.

| Folder (`rnn/models/`) | Model | Figures |
|---|---|---|
| `model_a` | Flow -> trainable RNN -> linear readout | Fig. 4a, S4.1a |
| `model_b` | Flow -> Timing Module -> linear readout | Fig. 4b, S4.2a |
| `model_c` | Flow -> Timing Module -> FFN readout | Fig. 4c, S4.1b, S4.2b |
| `model_d` | input embedding -> Timing Module -> FFN readout | Fig. 4d |
| `model_mixed_input` | [Flow; Start] -> Timing Module -> FFN readout | S4.3 |
| `model_modular_readout` | Timing Module + modular readout | S4.2c, S5.3a |
| `model_embedded_modular` | input embedding + Timing Module + modular readout | S5.3b |
| `sweep` | input entanglement x output alignment sweep | Fig. 5g |

Code: `rnn/vendored/` (models, task, plotting), `rnn/training/` (training scripts). Requires PyTorch (CPU).

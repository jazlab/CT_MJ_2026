"""Load any DMFC model version and run predictions.

Model types (config.json 'version'): rnn_readout (models a-c), modular_readout, embed_rnn_readout (model d),
joint_input_rnn_readout (mixed input) and sweep (Fig. 5g).

Usage:
    from src.rnn.analysis.model_loader import load_and_predict
    labels, pred, grounded, internals = load_and_predict(model_dir)
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.rnn.models.dmfc_model_v2 import Rank1Integrator
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2


def load_config(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / 'config.json') as f:
        raw = f.read().replace('Infinity', '1e308')
        cfg = json.loads(raw)
    s1_path = model_dir / 'config_stage1.json'
    if s1_path.exists():
        with open(s1_path) as f:
            s1 = json.load(f)
    else:
        s1 = {'dt': cfg['dt'], 'tau': cfg['tau'], 'hidden_size': cfg['hidden_size']}
    return cfg, s1


def _build_rnn_readout(cfg, s1):
    from src.rnn.training.train_rnn_readout import V1Model, FullRankRNN, ReadoutMLP
    hidden_size = cfg['hidden_size']
    alpha = cfg['dt'] / cfg['tau']
    n_retino = cfg.get('n_retino', 49)
    mlp_layers = cfg.get('mlp_layers', 3)
    mlp_flatness = cfg.get('mlp_flatness', 1.0)
    rank = cfg.get('rank', None)
    if rank is not None and rank != 'full' and rank < hidden_size:
        from src.rnn.training.train_rank_n import RankNIntegrator
        rnn = RankNIntegrator(input_size=4, hidden_size=hidden_size, rank=rank,
                               alpha=alpha, flatness=1.0)
    else:
        rnn = FullRankRNN(input_size=4, hidden_size=hidden_size, alpha=alpha)
    if cfg.get('linear_readout', False):
        readout_mlp = nn.Linear(hidden_size + n_retino, 2)
    else:
        readout_mlp = ReadoutMLP(hidden_size + n_retino, cfg['mlp_hidden'],
                                  n_layers=mlp_layers, flatness=mlp_flatness)
    return V1Model(rnn, readout_mlp)


def _build_sweep(cfg, s1):
    """Input-entanglement x output-alignment sweep (Fig. 5g): embed([flow; α·start]) → frozen RNN → DisplacementMLP → disp,
    start → w_fef → start_2D, [start_2D; disp] → CombineMLP(β) → target."""
    from src.rnn.training.train_rnn_readout import FullRankRNN
    from src.rnn.training.train_modular_readout import EmbeddingMLP53, DisplacementMLP

    hidden_size = cfg['hidden_size']
    alpha_rnn = cfg['dt'] / cfg['tau']
    n_retino = cfg.get('n_retino', 49)
    alpha_input = cfg.get('alpha', 0.0)
    beta_str = str(cfg.get('beta', 'linear'))

    embed = EmbeddingMLP53(hidden_size=32, flatness=1.0, n_layers=2)
    rnn = FullRankRNN(input_size=4, hidden_size=hidden_size, alpha=alpha_rnn)
    ramp_readout = nn.Linear(hidden_size, 1)
    disp_mlp = DisplacementMLP(hidden_size, 32, output_dim=2)
    w_fef = nn.Linear(n_retino, 2)
    start_mix_head = nn.Linear(hidden_size, 2)   # blends the RNN-extracted Start with the clean Start

    if beta_str == 'linear':
        combine = nn.Linear(4, 2)
    else:
        beta_val = float(beta_str)
        class ScaledTanh(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c
            def forward(self, x):
                return self.c * torch.tanh(x / self.c)
        combine = nn.Sequential(
            nn.Linear(4, 32), ScaledTanh(beta_val),
            nn.Linear(32, 32), ScaledTanh(beta_val),
            nn.Linear(32, 2)
        )

    class SweepModel(nn.Module):
        def __init__(self, embed, rnn, ramp_readout, disp_mlp, w_fef,
                     combine, alpha_in, start_mix_head):
            super().__init__()
            self.embed = embed
            self.rnn = rnn
            self.ramp_readout = ramp_readout
            self.disp_mlp = disp_mlp
            self.w_fef = w_fef
            self.combine = combine
            self.alpha_in = alpha_in
            self.start_mix_head = start_mix_head

        def forward(self, dir_onehot, start_retino, flow_off_bins):
            B, T, _ = dir_onehot.shape
            start_to_embed = self.alpha_in * start_retino
            combined_in = torch.cat([dir_onehot, start_to_embed], dim=2)
            embedded = self.embed(combined_in.reshape(B * T, 53)).reshape(B, T, 53)
            emb_flow = embedded[:, :, :4]
            h_int = self.rnn(emb_flow)
            h_snap = torch.stack([h_int[i, flow_off_bins[i]] for i in range(B)])
            disp = self.disp_mlp(h_snap)
            # start_2d from w_fef (full start, no alpha scaling)
            s_snap = torch.stack([start_retino[i, flow_off_bins[i]] for i in range(B)])
            start_2d = self.w_fef(s_snap)
            # blend the RNN-extracted Start with the clean Start at the 2D level
            start_mix = self.start_mix_head(h_snap)
            readout_start = self.alpha_in * start_mix + (1.0 - self.alpha_in) * start_2d
            target = self.combine(torch.cat([readout_start, disp], dim=1))
            return target, {'h_rnn': h_int, 'h_snapshot': h_snap, 'disp': disp}

    return SweepModel(embed, rnn, ramp_readout, disp_mlp, w_fef, combine,
                      alpha_input, start_mix_head)


def _build_embed_rnn_readout(cfg, s1):
    """Model d: EmbeddingMLP53(f=1) + frozen full-rank RNN + ReadoutMLP(6x128)."""
    from src.rnn.training.train_rnn_readout import FullRankRNN, ReadoutMLP
    from src.rnn.training.train_modular_readout import EmbeddingMLP53

    hidden_size = cfg['hidden_size']
    alpha = cfg['dt'] / cfg['tau']
    n_retino = cfg.get('n_retino', 49)
    embed_h = cfg.get('embed_hidden', 32)
    embed_l = cfg.get('embed_layers', 2)
    embed_f = cfg.get('embed_flatness', 1.0)
    mlp_hidden = cfg.get('mlp_hidden', 128)
    mlp_layers = cfg.get('mlp_layers', 6)
    mlp_flatness = cfg.get('mlp_flatness', 0.125)

    embed = EmbeddingMLP53(hidden_size=embed_h, flatness=embed_f, n_layers=embed_l)
    rnn = FullRankRNN(input_size=4, hidden_size=hidden_size, alpha=alpha)
    ramp_readout = nn.Linear(hidden_size, 1)
    readout_mlp = ReadoutMLP(hidden_size + n_retino, mlp_hidden,
                              n_layers=mlp_layers, flatness=mlp_flatness)

    class EmbedRNNReadoutModel(nn.Module):
        def __init__(self, embed, rnn, ramp_readout, readout_mlp):
            super().__init__()
            self.embed = embed
            self.rnn = rnn
            self.ramp_readout = ramp_readout
            self.readout_mlp = readout_mlp

        def forward(self, dir_onehot, start_retino, flow_off_bins):
            B, T, _ = dir_onehot.shape
            combined = torch.cat([dir_onehot, start_retino], dim=2)
            embedded = self.embed(combined.reshape(B * T, 53)).reshape(B, T, 53)
            emb_flow = embedded[:, :, :4]
            emb_start = embedded[:, :, 4:]
            h_int = self.rnn(emb_flow)
            h_snap = torch.stack([h_int[i, flow_off_bins[i]] for i in range(B)])
            s_snap = torch.stack([emb_start[i, flow_off_bins[i]] for i in range(B)])
            target = self.readout_mlp(torch.cat([h_snap, s_snap], dim=1))
            return target, {'h_rnn': h_int, 'h_snapshot': h_snap}

    return EmbedRNNReadoutModel(embed, rnn, ramp_readout, readout_mlp)


def _build_joint_input_rnn_readout(cfg, s1):
    """Mixed-input model (S4.3): [flow;start]=53D → frozen full-rank RNN(32) → h → ReadoutMLP."""
    from src.rnn.training.train_rnn_readout import FullRankRNN, ReadoutMLP

    hidden_size = cfg['hidden_size']
    alpha = cfg['dt'] / cfg['tau']
    mlp_hidden = cfg.get('mlp_hidden', 128)
    mlp_layers = cfg.get('mlp_layers', 6)
    mlp_flatness = cfg.get('mlp_flatness', 0.125)
    n_retino = cfg.get('n_retino', 49)

    rnn = FullRankRNN(input_size=4 + n_retino, hidden_size=hidden_size, alpha=alpha)
    ramp_readout = nn.Linear(hidden_size, 1)
    readout_mlp = ReadoutMLP(hidden_size, mlp_hidden,
                              n_layers=mlp_layers, flatness=mlp_flatness)

    class JointInputRNNReadoutModel(nn.Module):
        def __init__(self, rnn, ramp_readout, readout_mlp):
            super().__init__()
            self.rnn = rnn
            self.ramp_readout = ramp_readout
            self.readout_mlp = readout_mlp

        def forward(self, dir_onehot, start_retino, flow_off_bins):
            B, T, _ = dir_onehot.shape
            joint = torch.cat([dir_onehot, start_retino], dim=2)
            h_int = self.rnn(joint)
            h_snap = torch.stack([h_int[i, flow_off_bins[i]] for i in range(B)])
            target = self.readout_mlp(h_snap)
            return target, {'h_rnn': h_int, 'h_snapshot': h_snap}

    return JointInputRNNReadoutModel(rnn, ramp_readout, readout_mlp)


def _build_modular_readout(cfg, s1):
    from src.rnn.training.train_modular_readout import ModularReadoutModel, DisplacementMLP

    hidden_size = s1['hidden_size']
    alpha = s1['dt'] / s1['tau']
    linear = s1.get('linear', False)
    n_retino = cfg.get('n_retino', 49)
    raw_input = cfg.get('raw_input', False)
    embed_flatness = cfg.get('embed_flatness', None)
    if embed_flatness is not None and embed_flatness > 1e300:
        embed_flatness = float('inf')

    joint_proj_flag = cfg.get('joint_proj', False)
    joint_proj_layer = None
    embed_mlp_layer = None

    if embed_flatness is not None:
        from src.rnn.training.train_modular_readout import EmbeddingMLP53
        w_mt = None
        w_fef = None
        embed_h = cfg.get('embed_hidden', 32)
        embed_l = cfg.get('embed_layers', 2)
        embed_mlp_layer = EmbeddingMLP53(hidden_size=embed_h, flatness=embed_flatness,
                                          n_layers=embed_l)
        flow_dim = 4
        start_dim = n_retino
    elif joint_proj_flag:
        w_mt = None
        w_fef = None
        joint_proj_layer = nn.Linear(4 + n_retino, 4, bias=False)
        flow_dim = 2
        start_dim = 2
    elif raw_input:
        w_mt = None
        w_fef = None
        flow_dim = 4
        start_dim = n_retino
    else:
        w_mt = nn.Linear(4, 2, bias=False)
        w_fef = nn.Linear(n_retino, 2, bias=False)
        flow_dim = 2
        start_dim = 2

    rnn_type = cfg.get('rnn_type', None)
    rank = cfg.get('rank', None)
    if rnn_type == 'full_rank':
        from src.rnn.training.train_rnn_readout import FullRankRNN
        rnn_linear = cfg.get('rnn_linear', False)
        integrator = FullRankRNN(input_size=flow_dim, hidden_size=hidden_size,
                                  alpha=alpha, linear=rnn_linear)
    elif rnn_type == 'rank' and rank is not None and isinstance(rank, int) and rank < hidden_size:
        from src.rnn.training.train_rank_n import RankNIntegrator
        rnn_flat = cfg.get('rnn_flatness', 1.0)
        if rnn_flat == 'linear':
            rnn_flat = None
        integrator = RankNIntegrator(input_size=flow_dim, hidden_size=hidden_size,
                                     rank=rank, alpha=alpha, flatness=rnn_flat)
    else:
        integrator = Rank1Integrator(input_size=flow_dim, hidden_size=hidden_size,
                                  alpha=alpha, linear=linear)
    ramp_readout = nn.Linear(hidden_size, 1)
    disp_mlp = DisplacementMLP(hidden_size, cfg['mlp_hidden'])
    w_target = nn.Linear(start_dim + 2, 2)
    model = ModularReadoutModel(w_mt, w_fef, integrator, ramp_readout, disp_mlp, w_target,
                      joint_proj=joint_proj_layer, embed_mlp=embed_mlp_layer)
    return model


def load_model(model_dir):
    """Load model from directory. Returns (model, cfg, s1)."""
    model_dir = Path(model_dir)
    cfg, s1 = load_config(model_dir)
    version = cfg['version']
    builders = {'rnn_readout': _build_rnn_readout, 'modular_readout': _build_modular_readout,
                'embed_rnn_readout': _build_embed_rnn_readout,
                'joint_input_rnn_readout': _build_joint_input_rnn_readout, 'sweep': _build_sweep}
    if version not in builders:
        raise ValueError(f'Unknown model type: {version}')
    model = builders[version](cfg, s1)

    state = torch.load(model_dir / 'model.pt', weights_only=True)
    # Flatten nested state dicts (e.g. {'embed': {...}, 'rnn': {...}} → 'embed.k': v)
    if state and isinstance(next(iter(state.values())), dict):
        flat = {}
        for prefix, sub_dict in state.items():
            for k, v in sub_dict.items():
                flat[f'{prefix}.{k}'] = v
        state = flat
    # Remap old EmbeddingMLP53 keys (fc1/fc2/fc3 → layers.0/1/2)
    remapped = {}
    for k, v in state.items():
        nk = k
        for old, new in [('embed_mlp.fc1.', 'embed_mlp.layers.0.'),
                          ('embed_mlp.fc2.', 'embed_mlp.layers.1.'),
                          ('embed_mlp.fc3.', 'embed_mlp.layers.2.')]:
            if k.startswith(old):
                nk = new + k[len(old):]
                break
        remapped[nk] = v
    # Remap rnn.* → integrator.* if needed
    model_keys = set(model.state_dict().keys())
    remapped2 = {}
    for k, v in remapped.items():
        if k.startswith('rnn.') and k not in model_keys:
            alt = 'integrator.' + k[4:]
            if alt in model_keys:
                remapped2[alt] = v
                continue
        # Remap bare readout_mlp.weight → readout_mlp.net.weight (nn.Linear vs ReadoutMLP)
        if k.startswith('readout_mlp.') and k not in model_keys:
            alt = k.replace('readout_mlp.', 'readout_mlp.net.', 1)
            if alt in model_keys:
                remapped2[alt] = v
                continue
        # Remap embed.layers.* → embed.layers.* (already matched) or embed_mlp.* → embed.*
        if k.startswith('embed.') and k not in model_keys:
            alt = k.replace('embed.', 'embed_mlp.', 1)
            if alt in model_keys:
                remapped2[alt] = v
                continue
        remapped2[k] = v
    # Drop keys model doesn't have (e.g. ramp_readout)
    remapped2 = {k: v for k, v in remapped2.items() if k in model_keys}
    model.load_state_dict(remapped2)
    model.eval()
    return model, cfg, s1


def _prepare_inputs(task, conditions):
    """Build model inputs (dir_onehot, start_retino) from task conditions."""
    from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode

    # Read sigma/grid from the caller's cfg — passed via task isn't ideal,
    # so we accept them as attributes on the task object or use defaults
    sigma = getattr(task, '_retino_sigma', 1.0)
    grid_size = getattr(task, '_retino_grid_size', 7)
    centers = make_retinotopic_centers(grid_size)

    ball_speed = task.BALL_SPEED
    dt = task.dt
    dir_list, retino_list, flow_off_list = [], [], []

    for cond in conditions:
        inp, tgt, ramp = task._make_trial(cond)
        start_xy = np.array(cond[0], dtype=np.float32).reshape(1, 2)
        vec = np.array(cond[1])
        flow_off = int(np.round((task.FLOW_ONSET_TIME + np.linalg.norm(vec) / ball_speed) / dt))

        retino = retinotopic_encode(start_xy, centers, sigma)[0]
        retino_tiled = np.tile(retino, (inp.shape[0], 1))

        dir_list.append(inp[:, 2:6])
        retino_list.append(retino_tiled)
        flow_off_list.append(flow_off)

    dir_t = torch.tensor(np.stack(dir_list))
    ret_t = torch.tensor(np.stack(retino_list))
    foff_t = torch.tensor(flow_off_list)
    return dir_t, ret_t, foff_t


def load_and_predict(model_dir):
    """Load model, run all conditions, return results.

    Returns:
        labels: list of (start, vec, target) tuples
        pred: (N, 2) array of predicted targets
        grounded: (N,) array of 0/1
        internals: dict with 'h_integrator' (N, T, H) at minimum
    """
    model_dir = Path(model_dir)
    model, cfg, s1 = load_model(model_dir)
    version = cfg['version']
    dt = s1['dt']; tau = s1['tau']

    training_set = cfg.get('training_set', 'circular')
    task = DMFCTaskV2(delay=0.2, dt=dt, vector_lengths=[1, 2, 3, 4, 5, 6],
                      training_set=training_set, scale=1, tau=tau)

    labels = []
    grounded_list = []
    for cond in task.conditions_joint:
        labels.append(cond)
        grounded_list.append(1 if cond in task.conditions_grounded else 0)

    ball_speed = task.BALL_SPEED

    with torch.no_grad():
        task._retino_sigma = cfg.get('sigma', 1.0)
        task._retino_grid_size = cfg.get('grid_size', 7)
        dir_t, ret_t, foff_t = _prepare_inputs(task, task.conditions_joint)
        if version == 'modular_readout':
            pred, _, _, internals = model(dir_t, ret_t, foff_t)
        else:
            pred, internals = model(dir_t, ret_t, foff_t)
            internals['h_integrator'] = internals['h_rnn']

    pred = pred.numpy()
    grounded = np.array(grounded_list)

    return labels, pred, grounded, internals, task

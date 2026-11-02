"""Modular readout: retinotopic start (FEF) + phasic flow (MT) → two-stream architecture.

Inputs:
  MT:  4 one-hot phasic direction channels
  FEF: 49-D retinotopic population code (7x7 Gaussian bumps, sigma=1)

Architecture:
  W_mt  (4→2 linear)  → flow_2D  → rank-1 integrator → disp MLP → disp_2D
  W_fef (49→2 linear) → start_2D → passthrough
  concat [start_2D, disp_2D] → W_target (4→2 linear) → target

Stage 1: rank-1 integrator pre-trained (u, v, bias, ramp readout frozen)
Stage 2: train W_mt, W_fef, W_ih, disp MLP, W_target jointly

Run:
  python src/rnn/training/train_modular_readout.py --out_dir OUT_DIR/s0
"""
import sys, os, time, json, argparse
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np

from src.rnn.models.dmfc_model_v2 import Rank1Integrator
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2


LOG_PATH = None

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def make_retinotopic_centers(grid_size=7, x_range=(-3, 3), y_range=(-3, 3)):
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    cx, cy = np.meshgrid(xs, ys)
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)  # (N_retino, 2)
    return centers.astype(np.float32)


def retinotopic_encode(start_xy, centers, sigma=1.0):
    """Encode 2D positions as Gaussian population activity.

    Args:
        start_xy: (B, 2) array of start positions
        centers: (N_retino, 2) array of RF centers
        sigma: RF width

    Returns:
        (B, N_retino) activation array
    """
    diff = start_xy[:, None, :] - centers[None, :, :]  # (B, N, 2)
    dist_sq = (diff ** 2).sum(axis=2)  # (B, N)
    return np.exp(-dist_sq / (2 * sigma ** 2))


class EmbeddingMLP53(nn.Module):
    """Nonlinear embedding: 53→H→...→H→53. Flatness c: c*tanh(x/c)."""
    def __init__(self, hidden_size=32, flatness=1.0, n_layers=2):
        super().__init__()
        self.flatness = flatness
        layers = [nn.Linear(53, hidden_size)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, 53))
        self.layers = nn.ModuleList(layers)

    def _act(self, x):
        c = self.flatness
        if c == float('inf'):
            return x
        return c * torch.tanh(x / c)

    def forward(self, x):
        for layer in self.layers:
            x = self._act(layer(x))
        return x


class DisplacementMLP(nn.Module):
    """MLP: 32→32→32→2. Computes direction × time from integrator state."""
    def __init__(self, input_dim, hidden_size, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModularReadoutModel(nn.Module):
    """Two-stream: W_mt→integrator→disp_MLP, W_fef→passthrough, concat→W_target."""

    def __init__(self, w_mt, w_fef, integrator, ramp_readout, disp_mlp, w_target,
                 joint_proj=None, embed_mlp=None):
        super().__init__()
        self.w_mt = w_mt
        self.w_fef = w_fef
        self.joint_proj = joint_proj
        self.embed_mlp = embed_mlp
        self.integrator = integrator
        self.ramp_readout = ramp_readout
        self.disp_mlp = disp_mlp
        self.w_target = w_target

    def forward(self, dir_onehot, start_retino, flow_off_bins):
        """
        Args:
            dir_onehot: (B, T, 4) phasic direction input
            start_retino: (B, T, 49) tonic retinotopic start
            flow_off_bins: (B,) snapshot timestep indices
        """
        B, T, _ = dir_onehot.shape

        if self.embed_mlp is not None:
            combined_in = torch.cat([dir_onehot, start_retino], dim=2)  # (B, T, 53)
            B2, T2, D = combined_in.shape
            embedded = self.embed_mlp(combined_in.reshape(B2*T2, D)).reshape(B2, T2, D)
            flow = embedded[:, :, :4]   # (B, T, 4)
            start = embedded[:, :, 4:]  # (B, T, 49)
        elif self.joint_proj is not None:
            combined_in = torch.cat([dir_onehot, start_retino], dim=2)
            proj = self.joint_proj(combined_in)
            flow = proj[:, :, :2]
            start = proj[:, :, 2:]
        else:
            flow = self.w_mt(dir_onehot) if self.w_mt is not None else dir_onehot
            start = self.w_fef(start_retino) if self.w_fef is not None else start_retino

        h_int = self.integrator(flow)
        ramp = self.ramp_readout(h_int)

        h_snapshot = torch.stack([h_int[i, flow_off_bins[i]] for i in range(B)])
        start_snapshot = torch.stack([start[i, flow_off_bins[i]] for i in range(B)])

        disp = self.disp_mlp(h_snapshot)

        combined = torch.cat([start_snapshot, disp], dim=1)
        target = self.w_target(combined)

        return target, disp, ramp, {
            'h_integrator': h_int, 'h_snapshot': h_snapshot,
            'start_2d': start_snapshot, 'disp': disp,
        }


def compute_ramp_loss(ramp_pred, ramp_tgt):
    valid = ~torch.isnan(ramp_tgt)
    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return ((ramp_pred[valid] - ramp_tgt[valid]) ** 2).mean()


def train(out_dir, delay=0.2, seed=0, lr=1e-3, ramp_lambda=15.0,
          disp_lambda=1.0, max_epochs=40000, sigma=1.0, raw_input=False,
          joint_proj=False, embed_flatness=None, input_noise=0.0,
          training_set='circular', embed_hidden=32, embed_layers=2):
    global LOG_PATH
    os.makedirs(out_dir, exist_ok=True)
    LOG_PATH = f'{out_dir}/train_log.txt'
    open(LOG_PATH, 'w').close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(f'{out_dir}/config_stage1.json') as f:
        s1_cfg = json.load(f)
    s1_ckpt = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)

    hidden_size = s1_cfg['hidden_size']
    dt = s1_cfg['dt']; tau = s1_cfg['tau']
    alpha = dt / tau
    linear = s1_cfg.get('linear', False)
    mlp_hidden = 32
    grid_size = 7
    n_retino = grid_size * grid_size  # 49

    centers = make_retinotopic_centers(grid_size)

    joint_proj_layer = None
    embed_mlp_layer = None
    if embed_flatness is not None:
        w_mt = None
        w_fef = None
        embed_mlp_layer = EmbeddingMLP53(hidden_size=embed_hidden, flatness=embed_flatness,
                                         n_layers=embed_layers)
        flow_dim = 4
        start_dim = n_retino
    elif joint_proj:
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

    integrator = Rank1Integrator(input_size=flow_dim, hidden_size=hidden_size,
                                  alpha=alpha, linear=linear)
    integrator.u.data = s1_ckpt['integrator']['u']
    integrator.v.data = s1_ckpt['integrator']['v']
    integrator.bias.data = s1_ckpt['integrator']['bias']
    integrator.W_ih.bias.data = s1_ckpt['integrator']['W_ih.bias']

    ramp_readout = nn.Linear(hidden_size, 1)
    ramp_readout.weight.data = s1_ckpt['readout']['weight']
    ramp_readout.bias.data = s1_ckpt['readout']['bias']

    disp_mlp = DisplacementMLP(hidden_size, mlp_hidden)
    w_target = nn.Linear(start_dim + 2, 2)

    model = ModularReadoutModel(w_mt, w_fef, integrator, ramp_readout, disp_mlp, w_target,
                      joint_proj=joint_proj_layer, embed_mlp=embed_mlp_layer)

    integrator.u.requires_grad = False
    integrator.v.requires_grad = False
    integrator.bias.requires_grad = False
    for p in ramp_readout.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    config = {
        'version': 'modular_readout', 'dt': dt, 'tau': tau,
        'hidden_size': hidden_size, 'linear': linear,
        'mlp_hidden': mlp_hidden, 'grid_size': grid_size,
        'n_retino': n_retino, 'sigma': sigma,
        'raw_input': raw_input, 'joint_proj': joint_proj,
        'embed_flatness': embed_flatness, 'embed_hidden': embed_hidden,
        'embed_layers': embed_layers, 'input_noise': input_noise,
        'training_set': training_set,
        'flow_dim': flow_dim, 'start_dim': start_dim,
        'ramp_lambda': ramp_lambda, 'disp_lambda': disp_lambda,
        'training': {'lr': lr},
    }
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    if embed_flatness is not None:
        log(f'modular readout: embed_MLP(53→32→32→53, flatness={embed_flatness}) → [flow_4D→rank-1→disp MLP, start_49D] → W_target')
    elif joint_proj:
        log(f'modular readout: joint_proj(53→4) → [flow_2D→rank-1→disp MLP, start_2D] → W_target')
    elif raw_input:
        log(f'modular readout: raw 4D→rank-1 → disp MLP, raw 49D→W_target({n_retino+2}→2)')
    else:
        log(f'modular readout: W_mt(4→2) + W_fef(49→2) → rank-1 → disp MLP → W_target')
    log(f'  sigma={sigma}, grid={grid_size}x{grid_size}={n_retino}')
    log(f'  trainable: {n_trainable}, frozen: {n_frozen}')
    log(f'  ramp_lambda={ramp_lambda}, disp_lambda={disp_lambda}, lr={lr}')

    task = DMFCTaskV2(delay=delay, dt=dt, vector_lengths=[1, 2, 3, 4, 5, 6],
                      training_set=training_set, scale=1, tau=tau)
    task.summary()
    ds_train, ds_val = task.get_datasets()
    ball_speed = task.BALL_SPEED

    def build_data(ds):
        data = []
        for i in range(len(ds)):
            inp_raw, tgt_raw, ramp, grounded = ds[i]
            cond = ds.condition_labels[i]
            start_xy = np.array(cond[0], dtype=np.float32).reshape(1, 2)
            vec = np.array(cond[1])
            flow_dur = np.linalg.norm(vec) / ball_speed
            flow_off = int(np.round((task.FLOW_ONSET_TIME + flow_dur) / dt))
            target = np.array(cond[2], dtype=np.float32)
            disp_target = np.array(vec, dtype=np.float32)

            retino = retinotopic_encode(start_xy, centers, sigma)[0]  # (49,)

            dir_channels = inp_raw[:, 2:6]  # (T, 4)
            retino_tiled = np.tile(retino, (dir_channels.shape[0], 1))  # (T, 49)

            data.append((dir_channels, retino_tiled, ramp, target,
                          disp_target, flow_off, grounded))
        return data

    val_data = build_data(ds_val)
    all_data = build_data(ds_train)

    def stack_tensors(data_list):
        dir_t = torch.tensor(np.stack([d[0] for d in data_list]))
        ret_t = torch.tensor(np.stack([d[1] for d in data_list]))
        ramp_t = torch.tensor(np.stack([d[2] for d in data_list]))
        tgt_t = torch.tensor(np.stack([d[3] for d in data_list]))
        disp_t = torch.tensor(np.stack([d[4] for d in data_list]))
        foff_t = torch.tensor([d[5] for d in data_list])
        gnd_t = torch.tensor([d[6] for d in data_list])
        return dir_t, ret_t, ramp_t, tgt_t, disp_t, foff_t, gnd_t

    train_dir, train_ret, train_ramp, train_tgt, train_disp, train_foff, _ = \
        stack_tensors(val_data)
    all_dir, all_ret, all_ramp, all_tgt, all_disp, all_foff, all_gnd = \
        stack_tensors(all_data)

    log(f'  train (grounded): {len(val_data)}, all: {len(all_data)}')

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5)

    best_behav = float('inf')
    patience_count = 0
    ramp_warmup = 2000
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        if input_noise > 0:
            noisy_dir = train_dir * (1 + input_noise * torch.randn_like(train_dir))
            noisy_ret = train_ret * (1 + input_noise * torch.randn_like(train_ret))
        else:
            noisy_dir, noisy_ret = train_dir, train_ret
        target_pred, disp_pred, ramp_pred, _ = model(noisy_dir, noisy_ret, train_foff)

        loss_behav = ((target_pred - train_tgt) ** 2).mean()
        loss_disp = ((disp_pred - train_disp) ** 2).mean()
        loss_ramp = compute_ramp_loss(ramp_pred, train_ramp)
        rl = ramp_lambda if epoch < ramp_warmup else ramp_lambda * 0.1
        loss = loss_behav + disp_lambda * loss_disp + rl * loss_ramp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if loss_behav.item() < best_behav:
            best_behav = loss_behav.item()
            patience_count = 0
            torch.save(model.state_dict(), f'{out_dir}/model.pt')
        else:
            patience_count += 1

        if epoch % 100 == 0:
            elapsed = time.time() - t0
            with torch.no_grad():
                tp, dp, rp, _ = model(all_dir, all_ret, all_foff)
                bg = ((tp[all_gnd == 1] - all_tgt[all_gnd == 1]) ** 2).mean().item()
                bh = ((tp[all_gnd == 0] - all_tgt[all_gnd == 0]) ** 2).mean().item()
                dg = ((dp[all_gnd == 1] - all_disp[all_gnd == 1]) ** 2).mean().item()
                dh = ((dp[all_gnd == 0] - all_disp[all_gnd == 0]) ** 2).mean().item()
                lr_ = compute_ramp_loss(rp, all_ramp).item()
            msg = (f'  epoch {epoch:5d}  loss={loss.item():.6f}  '
                   f'behav={bg:.4f}/{bh:.4f}  disp={dg:.4f}/{dh:.4f}  ramp={lr_:.4f}  '
                   f'lr={optimizer.param_groups[0]["lr"]:.1e}  ({elapsed:.0f}s)')
            print(msg, flush=True); log(msg)

        if patience_count >= 5000:
            msg = f'  early stop at epoch {epoch}, best_behav={best_behav:.6f}'
            print(msg, flush=True); log(msg)
            break

    model.load_state_dict(torch.load(f'{out_dir}/model.pt', weights_only=True))

    model.eval()
    with torch.no_grad():
        tp, dp, rp, _ = model(all_dir, all_ret, all_foff)
        bg = ((tp[all_gnd == 1] - all_tgt[all_gnd == 1]) ** 2).mean().item()
        bh = ((tp[all_gnd == 0] - all_tgt[all_gnd == 0]) ** 2).mean().item()
        msg = f'  grounded: behav={bg:.6f}'
        print(msg, flush=True); log(msg)
        msg = f'  holdout:  behav={bh:.6f}'
        print(msg, flush=True); log(msg)
        msg = f'  behav gap: {bh - bg:.6f}'
        print(msg, flush=True); log(msg)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ramp_lambda', type=float, default=15.0)
    parser.add_argument('--disp_lambda', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--raw_input', action='store_true')
    parser.add_argument('--joint_proj', action='store_true')
    parser.add_argument('--embed_flatness', type=float, default=None)
    parser.add_argument('--embed_hidden', type=int, default=32)
    parser.add_argument('--embed_layers', type=int, default=2)
    parser.add_argument('--input_noise', type=float, default=0.0)
    parser.add_argument('--training_set', type=str, default='circular')
    args = parser.parse_args()
    train(args.out_dir, lr=args.lr, ramp_lambda=args.ramp_lambda,
          disp_lambda=args.disp_lambda, seed=args.seed, sigma=args.sigma,
          raw_input=args.raw_input, joint_proj=args.joint_proj,
          embed_flatness=args.embed_flatness, input_noise=args.input_noise,
          training_set=args.training_set, embed_hidden=args.embed_hidden,
          embed_layers=args.embed_layers)

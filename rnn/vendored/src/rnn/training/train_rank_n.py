"""Rank sweep: Rank-N RNN (1,2,4,full) with or without pretraining.

Usage:
  # No pretrain, end-to-end:
  python src/rnn/training/train_rank_n.py --out_dir ... --rank 2 --seed 0

  # Pretrain + freeze:
  python src/rnn/training/train_rank_n.py --stage1 --out_dir ... --rank 2 --seed 0
  python src/rnn/training/train_rank_n.py --out_dir ... --rank 2 --seed 0 --pretrained
"""
import sys, os, time, json, argparse
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np

from src.rnn.training.train_rank1_rnn_readout import ReadoutMLP, compute_ramp_loss


class FullRankRNNFlat(nn.Module):
    """Full-rank RNN with ScaledTanh: c * tanh(x/c)."""
    def __init__(self, input_size, hidden_size, alpha, flatness=0.125):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.flatness = flatness
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, h0=None):
        B, T, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x_seq.device)
        hiddens = []
        c = self.flatness
        for t in range(T):
            pre = self.W_ih(x_seq[:, t]) + self.W_hh(h) + self.bias
            h_new = c * torch.tanh(pre / c)
            h = (1 - self.alpha) * h + self.alpha * h_new
            hiddens.append(h)
        return torch.stack(hiddens, dim=1)
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode
from src.rnn.training.train_integrator_stage1 import generate_batch


LOG_PATH = None

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


class RankNIntegrator(nn.Module):
    """Rank-N RNN: W_hh = U @ V^T where U,V are (hidden, N)."""
    def __init__(self, input_size, hidden_size, rank, alpha, flatness=0.125):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.flatness = flatness
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 0.1)
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, h0=None):
        B, T, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x_seq.device)
        hiddens = []
        c = self.flatness
        linear = (c is None)
        for t in range(T):
            rec = (h @ self.V) @ self.U.T
            pre = self.W_ih(x_seq[:, t]) + rec + self.bias
            if linear:
                h_new = pre
            else:
                h_new = c * torch.tanh(pre / c)
            h = (1 - self.alpha) * h + self.alpha * h_new
            hiddens.append(h)
        return torch.stack(hiddens, dim=1)


class RankNModel(nn.Module):
    def __init__(self, integrator, ramp_readout, readout_mlp):
        super().__init__()
        self.integrator = integrator
        self.ramp_readout = ramp_readout
        self.readout_mlp = readout_mlp

    def forward(self, dir_onehot, start_retino, flow_off_bins):
        B, T, _ = dir_onehot.shape
        h_int = self.integrator(dir_onehot)
        ramp = self.ramp_readout(h_int)
        h_snapshot = torch.stack([h_int[i, flow_off_bins[i]] for i in range(B)])
        start_snapshot = torch.stack([start_retino[i, flow_off_bins[i]] for i in range(B)])
        combined = torch.cat([h_snapshot, start_snapshot], dim=1)
        target = self.readout_mlp(combined)
        return target, ramp, {'h_integrator': h_int, 'h_snapshot': h_snapshot}


def build_integrator(input_size, hidden_size, rank, alpha, flatness=0.125):
    if rank == 'full' or rank >= hidden_size:
        return FullRankRNNFlat(input_size=input_size, hidden_size=hidden_size,
                               alpha=alpha, flatness=flatness)
    else:
        return RankNIntegrator(input_size=input_size, hidden_size=hidden_size,
                               rank=rank, alpha=alpha, flatness=flatness)


def train_stage1(out_dir, rank, hidden_size=32, tau=0.1, dt=1/30, lr=3e-3,
                 max_epochs=10000, seed=0, flatness=1.0):
    global LOG_PATH
    os.makedirs(out_dir, exist_ok=True)
    LOG_PATH = f'{out_dir}/train_log.txt'
    open(LOG_PATH, 'w').close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    alpha = dt / tau
    integrator = build_integrator(1, hidden_size, rank, alpha, flatness=flatness)
    readout = nn.Linear(hidden_size, 1)

    params = list(integrator.parameters()) + list(readout.parameters())

    config = {
        'stage': 1, 'rank': rank,
        'dt': dt, 'tau': tau,
        'hidden_size': hidden_size,
        'flatness': flatness,
        'lr': lr, 'seed': seed,
    }
    with open(f'{out_dir}/config_stage1.json', 'w') as f:
        json.dump(config, f, indent=2)

    log(f'Stage 1: Train rank-{rank} integrator on timing')

    optimizer = torch.optim.AdamW(params, lr=lr)
    t0 = time.time()
    best_loss = float('inf')

    for epoch in range(max_epochs):
        inp, tgt = generate_batch(256, dt=dt)
        h = integrator(inp)
        ramp_pred = readout(h)
        valid = ~torch.isnan(tgt)
        loss = ((ramp_pred[valid] - tgt[valid]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'integrator': integrator.state_dict(),
                'readout': readout.state_dict(),
            }, f'{out_dir}/model_stage1.pt')

        if epoch % 500 == 0:
            msg = f'  epoch {epoch:5d}  loss={loss.item():.6f}  best={best_loss:.6f}  ({time.time()-t0:.0f}s)'
            print(msg, flush=True); log(msg)

        if best_loss < 1e-3:
            msg = f'  converged at epoch {epoch}, best={best_loss:.6f}'
            print(msg, flush=True); log(msg)
            break

    msg = f'  final best loss: {best_loss:.6f}'
    print(msg, flush=True); log(msg)


def train_stage2(out_dir, rank, seed=0, lr=1e-3, ramp_lambda=30.0,
                 max_epochs=20000, sigma=1.0, training_set='monkey',
                 pretrained=False, flatness=1.0):
    global LOG_PATH
    os.makedirs(out_dir, exist_ok=True)
    if pretrained:
        LOG_PATH = f'{out_dir}/train_log.txt'
        with open(LOG_PATH, 'a') as f:
            f.write('\n--- Stage 2 ---\n')
    else:
        LOG_PATH = f'{out_dir}/train_log.txt'
        open(LOG_PATH, 'w').close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = 1/30; tau = 0.1; alpha = dt / tau
    hidden_size = 32
    mlp_hidden = 32
    grid_size = 7; n_retino = grid_size * grid_size

    centers = make_retinotopic_centers(grid_size)

    if pretrained:
        with open(f'{out_dir}/config_stage1.json') as f:
            import json as _json
            s1_cfg = _json.load(f)
        flatness = s1_cfg.get('flatness', flatness)

    integrator = build_integrator(4, hidden_size, rank, alpha, flatness=flatness)

    if pretrained:
        s1_ckpt = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)
        s1_state = s1_ckpt['integrator']
        if rank == 'full' or rank >= hidden_size:
            integrator.W_hh.weight.data = s1_state['W_hh.weight']
            integrator.bias.data = s1_state['bias']
            integrator.W_ih.bias.data = s1_state['W_ih.bias']
            integrator.W_hh.weight.requires_grad = False
            if integrator.W_hh.bias is not None:
                integrator.W_hh.bias.requires_grad = False
            integrator.bias.requires_grad = False
        else:
            integrator.U.data = s1_state['U']
            integrator.V.data = s1_state['V']
            integrator.bias.data = s1_state['bias']
            integrator.W_ih.bias.data = s1_state['W_ih.bias']
            integrator.U.requires_grad = False
            integrator.V.requires_grad = False
            integrator.bias.requires_grad = False

        ramp_readout = nn.Linear(hidden_size, 1)
        ramp_readout.weight.data = s1_ckpt['readout']['weight']
        ramp_readout.bias.data = s1_ckpt['readout']['bias']
        for p in ramp_readout.parameters():
            p.requires_grad = False
    else:
        ramp_readout = nn.Linear(hidden_size, 1)

    readout_mlp = ReadoutMLP(hidden_size + n_retino, mlp_hidden)

    if rank == 'full' or (isinstance(rank, int) and rank >= hidden_size):
        model = RankNModel(integrator, ramp_readout, readout_mlp)
    else:
        model = RankNModel(integrator, ramp_readout, readout_mlp)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    version_str = 'rank_n_fullrank' if (rank == 'full' or (isinstance(rank, int) and rank >= hidden_size)) else f'rank_n_r{rank}'
    config = {
        'version': version_str, 'dt': dt, 'tau': tau,
        'hidden_size': hidden_size, 'rank': rank, 'flatness': flatness, 'flatness': flatness,
        'mlp_hidden': mlp_hidden,
        'grid_size': grid_size, 'n_retino': n_retino, 'sigma': sigma,
        'ramp_lambda': ramp_lambda, 'training_set': training_set,
        'pretrained': pretrained, 'frozen_integrator': pretrained,
        'training': {'lr': lr},
    }
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    log(f'Rank-{rank} {"pretrained+frozen" if pretrained else "end-to-end"}')
    log(f'  trainable: {n_trainable}, frozen: {n_frozen}')

    task = DMFCTaskV2(delay=0.2, dt=dt, vector_lengths=[1, 2, 3, 4, 5, 6],
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
            retino = retinotopic_encode(start_xy, centers, sigma)[0]
            dir_channels = inp_raw[:, 2:6]
            retino_tiled = np.tile(retino, (dir_channels.shape[0], 1))
            data.append((dir_channels, retino_tiled, ramp, target, flow_off, grounded))
        return data

    val_data = build_data(ds_val)
    all_data = build_data(ds_train)

    def stack_tensors(data_list):
        dir_t = torch.tensor(np.stack([d[0] for d in data_list]))
        ret_t = torch.tensor(np.stack([d[1] for d in data_list]))
        ramp_t = torch.tensor(np.stack([d[2] for d in data_list]))
        tgt_t = torch.tensor(np.stack([d[3] for d in data_list]))
        foff_t = torch.tensor([d[4] for d in data_list])
        gnd_t = torch.tensor([d[5] for d in data_list])
        return dir_t, ret_t, ramp_t, tgt_t, foff_t, gnd_t

    train_dir, train_ret, train_ramp, train_tgt, train_foff, _ = stack_tensors(val_data)
    all_dir, all_ret, all_ramp, all_tgt, all_foff, all_gnd = stack_tensors(all_data)

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    if pretrained:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=500, factor=0.5, min_lr=1e-5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5)

    best_behav = float('inf')
    patience_count = 0
    ramp_warmup = 2000
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        if rank == 'full' or (isinstance(rank, int) and rank >= hidden_size):
            target_pred, ramp_pred, _ = model(train_dir, train_ret, train_foff)
        else:
            target_pred, ramp_pred, _ = model(train_dir, train_ret, train_foff)
        loss_behav = ((target_pred - train_tgt) ** 2).mean()
        loss_ramp = compute_ramp_loss(ramp_pred, train_ramp)
        rl = ramp_lambda if epoch < ramp_warmup else ramp_lambda * 0.1
        loss = loss_behav + rl * loss_ramp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if pretrained:
            scheduler.step(loss.item())
        else:
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
                tp, rp, _ = model(all_dir, all_ret, all_foff)
                bg = ((tp[all_gnd == 1] - all_tgt[all_gnd == 1]) ** 2).mean().item()
                bh = ((tp[all_gnd == 0] - all_tgt[all_gnd == 0]) ** 2).mean().item()
            msg = (f'  epoch {epoch:5d}  loss={loss.item():.6f}  '
                   f'behav={bg:.4f}/{bh:.4f}  '
                   f'lr={optimizer.param_groups[0]["lr"]:.1e}  ({elapsed:.0f}s)')
            print(msg, flush=True); log(msg)

        if pretrained and patience_count >= 2000:
            msg = f'  early stop at epoch {epoch}, best_behav={best_behav:.6f}'
            print(msg, flush=True); log(msg)
            break
        if not pretrained and patience_count >= 5000:
            msg = f'  early stop at epoch {epoch}, best_behav={best_behav:.6f}'
            print(msg, flush=True); log(msg)
            break

    model.load_state_dict(torch.load(f'{out_dir}/model.pt', weights_only=True))
    model.eval()
    with torch.no_grad():
        tp, rp, _ = model(all_dir, all_ret, all_foff)
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
    parser.add_argument('--rank', type=str, required=True)
    parser.add_argument('--stage1', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ramp_lambda', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--training_set', type=str, default='monkey')
    parser.add_argument('--flatness', type=float, default=1.0)
    args = parser.parse_args()

    rank = 'full' if args.rank == 'full' else int(args.rank)

    if args.stage1:
        train_stage1(args.out_dir, rank=rank, seed=args.seed, flatness=args.flatness)
    else:
        train_stage2(args.out_dir, rank=rank, seed=args.seed, lr=args.lr,
                     ramp_lambda=args.ramp_lambda, sigma=args.sigma,
                     training_set=args.training_set, pretrained=args.pretrained,
                     flatness=args.flatness)

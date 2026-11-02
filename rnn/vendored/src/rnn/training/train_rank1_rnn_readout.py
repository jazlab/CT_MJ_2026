"""Rank-1 RNN + MLP readout.

MT(4D) → rank-1 RNN(32, ramp-supervised) → snapshot at flow_off
[h_snapshot(32), FEF(49D)] → MLP(81→32→32→32→2) → target

Loss: L_behav + λ_ramp * L_ramp. No displacement supervision.

Run:
  python src/rnn/training/train_rank1_rnn_readout.py --out_dir OUT_DIR/s0
"""
import sys, os, time, json, argparse
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np

from src.rnn.models.dmfc_model_v2 import Rank1Integrator
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode


LOG_PATH = None

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


class ReadoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class V2Model(nn.Module):
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


def compute_ramp_loss(ramp_pred, ramp_tgt):
    valid = ~torch.isnan(ramp_tgt)
    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return ((ramp_pred[valid] - ramp_tgt[valid]) ** 2).mean()


def train(out_dir, delay=0.2, seed=0, lr=1e-3, ramp_lambda=30.0,
          max_epochs=20000, sigma=1.0, input_noise=0.0, training_set='circular'):
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
    grid_size = 7; n_retino = grid_size * grid_size

    centers = make_retinotopic_centers(grid_size)

    integrator = Rank1Integrator(input_size=4, hidden_size=hidden_size,
                                  alpha=alpha, linear=linear)
    integrator.u.data = s1_ckpt['integrator']['u']
    integrator.v.data = s1_ckpt['integrator']['v']
    integrator.bias.data = s1_ckpt['integrator']['bias']
    integrator.W_ih.bias.data = s1_ckpt['integrator']['W_ih.bias']

    ramp_readout = nn.Linear(hidden_size, 1)
    ramp_readout.weight.data = s1_ckpt['readout']['weight']
    ramp_readout.bias.data = s1_ckpt['readout']['bias']

    readout_mlp = ReadoutMLP(hidden_size + n_retino, mlp_hidden)

    model = V2Model(integrator, ramp_readout, readout_mlp)

    integrator.u.requires_grad = False
    integrator.v.requires_grad = False
    integrator.bias.requires_grad = False
    for p in ramp_readout.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    config = {
        'version': 'rank1_rnn_readout', 'dt': dt, 'tau': tau,
        'hidden_size': hidden_size, 'linear': linear,
        'mlp_hidden': mlp_hidden,
        'grid_size': grid_size, 'n_retino': n_retino, 'sigma': sigma,
        'ramp_lambda': ramp_lambda,
        'input_noise': input_noise,
        'training': {'lr': lr},
    }
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    log(f'rank-1 RNN readout: MT(4D)→rank-1 RNN(32)→[h+FEF]→MLP→target')
    log(f'  sigma={sigma}, grid={grid_size}x{grid_size}={n_retino}, input_noise={input_noise}')
    log(f'  trainable: {n_trainable}, frozen: {n_frozen}')
    log(f'  ramp_lambda={ramp_lambda}, lr={lr}')

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

    log(f'  train (grounded): {len(val_data)}, all: {len(all_data)}')

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-5)

    best_loss = float('inf')
    patience_count = 0
    eval_losses = []
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        if input_noise > 0:
            noisy_dir = train_dir * (1 + input_noise * torch.randn_like(train_dir))
            noisy_ret = train_ret * (1 + input_noise * torch.randn_like(train_ret))
        else:
            noisy_dir, noisy_ret = train_dir, train_ret
        target_pred, ramp_pred, _ = model(noisy_dir, noisy_ret, train_foff)
        loss_behav = ((target_pred - train_tgt) ** 2).mean()
        loss_ramp = compute_ramp_loss(ramp_pred, train_ramp)
        loss = loss_behav + ramp_lambda * loss_ramp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
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
                lr_ = compute_ramp_loss(rp, all_ramp).item()
            msg = (f'  epoch {epoch:5d}  loss={loss.item():.6f}  '
                   f'behav={bg:.4f}/{bh:.4f}  ramp={lr_:.4f}  '
                   f'lr={optimizer.param_groups[0]["lr"]:.1e}  ({elapsed:.0f}s)')
            print(msg, flush=True); log(msg)

            eval_losses.append(loss.item())
            if len(eval_losses) >= 11:
                last10 = eval_losses[-10:]
                prev = eval_losses[-11]
                if all(l >= prev for l in last10):
                    msg = f'  no decay for 10 consecutive evals at epoch {epoch}, best={best_loss:.6f}'
                    print(msg, flush=True); log(msg)
                    break

        scheduler.step(loss.item())

        if patience_count >= 2000:
            msg = f'  early stop at epoch {epoch}, best={best_loss:.6f}'
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ramp_lambda', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--input_noise', type=float, default=0.0)
    parser.add_argument('--training_set', type=str, default='circular')
    args = parser.parse_args()
    train(args.out_dir, lr=args.lr, ramp_lambda=args.ramp_lambda,
          seed=args.seed, sigma=args.sigma, input_noise=args.input_noise,
          training_set=args.training_set)

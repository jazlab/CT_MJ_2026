"""Full-rank RNN + readout ([h; Start] -> linear or MLP): models a-c.

MT(4D) → full-rank RNN(32) → snapshot at flow_off
[h_snapshot(32), FEF(49D)] → MLP(81→32→32→32→2) → target

Loss: L_behav only. No ramp, no displacement supervision.

Run:
  python src/rnn/training/train_rnn_readout.py --out_dir OUT_DIR/s0
"""
import sys, os, time, json, argparse
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np

from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode


LOG_PATH = None

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


class FullRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, linear=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.linear = linear
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, h0=None):
        B, T, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x_seq.device)
        hiddens = []
        for t in range(T):
            pre = self.W_ih(x_seq[:, t]) + self.W_hh(h) + self.bias
            if self.linear:
                h_new = pre
            else:
                h_new = torch.tanh(pre)
            h = (1 - self.alpha) * h + self.alpha * h_new
            hiddens.append(h)
        return torch.stack(hiddens, dim=1)


class ScaledTanh(nn.Module):
    def __init__(self, flatness=1.0):
        super().__init__()
        self.flatness = flatness

    def forward(self, x):
        c = self.flatness
        if c == float('inf'):
            return x
        return c * torch.tanh(x / c)


class ReadoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim=2, n_layers=3,
                 flatness=1.0):
        super().__init__()
        if n_layers == 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_size), ScaledTanh(flatness)]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_size, hidden_size), ScaledTanh(flatness)]
            layers.append(nn.Linear(hidden_size, output_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class V1Model(nn.Module):
    def __init__(self, rnn, readout_mlp):
        super().__init__()
        self.rnn = rnn
        self.readout_mlp = readout_mlp

    def forward(self, dir_onehot, start_retino, flow_off_bins):
        B, T, _ = dir_onehot.shape
        h_int = self.rnn(dir_onehot)
        h_snapshot = torch.stack([h_int[i, flow_off_bins[i]] for i in range(B)])
        start_snapshot = torch.stack([start_retino[i, flow_off_bins[i]] for i in range(B)])
        combined = torch.cat([h_snapshot, start_snapshot], dim=1)
        target = self.readout_mlp(combined)
        return target, {'h_rnn': h_int, 'h_snapshot': h_snapshot}


def train(out_dir, delay=0.2, seed=0, lr=1e-3, max_epochs=20000, sigma=1.0,
          input_noise=0.0, training_set='circular', mlp_hidden=32,
          mlp_layers=3, mlp_flatness=1.0):
    global LOG_PATH
    os.makedirs(out_dir, exist_ok=True)
    LOG_PATH = f'{out_dir}/train_log.txt'
    open(LOG_PATH, 'w').close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = 1/30; tau = 0.1; alpha = dt / tau
    hidden_size = 32
    grid_size = 7; n_retino = grid_size * grid_size

    centers = make_retinotopic_centers(grid_size)

    rnn = FullRankRNN(input_size=4, hidden_size=hidden_size, alpha=alpha)
    readout_mlp = ReadoutMLP(hidden_size + n_retino, mlp_hidden,
                              n_layers=mlp_layers, flatness=mlp_flatness)
    model = V1Model(rnn, readout_mlp)

    trainable = list(model.parameters())
    n_trainable = sum(p.numel() for p in trainable)

    config = {
        'version': 'rnn_readout', 'dt': dt, 'tau': tau,
        'hidden_size': hidden_size, 'mlp_hidden': mlp_hidden,
        'mlp_layers': mlp_layers, 'mlp_flatness': mlp_flatness,
        'grid_size': grid_size, 'n_retino': n_retino, 'sigma': sigma,
        'input_noise': input_noise, 'training_set': training_set,
        'training': {'lr': lr},
    }
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    log(f'RNN readout: MT(4D)→full-rank RNN(32)→[h+FEF]→MLP→target')
    log(f'  sigma={sigma}, grid={grid_size}x{grid_size}={n_retino}, input_noise={input_noise}')
    log(f'  trainable: {n_trainable}, lr={lr}')

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
            data.append((dir_channels, retino_tiled, target, flow_off, grounded))
        return data

    val_data = build_data(ds_val)
    all_data = build_data(ds_train)

    def stack_tensors(data_list):
        dir_t = torch.tensor(np.stack([d[0] for d in data_list]))
        ret_t = torch.tensor(np.stack([d[1] for d in data_list]))
        tgt_t = torch.tensor(np.stack([d[2] for d in data_list]))
        foff_t = torch.tensor([d[3] for d in data_list])
        gnd_t = torch.tensor([d[4] for d in data_list])
        return dir_t, ret_t, tgt_t, foff_t, gnd_t

    train_dir, train_ret, train_tgt, train_foff, _ = stack_tensors(val_data)
    all_dir, all_ret, all_tgt, all_foff, all_gnd = stack_tensors(all_data)

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
        target_pred, _ = model(noisy_dir, noisy_ret, train_foff)
        loss = ((target_pred - train_tgt) ** 2).mean()

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
                tp, _ = model(all_dir, all_ret, all_foff)
                bg = ((tp[all_gnd == 1] - all_tgt[all_gnd == 1]) ** 2).mean().item()
                bh = ((tp[all_gnd == 0] - all_tgt[all_gnd == 0]) ** 2).mean().item()
            msg = (f'  epoch {epoch:5d}  loss={loss.item():.6f}  '
                   f'behav={bg:.4f}/{bh:.4f}  '
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
        tp, _ = model(all_dir, all_ret, all_foff)
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--input_noise', type=float, default=0.0)
    parser.add_argument('--training_set', type=str, default='circular')
    parser.add_argument('--mlp_hidden', type=int, default=32)
    parser.add_argument('--mlp_layers', type=int, default=3)
    parser.add_argument('--mlp_flatness', type=float, default=1.0)
    args = parser.parse_args()
    train(args.out_dir, lr=args.lr, seed=args.seed, sigma=args.sigma,
          input_noise=args.input_noise, training_set=args.training_set,
          mlp_hidden=args.mlp_hidden, mlp_layers=args.mlp_layers,
          mlp_flatness=args.mlp_flatness)

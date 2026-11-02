"""Stage 1: Train rank-1 integrator alone on pure timing task.

Input: 1D phasic signal (random amplitude ±uniform) for variable duration.
Target: linear ramp 0 → duration (in seconds) during the phasic input.
No comp RNN, no behavioral loss. Just learn to integrate time.

The input amplitude varies randomly so the integrator can't just
track the input — it must integrate elapsed time regardless of amplitude.

Run:
  python src/rnn/training/train_integrator_stage1.py --out_dir OUT_DIR
"""
import sys, os, time, json, argparse
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np

from src.rnn.models.dmfc_model_v2 import Rank1Integrator


LOG_PATH = None

def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def generate_batch(batch_size, dt=1/30, flow_onset=0.3, max_flow_dur=2.0,
                   total_time=2.7, n_durations=6, ball_speed=3):
    """Generate random timing trials.

    Input: 1D phasic signal, always +1, for variable duration.
    Target: linear ramp 0 → flow_duration during flow, NaN elsewhere.
    """
    n_timesteps = int(np.round(total_time / dt))
    flow_on = int(np.round(flow_onset / dt))

    # Durations matching task lengths 1-6 at ball_speed=3
    durations = [L / ball_speed for L in range(1, n_durations + 1)]

    inputs = np.zeros((batch_size, n_timesteps, 1), dtype=np.float32)
    targets = np.full((batch_size, n_timesteps, 1), np.nan, dtype=np.float32)

    for i in range(batch_size):
        dur = durations[np.random.randint(len(durations))]
        n_flow = int(np.round(dur / dt))
        flow_off = flow_on + n_flow

        inputs[i, flow_on:flow_off, 0] = 1.0
        targets[i, flow_on:flow_off, 0] = np.linspace(0, dur, n_flow)

    return torch.tensor(inputs), torch.tensor(targets)


def train(out_dir, hidden_size=32, tau=0.1, dt=1/30, lr=3e-3,
          max_epochs=5000, seed=0, linear=False):
    global LOG_PATH
    os.makedirs(out_dir, exist_ok=True)
    LOG_PATH = f'{out_dir}/train_log.txt'
    open(LOG_PATH, 'w').close()

    torch.manual_seed(seed)
    np.random.seed(seed)

    alpha = dt / tau

    model = Rank1Integrator(input_size=1, hidden_size=hidden_size, alpha=alpha,
                            linear=linear)
    readout = nn.Linear(hidden_size, 1)

    params = list(model.parameters()) + list(readout.parameters())
    n_params = sum(p.numel() for p in params)

    config = {
        'stage': 1,
        'dt': dt,
        'tau': tau,
        'hidden_size': hidden_size,
        'input_size': 1,
        'linear': linear,
        'n_params': n_params,
        'lr': lr,
        'seed': seed,
    }
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    log(f'Stage 1: Train rank-1 integrator on timing (linear={linear})')
    log(f'  hidden={hidden_size}, tau={tau}, dt={dt}, alpha={alpha:.4f}')
    log(f'  params={n_params}')

    optimizer = torch.optim.AdamW(params, lr=lr)
    batch_size = 256

    t0 = time.time()
    best_loss = float('inf')
    for epoch in range(max_epochs):
        inp, tgt = generate_batch(batch_size, dt=dt)

        h = model(inp)
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
                'integrator': model.state_dict(),
                'readout': readout.state_dict(),
            }, f'{out_dir}/model.pt')

        if epoch % 100 == 0:
            elapsed = time.time() - t0
            msg = f'  epoch {epoch:5d}  loss={loss.item():.6f}  best={best_loss:.6f}  ({elapsed:.0f}s)'
            print(msg, flush=True)
            log(msg)

        if best_loss < 1e-3:
            msg = f'  converged at epoch {epoch}, best={best_loss:.6f}'
            print(msg, flush=True); log(msg)
            break

    msg = f'  final best loss: {best_loss:.6f}'
    print(msg, flush=True); log(msg)

    # Check rank-1 structure
    u = model.u.detach().numpy().flatten()
    v = model.v.detach().numpy().flatten()
    w = readout.weight.detach().numpy().flatten()
    u_n = u / np.linalg.norm(u)
    v_n = v / np.linalg.norm(v)
    w_n = w / np.linalg.norm(w)
    msg = (f'  cos(u,v)={np.dot(u_n,v_n):.4f}  '
           f'cos(w,v)={np.dot(w_n,v_n):.4f}  '
           f'cos(w,u)={np.dot(w_n,u_n):.4f}  '
           f'eigenval v^Tu={np.dot(v,u):.4f}')
    print(msg, flush=True); log(msg)

    return model, readout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--linear', action='store_true')
    args = parser.parse_args()

    train(args.out_dir, hidden_size=args.hidden_size, lr=args.lr, seed=args.seed,
          linear=args.linear)

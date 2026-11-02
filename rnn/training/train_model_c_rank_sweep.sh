#!/bin/bash
#SBATCH --job-name=train_model_c_rank_sweep
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-39
#SBATCH --time=06:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Model c rank sweep (S4.1b): pretrained + frozen rank-N RNN, deep FFN readout (6 x 128, flatness 0.125), ramp loss
# ranks 1, 2, 4, 16 x 10 seeds -> rnn/models/model_c/rank{1,2,4,16}/s{0..9} (full rank: train_model_c.sh)

REPO=/path/to/CT_MJ_2026                 # set to this repository
ROOT="$REPO/rnn/vendored"
MODELS="$REPO/rnn/models"
cd "$ROOT"

export LD_LIBRARY_PATH=$HOME/.conda/envs/rnn_cpu/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=.

PY=$HOME/.conda/envs/rnn_cpu/bin/python
TID=$SLURM_ARRAY_TASK_ID
RANKS=(1 2 4 16)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

RANK_IDX=$((TID / 10))
SEED_IDX=$((TID % 10))
RANK=${RANKS[$RANK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT_DIR="$MODELS/model_c/rank${RANK}/s${SEED}"
mkdir -p "$OUT_DIR"

echo "model_c rank=${RANK} seed=${SEED}"

$PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rank_n import RankNIntegrator, FullRankRNNFlat
from src.rnn.training.train_rnn_readout import ReadoutMLP
from src.rnn.training.train_rank1_rnn_readout import compute_ramp_loss
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode
from src.rnn.training.train_integrator_stage1 import generate_batch

out_dir = '${OUT_DIR}'
rank_str = '${RANK}'
rank = 32 if rank_str == 'full' else int(rank_str)
seed = ${SEED}
torch.manual_seed(seed); np.random.seed(seed)
dt=1/30; tau=0.1; alpha=dt/tau; H=32; grid_size=7; n_retino=49; sigma=1.0
centers = make_retinotopic_centers(grid_size)

def make_rnn(in_size, rank, H, alpha):
    if rank >= H:
        return FullRankRNNFlat(input_size=in_size, hidden_size=H, alpha=alpha, flatness=1.0)
    else:
        return RankNIntegrator(input_size=in_size, hidden_size=H, rank=rank, alpha=alpha, flatness=1.0)

# Stage 1: pretrain on timing
print('=== Stage 1 ===')
rnn_s1 = make_rnn(1, rank, H, alpha)
readout_s1 = nn.Linear(H, 1)
params_s1 = list(rnn_s1.parameters()) + list(readout_s1.parameters())
opt_s1 = torch.optim.AdamW(params_s1, lr=3e-3)
bl = float('inf'); t0 = time.time()
for ep in range(10000):
    inp, tgt = generate_batch(256, dt=dt)
    h = rnn_s1(inp); rp = readout_s1(h)
    valid = ~torch.isnan(tgt); loss = ((rp[valid]-tgt[valid])**2).mean()
    opt_s1.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params_s1, 1.0); opt_s1.step()
    if loss.item() < bl:
        bl = loss.item()
        torch.save({'rnn': rnn_s1.state_dict(), 'readout': readout_s1.state_dict()}, f'{out_dir}/model_stage1.pt')
    if ep % 500 == 0:
        print(f'  s1 ep {ep:5d}  loss={loss.item():.6f}  best={bl:.6f}  ({time.time()-t0:.0f}s)', flush=True)
    if bl < 1e-3:
        print(f'  converged at ep {ep}', flush=True); break
with open(f'{out_dir}/config_stage1.json', 'w') as f:
    json.dump({'stage':1, 'rank':rank, 'dt':dt, 'tau':tau, 'hidden_size':H}, f, indent=2)

# Stage 2: freeze integrator, train with ramp loss
print('=== Stage 2 (with ramp loss) ===')
rnn = make_rnn(4, rank, H, alpha)
s1 = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)

if rank >= H:
    rnn.W_hh.weight.data = s1['rnn']['W_hh.weight']
    rnn.bias.data = s1['rnn']['bias']
    rnn.W_ih.bias.data = s1['rnn']['W_ih.bias']
    rnn.W_hh.weight.requires_grad = False
    rnn.bias.requires_grad = False
else:
    rnn.U.data = s1['rnn']['U']; rnn.V.data = s1['rnn']['V']
    rnn.bias.data = s1['rnn']['bias']; rnn.W_ih.bias.data = s1['rnn']['W_ih.bias']
    rnn.U.requires_grad = False; rnn.V.requires_grad = False
    rnn.bias.requires_grad = False

ramp_readout = nn.Linear(H, 1)
ramp_readout.weight.data = s1['readout']['weight']
ramp_readout.bias.data = s1['readout']['bias']
for p in ramp_readout.parameters(): p.requires_grad = False

mlp = ReadoutMLP(H+n_retino, 128, n_layers=6, flatness=0.125)

class Model(nn.Module):
    def __init__(self, rnn, ramp_readout, mlp):
        super().__init__()
        self.rnn = rnn; self.ramp_readout = ramp_readout; self.readout_mlp = mlp
    def forward(self, d, s, fo):
        B,T,_=d.shape; h=self.rnn(d); ramp=self.ramp_readout(h)
        hs=torch.stack([h[i,fo[i]] for i in range(B)])
        ss=torch.stack([s[i,fo[i]] for i in range(B)])
        return self.readout_mlp(torch.cat([hs,ss],dim=1)), ramp, {'h_integrator':h}

model = Model(rnn, ramp_readout, mlp)
trainable = [p for p in model.parameters() if p.requires_grad]
nt = sum(p.numel() for p in trainable)
nf = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f'  trainable: {nt}, frozen: {nf}')

config = {'version': 'rnn_readout','dt':dt,'tau':tau,'hidden_size':H,'mlp_hidden':128,
          'mlp_layers':6,'mlp_flatness':0.125,'grid_size':grid_size,'n_retino':n_retino,
          'sigma':sigma,'training_set':'monkey','rank':rank if rank<H else 'full',
          'pretrained':True,'frozen_integrator':True,
          'ramp_lambda':30.0,'training':{'lr':0.001}}
with open(f'{out_dir}/config.json','w') as f: json.dump(config,f,indent=2)

task = DMFCTaskV2(delay=0.2, dt=dt, vector_lengths=[1,2,3,4,5,6], training_set='monkey', scale=1, tau=tau)
ball_speed = task.BALL_SPEED; ds_train, ds_val = task.get_datasets()
def build(ds):
    data = []
    for i in range(len(ds)):
        inp_raw, tgt_raw, ramp, grounded = ds[i]; cond = ds.condition_labels[i]
        sxy = np.array(cond[0], dtype=np.float32).reshape(1,2); vec = np.array(cond[1])
        fo = int(np.round((task.FLOW_ONSET_TIME + np.linalg.norm(vec)/ball_speed) / dt))
        t = np.array(cond[2], dtype=np.float32); r = retinotopic_encode(sxy, centers, sigma)[0]
        data.append((inp_raw[:,2:6], np.tile(r,(inp_raw.shape[0],1)), ramp, t, fo, grounded))
    return data
vd = build(ds_val); ad = build(ds_train)
def stk(dl):
    return (torch.tensor(np.stack([d[0] for d in dl])), torch.tensor(np.stack([d[1] for d in dl])),
            torch.tensor(np.stack([d[2] for d in dl])), torch.tensor(np.stack([d[3] for d in dl])),
            torch.tensor([d[4] for d in dl]), torch.tensor([d[5] for d in dl]))
td,tr,tramp,tt,tf,_ = stk(vd); ad2,ar,aramp,at,af,ag = stk(ad)

opt = torch.optim.AdamW(trainable, lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=500, factor=0.5, min_lr=1e-5)
bl = float('inf'); pc = 0; t0 = time.time()
for ep in range(20000):
    model.train(); tp, rp, _ = model(td, tr, tf)
    loss_behav = ((tp-tt)**2).mean()
    loss_ramp = compute_ramp_loss(rp, tramp)
    loss = loss_behav + 30.0 * loss_ramp
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    if loss.item() < bl: bl=loss.item(); pc=0; torch.save(model.state_dict(), f'{out_dir}/model.pt')
    else: pc += 1
    if ep % 100 == 0:
        with torch.no_grad():
            tp2, rp2, _ = model(ad2, ar, af)
            bg = ((tp2[ag==1]-at[ag==1])**2).mean().item()
            bh = ((tp2[ag==0]-at[ag==0])**2).mean().item()
            lr_ = compute_ramp_loss(rp2, aramp).item()
        print(f'  ep {ep:5d}  loss={loss.item():.6f}  behav={bg:.4f}/{bh:.4f}  ramp={lr_:.4f}  lr={opt.param_groups[0][\"lr\"]:.1e}  ({time.time()-t0:.0f}s)', flush=True)
    sch.step(loss.item())
    if pc >= 2000: print(f'  early stop at ep {ep}', flush=True); break

model.load_state_dict(torch.load(f'{out_dir}/model.pt', weights_only=True)); model.eval()
with torch.no_grad():
    tp, rp, _ = model(ad2, ar, af)
    bg = ((tp[ag==1]-at[ag==1])**2).mean().item()
    bh = ((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded: behav={bg:.6f}'); print(f'  holdout:  behav={bh:.6f}')
    print(f'  behav gap: {bh-bg:.6f}')
"

echo "Done task $TID"

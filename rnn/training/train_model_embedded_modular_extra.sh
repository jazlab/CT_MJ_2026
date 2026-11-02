#!/bin/bash
#SBATCH --job-name=train_model_embedded_modular_extra
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-19
#SBATCH --time=12:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Embedded modular-readout model (S5.3b), additional embedding flatness 0.125 and 0.25, 10 seeds each
# Tasks 0-9: 0.125; 10-19: 0.25 -> rnn/models/model_embedded_modular/flatness{0.125,0.25}/s{0..9} (shipped: flatness0.125)

REPO=/path/to/CT_MJ_2026                 # set to this repository
ROOT="$REPO/rnn/vendored"
MODELS="$REPO/rnn/models"
cd "$ROOT"
export LD_LIBRARY_PATH=$HOME/.conda/envs/rnn_cpu/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export PYTHONPATH=.
PY=$HOME/.conda/envs/rnn_cpu/bin/python
TID=$SLURM_ARRAY_TASK_ID

FLAT_IDX=$((TID / 10))
SEED=$((TID % 10))
FLATNESS_VALS=(0.125 0.25)
FLATNESS_DIRS=(flatness0.125 flatness0.25)
FLAT=${FLATNESS_VALS[$FLAT_IDX]}
FDIR=${FLATNESS_DIRS[$FLAT_IDX]}

OUT_DIR="$MODELS/model_embedded_modular/${FDIR}/s${SEED}"
mkdir -p "$OUT_DIR"
echo "model_embedded_modular ${FDIR} seed=${SEED}"

$PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN
from src.rnn.training.train_modular_readout import (ModularReadoutModel, DisplacementMLP, EmbeddingMLP53,
    make_retinotopic_centers, retinotopic_encode, compute_ramp_loss)
from src.rnn.training.train_integrator_stage1 import generate_batch
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2

out_dir = '${OUT_DIR}'; seed = ${SEED}; embed_flatness = ${FLAT}
torch.manual_seed(seed); np.random.seed(seed)
dt = 1/30; tau = 0.1; alpha = dt/tau; H = 32; mlp_hidden = 32
grid_size = 7; n_retino = 49; sigma = 1.0; embed_hidden = 32; embed_layers = 1
centers = make_retinotopic_centers(grid_size)

print('=== Stage 1 ===')
rnn_s1 = FullRankRNN(input_size=1, hidden_size=H, alpha=alpha)
readout_s1 = nn.Linear(H, 1)
params_s1 = list(rnn_s1.parameters()) + list(readout_s1.parameters())
opt_s1 = torch.optim.AdamW(params_s1, lr=3e-3)
bl = float('inf'); t0 = time.time()
with open(f'{out_dir}/config_stage1.json', 'w') as f:
    json.dump({'stage':1, 'dt':dt, 'tau':tau, 'hidden_size':H}, f, indent=2)
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
        print(f'  converged ep {ep}', flush=True); break

print('=== Stage 2 ===')
s1 = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)
embed_mlp = EmbeddingMLP53(hidden_size=embed_hidden, flatness=embed_flatness, n_layers=embed_layers)
rnn = FullRankRNN(input_size=4, hidden_size=H, alpha=alpha)
rnn.W_hh.weight.data = s1['rnn']['W_hh.weight']
rnn.bias.data = s1['rnn']['bias']; rnn.W_ih.bias.data = s1['rnn']['W_ih.bias']
rnn.W_hh.weight.requires_grad = False; rnn.bias.requires_grad = False
ramp_readout = nn.Linear(H, 1)
ramp_readout.weight.data = s1['readout']['weight']; ramp_readout.bias.data = s1['readout']['bias']
for p in ramp_readout.parameters(): p.requires_grad = False
disp_mlp = DisplacementMLP(H, mlp_hidden)
w_target = nn.Linear(n_retino + 2, 2)
model = ModularReadoutModel(w_mt=None, w_fef=None, integrator=rnn, ramp_readout=ramp_readout,
                  disp_mlp=disp_mlp, w_target=w_target, embed_mlp=embed_mlp)
trainable = [p for p in model.parameters() if p.requires_grad]
config = {'version': 'modular_readout', 'dt':dt, 'tau':tau, 'hidden_size':H, 'mlp_hidden':mlp_hidden,
          'grid_size':grid_size, 'n_retino':n_retino, 'sigma':sigma,
          'raw_input':False, 'joint_proj':False, 'embed_flatness':embed_flatness,
          'embed_hidden':embed_hidden, 'embed_layers':embed_layers,
          'flow_dim':4, 'start_dim':n_retino, 'ramp_lambda':30.0, 'disp_lambda':1.0,
          'training_set':'monkey', 'rnn_type':'full_rank',
          'pretrained':True, 'frozen_integrator':True,
          'training':{'lr':0.001}}
with open(f'{out_dir}/config.json', 'w') as f: json.dump(config, f, indent=2)

task = DMFCTaskV2(delay=0.2, dt=dt, vector_lengths=[1,2,3,4,5,6], training_set='monkey', scale=1, tau=tau)
ball_speed = task.BALL_SPEED; ds_train, ds_val = task.get_datasets()
def build(ds):
    data = []
    for i in range(len(ds)):
        inp, tgt, ramp, grounded = ds[i]; cond = ds.condition_labels[i]
        sxy = np.array(cond[0], dtype=np.float32).reshape(1,2); vec = np.array(cond[1])
        fo = int(np.round((task.FLOW_ONSET_TIME + np.linalg.norm(vec)/ball_speed) / dt))
        t = np.array(cond[2], dtype=np.float32); disp = np.array(vec, dtype=np.float32)
        r = retinotopic_encode(sxy, centers, sigma)[0]
        data.append((inp[:,2:6], np.tile(r,(inp.shape[0],1)), ramp, t, disp, fo, grounded))
    return data
vd = build(ds_val); ad = build(ds_train)
def stk(dl):
    return (torch.tensor(np.stack([d[0] for d in dl])), torch.tensor(np.stack([d[1] for d in dl])),
            torch.tensor(np.stack([d[2] for d in dl])), torch.tensor(np.stack([d[3] for d in dl])),
            torch.tensor(np.stack([d[4] for d in dl])), torch.tensor([d[5] for d in dl]),
            torch.tensor([d[6] for d in dl]))
td,tr,tramp,tt,tdisp,tf,_ = stk(vd); ad2,ar,aramp,at,adisp,af,ag = stk(ad)
opt = torch.optim.AdamW(trainable, lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40000, eta_min=1e-5)
bl = float('inf'); pc = 0; t0 = time.time(); ramp_warmup = 2000
for ep in range(40000):
    model.train(); tp, dp, rp, _ = model(td, tr, tf)
    lb = ((tp-tt)**2).mean(); ld = ((dp-tdisp)**2).mean(); lr_ = compute_ramp_loss(rp, tramp)
    rl = 30.0 if ep < ramp_warmup else 3.0
    loss = lb + 1.0*ld + rl*lr_
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sch.step()
    if lb.item() < bl: bl = lb.item(); pc = 0; torch.save(model.state_dict(), f'{out_dir}/model.pt')
    else: pc += 1
    if ep % 200 == 0:
        with torch.no_grad():
            tp2, dp2, rp2, _ = model(ad2, ar, af)
            bg = ((tp2[ag==1]-at[ag==1])**2).mean().item()
            bh = ((tp2[ag==0]-at[ag==0])**2).mean().item()
        print(f'  ep {ep:5d}  behav={bg:.4f}/{bh:.4f}  lr={opt.param_groups[0][\"lr\"]:.1e}  ({time.time()-t0:.0f}s)', flush=True)
    if pc >= 5000: print(f'  early stop ep {ep}', flush=True); break
model.load_state_dict(torch.load(f'{out_dir}/model.pt', weights_only=True)); model.eval()
with torch.no_grad():
    tp, dp, rp, _ = model(ad2, ar, af)
    bg = ((tp[ag==1]-at[ag==1])**2).mean().item(); bh = ((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded:{bg:.6f} holdout:{bh:.6f} gap:{bh-bg:.6f}')
"
echo "Done task $SLURM_ARRAY_TASK_ID"

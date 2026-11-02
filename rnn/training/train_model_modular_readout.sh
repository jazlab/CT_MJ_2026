#!/bin/bash
#SBATCH --job-name=train_model_modular_readout
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-9
#SBATCH --time=06:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Modular-readout model (S4.2c, S5.3a): full-rank RNN + modular (factorized) readout on the raw input, ramp and
# displacement losses. 10 seeds -> rnn/models/model_modular_readout/s{0..9}

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
SEEDS=(0 1 2 3 4 5 6 7 8 9)

if [ $TID -lt 10 ]; then
    SEED=${SEEDS[$TID]}
    OUT_DIR="$MODELS/model_modular_readout/s${SEED}"
    mkdir -p "$OUT_DIR"
    echo "model_modular_readout seed=${SEED}"

    $PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN
from src.rnn.training.train_modular_readout import (ModularReadoutModel, DisplacementMLP, make_retinotopic_centers,
    retinotopic_encode, compute_ramp_loss)
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2

out_dir = '${OUT_DIR}'; seed = ${SEED}
torch.manual_seed(seed); np.random.seed(seed)
dt=1/30; tau=0.1; alpha=dt/tau; H=32; mlp_hidden=32
grid_size=7; n_retino=49; sigma=1.0
centers = make_retinotopic_centers(grid_size)

rnn = FullRankRNN(input_size=4, hidden_size=H, alpha=alpha)
ramp_readout = nn.Linear(H, 1)
disp_mlp = DisplacementMLP(H, mlp_hidden)
w_target = nn.Linear(n_retino + 2, 2)
model = ModularReadoutModel(w_mt=None, w_fef=None, integrator=rnn, ramp_readout=ramp_readout,
                  disp_mlp=disp_mlp, w_target=w_target)

trainable = list(model.parameters())
config = {'version': 'modular_readout', 'dt':dt, 'tau':tau, 'hidden_size':H, 'mlp_hidden':mlp_hidden,
          'grid_size':grid_size, 'n_retino':n_retino, 'sigma':sigma,
          'raw_input':True, 'joint_proj':False, 'embed_flatness':None,
          'embed_hidden':32, 'embed_layers':2,
          'flow_dim':4, 'start_dim':n_retino, 'ramp_lambda':30.0, 'disp_lambda':1.0,
          'training_set':'monkey', 'rnn_type':'full_rank',
          'training':{'lr':0.001}}
with open(f'{out_dir}/config.json','w') as f: json.dump(config,f,indent=2)

task = DMFCTaskV2(delay=0.2, dt=dt, vector_lengths=[1,2,3,4,5,6], training_set='monkey', scale=1, tau=tau)
ball_speed = task.BALL_SPEED; ds_train, ds_val = task.get_datasets()
def build(ds):
    data = []
    for i in range(len(ds)):
        inp_raw, tgt_raw, ramp, grounded = ds[i]; cond = ds.condition_labels[i]
        sxy = np.array(cond[0], dtype=np.float32).reshape(1,2); vec = np.array(cond[1])
        fo = int(np.round((task.FLOW_ONSET_TIME + np.linalg.norm(vec)/ball_speed) / dt))
        target = np.array(cond[2], dtype=np.float32)
        disp_target = np.array(vec, dtype=np.float32)
        retino = retinotopic_encode(sxy, centers, sigma)[0]
        data.append((inp_raw[:,2:6], np.tile(retino,(inp_raw.shape[0],1)), ramp, target, disp_target, fo, grounded))
    return data
vd = build(ds_val); ad = build(ds_train)
def stk(dl):
    return (torch.tensor(np.stack([d[0] for d in dl])), torch.tensor(np.stack([d[1] for d in dl])),
            torch.tensor(np.stack([d[2] for d in dl])), torch.tensor(np.stack([d[3] for d in dl])),
            torch.tensor(np.stack([d[4] for d in dl])), torch.tensor([d[5] for d in dl]),
            torch.tensor([d[6] for d in dl]))
td,tr,tramp,tt,tdisp,tf,_ = stk(vd)
ad2,ar,aramp,at,adisp,af,ag = stk(ad)

opt = torch.optim.AdamW(trainable, lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20000, eta_min=1e-5)
bl = float('inf'); pc = 0; t0 = time.time(); ramp_warmup = 2000
for ep in range(20000):
    model.train()
    tp, dp, rp, _ = model(td, tr, tf)
    lb = ((tp-tt)**2).mean(); ld = ((dp-tdisp)**2).mean(); lr_ = compute_ramp_loss(rp, tramp)
    rl = 30.0 if ep < ramp_warmup else 3.0
    loss = lb + 1.0*ld + rl*lr_
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sch.step()
    if lb.item() < bl: bl=lb.item(); pc=0; torch.save(model.state_dict(), f'{out_dir}/model.pt')
    else: pc += 1
    if ep % 200 == 0:
        with torch.no_grad():
            tp2, dp2, rp2, _ = model(ad2, ar, af)
            bg = ((tp2[ag==1]-at[ag==1])**2).mean().item()
            bh = ((tp2[ag==0]-at[ag==0])**2).mean().item()
        print(f'  ep {ep:5d}  loss={loss.item():.6f}  behav={bg:.4f}/{bh:.4f}  lr={opt.param_groups[0][\"lr\"]:.1e}  ({time.time()-t0:.0f}s)', flush=True)
    if pc >= 5000: print(f'  early stop ep {ep}', flush=True); break
model.load_state_dict(torch.load(f'{out_dir}/model.pt', weights_only=True)); model.eval()
with torch.no_grad():
    tp, dp, rp, _ = model(ad2, ar, af)
    bg = ((tp[ag==1]-at[ag==1])**2).mean().item(); bh = ((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded: behav={bg:.6f}'); print(f'  holdout: behav={bh:.6f}'); print(f'  gap: {bh-bg:.6f}')
"

fi

echo "Done task $TID"

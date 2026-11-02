#!/bin/bash
#SBATCH --job-name=train_sweep
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-74
#SBATCH --time=04:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Fig. 5g sweep: alpha (input entanglement) x beta (output alignment)
# alpha·Start -> embedding + RNN; the RNN also outputs start_mix (start_mix_head)
# readout_start = alpha·start_mix + (1-alpha)·start_2d
# alpha: [0, 0.25, 0.5, 0.75, 1.0]; beta: [linear, 4, 1, 0.5, 0.125]; 25 conditions x 3 seeds = 75 tasks
# -> rnn/models/sweep/alpha{a}_beta{b}/s{0,1,2}

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

COND_IDX=$((TID / 3))
SEED=$((TID % 3))

ALPHA_IDX=$((COND_IDX / 5))
BETA_IDX=$((COND_IDX % 5))

ALPHAS=(0.0 0.25 0.5 0.75 1.0)
BETAS=("linear" "4.0" "1.0" "0.5" "0.125")

ALPHA=${ALPHAS[$ALPHA_IDX]}
BETA=${BETAS[$BETA_IDX]}

OUT_DIR="$MODELS/sweep/alpha${ALPHA}_beta${BETA}/s${SEED}"
mkdir -p "$OUT_DIR"

echo "alpha=${ALPHA} beta=${BETA} seed=${SEED} (task ${TID}, cond ${COND_IDX})"

# Stage 1: pretrain integrator
$PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN
from src.rnn.training.train_integrator_stage1 import generate_batch

out_dir = '${OUT_DIR}'
torch.manual_seed(${SEED}); np.random.seed(${SEED})
dt=1/30; tau=0.1; alpha_rnn=dt/tau; H=32
rnn = FullRankRNN(input_size=1, hidden_size=H, alpha=alpha_rnn)
readout = nn.Linear(H, 1)
params = list(rnn.parameters()) + list(readout.parameters())
opt = torch.optim.AdamW(params, lr=3e-3)
bl=float('inf'); t0=time.time()
for ep in range(10000):
    inp,tgt=generate_batch(256,dt=dt); h=rnn(inp); rp=readout(h)
    valid=~torch.isnan(tgt); loss=((rp[valid]-tgt[valid])**2).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if loss.item()<bl: bl=loss.item(); torch.save({'rnn':rnn.state_dict(),'readout':readout.state_dict()},f'{out_dir}/model_stage1.pt')
    if ep%500==0: print(f'  s1 epoch {ep:5d}  loss={loss.item():.6f}  best={bl:.6f}  ({time.time()-t0:.0f}s)',flush=True)
    if bl<1e-3: print(f'  s1 converged at epoch {ep}',flush=True); break
print(f'  s1 final best: {bl:.6f}')
"

# Stage 2
$PY -c "
import sys, os, json, time, math
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN
from src.rnn.training.train_modular_readout import (EmbeddingMLP53, DisplacementMLP,
    make_retinotopic_centers, retinotopic_encode, compute_ramp_loss)
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2

out_dir = '${OUT_DIR}'
alpha_input = ${ALPHA}
beta_str = '${BETA}'
torch.manual_seed(${SEED}); np.random.seed(${SEED})
dt=1/30; tau=0.1; alpha_rnn=dt/tau; H=32; grid_size=7; n_retino=49; sigma=1.0
centers = make_retinotopic_centers(grid_size)

s1 = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)

embed = EmbeddingMLP53(hidden_size=32, flatness=1.0, n_layers=2)

rnn = FullRankRNN(input_size=4, hidden_size=H, alpha=alpha_rnn)
rnn.W_hh.weight.data = s1['rnn']['W_hh.weight']
rnn.bias.data = s1['rnn']['bias']
for p in rnn.parameters():
    p.requires_grad = False

ramp_readout = nn.Linear(H, 1)
ramp_readout.weight.data = s1['readout']['weight']
ramp_readout.bias.data = s1['readout']['bias']

disp_mlp = DisplacementMLP(H, 32, output_dim=2)

# w_fef: full start(49D) -> start_2d(2D)
w_fef = nn.Linear(n_retino, 2)

# start_mix_head: h(32D) -> start_mix(2D)
start_mix_head = nn.Linear(H, 2)

# CombineMLP: [readout_start(2D); disp(2D)] -> 2D target
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

trainable = (list(embed.parameters()) + list(ramp_readout.parameters()) +
             list(disp_mlp.parameters()) + list(w_fef.parameters()) +
             list(start_mix_head.parameters()) + list(combine.parameters()))
nt = sum(p.numel() for p in trainable)
nf = sum(p.numel() for p in rnn.parameters())
print(f'  trainable: {nt}, frozen: {nf}, alpha={alpha_input}, beta={beta_str}')

config = {'version': 'sweep','dt':dt,'tau':tau,'hidden_size':H,
          'grid_size':grid_size,'n_retino':n_retino,'sigma':sigma,
          'training_set':'monkey','pretrained':True,'frozen_integrator':True,
          'ramp_lambda':30.0,'disp_lambda':1.0,
          'alpha':alpha_input,'beta':beta_str,
          'embed_flatness':1.0,'embed_hidden':32,'embed_layers':2,
          'training':{'lr':0.001}}
with open(f'{out_dir}/config.json','w') as f: json.dump(config,f,indent=2)

task = DMFCTaskV2(delay=0.2,dt=dt,vector_lengths=[1,2,3,4,5,6],training_set='monkey',scale=1,tau=tau)
ball_speed=task.BALL_SPEED; ds_train,ds_val=task.get_datasets()

def build(ds):
    data=[]
    for i in range(len(ds)):
        inp,tgt,ramp,grounded=ds[i]; cond=ds.condition_labels[i]
        sxy=np.array(cond[0],dtype=np.float32).reshape(1,2); vec=np.array(cond[1])
        fo=int(np.round((task.FLOW_ONSET_TIME+np.linalg.norm(vec)/ball_speed)/dt))
        t=np.array(cond[2],dtype=np.float32); r=retinotopic_encode(sxy,centers,sigma)[0]
        disp_tgt=np.array(vec,dtype=np.float32)
        data.append((inp[:,2:6], np.tile(r,(inp.shape[0],1)), t, fo, grounded, ramp, disp_tgt))
    return data

vd=build(ds_val); ad=build(ds_train)

def stk(dl):
    return (torch.tensor(np.stack([d[0] for d in dl])),
            torch.tensor(np.stack([d[1] for d in dl])),
            torch.tensor(np.stack([d[2] for d in dl])),
            torch.tensor([d[3] for d in dl]),
            torch.tensor([d[4] for d in dl]),
            torch.tensor(np.stack([d[5] for d in dl])),
            torch.tensor(np.stack([d[6] for d in dl])))

td,tr,tt,tf,_,tramp,tdisp = stk(vd)
ad2,ar,at,af,ag,aramp,adisp = stk(ad)

def forward(flow, start, foff):
    B,T,_=flow.shape
    # Embedding gets flow + alpha*start
    start_to_embed = alpha_input * start
    combined = torch.cat([flow, start_to_embed], dim=2)  # (B,T,53)
    embedded = embed(combined.reshape(B*T,53)).reshape(B,T,53)
    emb_flow = embedded[:,:,:4]
    # Frozen RNN
    h_int = rnn(emb_flow)
    ramp_pred = ramp_readout(h_int)
    h_snap = torch.stack([h_int[i,foff[i]] for i in range(B)])
    # Displacement from h (strict 2D bottleneck, supervised by disp_loss)
    disp = disp_mlp(h_snap)
    # start_mix from h (RNN's version of start, noisy/degraded)
    start_mix = start_mix_head(h_snap)
    # Clean start_2d from w_fef (full start input, no alpha scaling)
    s_snap = torch.stack([start[i,foff[i]] for i in range(B)])
    start_2d = w_fef(s_snap)
    # Blend: alpha*start_mix + (1-alpha)*start_2d
    readout_start = alpha_input * start_mix + (1.0 - alpha_input) * start_2d
    # Combine [readout_start; disp] -> target
    target = combine(torch.cat([readout_start, disp], dim=1))
    return target, disp, ramp_pred, {'h_integrator': h_int}

opt = torch.optim.AdamW(trainable, lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=500, factor=0.5, min_lr=1e-5)
bl=float('inf'); pc=0; t0=time.time(); ramp_lambda=30.0; disp_lambda=1.0

for ep in range(20000):
    embed.train(); disp_mlp.train()
    tp, dp, rp, _ = forward(td, tr, tf)
    loss_behav = ((tp - tt)**2).mean()
    loss_disp = ((dp - tdisp)**2).mean()
    loss_ramp = compute_ramp_loss(rp, tramp)
    loss = loss_behav + disp_lambda * loss_disp + ramp_lambda * loss_ramp
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    if loss.item() < bl:
        bl = loss.item(); pc = 0
        torch.save({'embed': embed.state_dict(), 'rnn': rnn.state_dict(),
                     'ramp_readout': ramp_readout.state_dict(),
                     'disp_mlp': disp_mlp.state_dict(),
                     'w_fef': w_fef.state_dict(),
                     'start_mix_head': start_mix_head.state_dict(),
                     'combine': combine.state_dict()}, f'{out_dir}/model.pt')
    else:
        pc += 1
    if ep%200==0:
        embed.eval(); disp_mlp.eval()
        with torch.no_grad():
            tp2, _, _, _ = forward(ad2, ar, af)
            bg=((tp2[ag==1]-at[ag==1])**2).mean().item()
            bh=((tp2[ag==0]-at[ag==0])**2).mean().item()
        print(f'  epoch {ep:5d}  loss={loss.item():.6f}  behav={loss_behav.item():.6f}  disp={loss_disp.item():.6f}  ramp={loss_ramp.item():.6f}  gen={bg:.4f}/{bh:.4f}  lr={opt.param_groups[0][\"lr\"]:.1e}  ({time.time()-t0:.0f}s)',flush=True)
    sch.step(loss.item())
    if pc>=2000: print(f'  early stop at epoch {ep}',flush=True); break

ckpt = torch.load(f'{out_dir}/model.pt', weights_only=True)
embed.load_state_dict(ckpt['embed']); disp_mlp.load_state_dict(ckpt['disp_mlp'])
w_fef.load_state_dict(ckpt['w_fef']); start_mix_head.load_state_dict(ckpt['start_mix_head'])
combine.load_state_dict(ckpt['combine'])
embed.eval(); disp_mlp.eval()
with torch.no_grad():
    tp,_,_,_ = forward(ad2, ar, af)
    bg=((tp[ag==1]-at[ag==1])**2).mean().item()
    bh=((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded: behav={bg:.6f}'); print(f'  holdout:  behav={bh:.6f}'); print(f'  behav gap: {bh-bg:.6f}')
"

echo "Done task ${TID} (alpha=${ALPHA} beta=${BETA} seed=${SEED})"

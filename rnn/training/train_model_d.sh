#!/bin/bash
#SBATCH --job-name=train_model_d
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-9
#SBATCH --time=04:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Model d (Fig. 4d): EmbeddingMLP53(flatness 1) + pretrained + frozen full-rank RNN(32) + ReadoutMLP(6 x 128, 0.125)
# = model c + a nonlinear input embedding of (Flow, Start)
# Stage 1: pretrain the RNN on a 1D ramp; stage 2: freeze it, train embedding + readout + ramp readout
# 10 seeds -> rnn/models/model_d/s{0..9}

REPO=/path/to/CT_MJ_2026                 # set to this repository
ROOT="$REPO/rnn/vendored"
MODELS="$REPO/rnn/models"
cd "$ROOT"

export LD_LIBRARY_PATH=$HOME/.conda/envs/rnn_cpu/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=.

PY=$HOME/.conda/envs/rnn_cpu/bin/python
SEED=$SLURM_ARRAY_TASK_ID

OUT_DIR="$MODELS/model_d/s${SEED}"
mkdir -p "$OUT_DIR"

echo "model_d seed=${SEED}"

# Stage 1: pretrain integrator
$PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN
from src.rnn.training.train_integrator_stage1 import generate_batch

out_dir = '${OUT_DIR}'
torch.manual_seed(${SEED}); np.random.seed(${SEED})
dt=1/30; tau=0.1; alpha=dt/tau; H=32
rnn = FullRankRNN(input_size=1, hidden_size=H, alpha=alpha)
readout = nn.Linear(H, 1)
params = list(rnn.parameters()) + list(readout.parameters())
opt = torch.optim.AdamW(params, lr=3e-3)
with open(f'{out_dir}/config_stage1.json','w') as f:
    json.dump({'stage':1,'rank':'full','dt':dt,'tau':tau,'hidden_size':H,'seed':${SEED}},f,indent=2)
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

# Stage 2: freeze integrator, train embedding + readout
$PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN, ReadoutMLP
from src.rnn.training.train_modular_readout import EmbeddingMLP53, make_retinotopic_centers, retinotopic_encode
from src.rnn.training.train_rank1_rnn_readout import compute_ramp_loss
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2

out_dir = '${OUT_DIR}'
torch.manual_seed(${SEED}); np.random.seed(${SEED})
dt=1/30; tau=0.1; alpha=dt/tau; H=32; grid_size=7; n_retino=49; sigma=1.0
centers = make_retinotopic_centers(grid_size)

# Load pretrained integrator
s1 = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)

# Build model components
embed = EmbeddingMLP53(hidden_size=32, flatness=1.0, n_layers=2)
rnn = FullRankRNN(input_size=4, hidden_size=H, alpha=alpha)
# Load recurrent weights from stage 1 (W_hh, bias; W_ih changes size 1→4)
rnn.W_hh.weight.data = s1['rnn']['W_hh.weight']
rnn.bias.data = s1['rnn']['bias']
# Freeze integrator
for p in rnn.parameters():
    p.requires_grad = False

ramp_readout = nn.Linear(H, 1)
ramp_readout.weight.data = s1['readout']['weight']
ramp_readout.bias.data = s1['readout']['bias']

readout_mlp = ReadoutMLP(H + n_retino, 128, n_layers=6, flatness=0.125)

# Trainable: embedding + readout_mlp + ramp_readout
trainable = list(embed.parameters()) + list(readout_mlp.parameters()) + list(ramp_readout.parameters())
nt = sum(p.numel() for p in trainable)
nf = sum(p.numel() for p in rnn.parameters())
print(f'  trainable: {nt}, frozen: {nf}')

config = {'version': 'embed_rnn_readout','dt':dt,'tau':tau,'hidden_size':H,'mlp_hidden':128,
          'mlp_layers':6,'mlp_flatness':0.125,'grid_size':grid_size,'n_retino':n_retino,
          'sigma':sigma,'training_set':'monkey','rank':'full',
          'pretrained':True,'frozen_integrator':True,'ramp_lambda':30.0,
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
        data.append((inp[:,2:6], np.tile(r,(inp.shape[0],1)), t, fo, grounded, ramp))
    return data

vd=build(ds_val); ad=build(ds_train)

def stk(dl):
    return (torch.tensor(np.stack([d[0] for d in dl])),
            torch.tensor(np.stack([d[1] for d in dl])),
            torch.tensor(np.stack([d[2] for d in dl])),
            torch.tensor([d[3] for d in dl]),
            torch.tensor([d[4] for d in dl]),
            torch.tensor(np.stack([d[5] for d in dl])))

td,tr,tt,tf,_,tramp = stk(vd)
ad2,ar,at,af,ag,aramp = stk(ad)

def forward(flow, start, foff):
    B,T,_=flow.shape
    combined = torch.cat([flow, start], dim=2)  # (B,T,53)
    embedded = embed(combined.reshape(B*T,53)).reshape(B,T,53)
    emb_flow = embedded[:,:,:4]
    emb_start = embedded[:,:,4:]
    h_int = rnn(emb_flow)
    ramp_pred = ramp_readout(h_int)
    h_snap = torch.stack([h_int[i,foff[i]] for i in range(B)])
    s_snap = torch.stack([emb_start[i,foff[i]] for i in range(B)])
    combined_out = torch.cat([h_snap, s_snap], dim=1)
    target = readout_mlp(combined_out)
    return target, ramp_pred, {'h_integrator': h_int}

opt = torch.optim.AdamW(trainable, lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=500, factor=0.5, min_lr=1e-5)
bl=float('inf'); pc=0; t0=time.time(); ramp_lambda=30.0

for ep in range(20000):
    embed.train(); readout_mlp.train()
    tp, rp, _ = forward(td, tr, tf)
    disp_loss = ((tp - tt)**2).mean()
    rl = compute_ramp_loss(rp, tramp)
    loss = disp_loss + ramp_lambda * rl
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    if loss.item() < bl:
        bl = loss.item(); pc = 0
        torch.save({'embed': embed.state_dict(), 'rnn': rnn.state_dict(),
                     'ramp_readout': ramp_readout.state_dict(),
                     'readout_mlp': readout_mlp.state_dict()}, f'{out_dir}/model.pt')
    else:
        pc += 1
    if ep%100==0:
        embed.eval(); readout_mlp.eval()
        with torch.no_grad():
            tp2, _, _ = forward(ad2, ar, af)
            bg=((tp2[ag==1]-at[ag==1])**2).mean().item()
            bh=((tp2[ag==0]-at[ag==0])**2).mean().item()
        print(f'  epoch {ep:5d}  loss={loss.item():.6f}  disp={disp_loss.item():.6f}  ramp={rl.item():.6f}  behav={bg:.4f}/{bh:.4f}  lr={opt.param_groups[0][\"lr\"]:.1e}  ({time.time()-t0:.0f}s)',flush=True)
    sch.step(loss.item())
    if pc>=2000: print(f'  early stop at epoch {ep}',flush=True); break

# Final eval
ckpt = torch.load(f'{out_dir}/model.pt', weights_only=True)
embed.load_state_dict(ckpt['embed']); readout_mlp.load_state_dict(ckpt['readout_mlp'])
embed.eval(); readout_mlp.eval()
with torch.no_grad():
    tp,_,_ = forward(ad2, ar, af)
    bg=((tp[ag==1]-at[ag==1])**2).mean().item()
    bh=((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded: behav={bg:.6f}'); print(f'  holdout:  behav={bh:.6f}'); print(f'  behav gap: {bh-bg:.6f}')
"

echo "Done seed ${SEED}"

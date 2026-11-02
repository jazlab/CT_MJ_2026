#!/bin/bash
#SBATCH --job-name=train_model_a_rank_sweep
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-39
#SBATCH --time=02:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Model a rank sweep (S4.1a): rank-N RNN + linear readout (no MLP), 10 seeds per rank
# Tasks 0-9: rank 1; 10-19: rank 2; 20-29: rank 4; 30-39: rank 16   -> rnn/models/model_a/rank{1,2,4,16}/s{0..9}

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
RANK_IDX=$((TID / 10))
SEED=$((TID % 10))
RANK=${RANKS[$RANK_IDX]}

OUT_DIR="$MODELS/model_a/rank${RANK}/s${SEED}"
mkdir -p "$OUT_DIR"

echo "model_a rank=${RANK} seed=${SEED}"
$PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rank_n import RankNIntegrator
from src.rnn.training.train_rnn_readout import V1Model
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode

out_dir = '${OUT_DIR}'
torch.manual_seed(${SEED}); np.random.seed(${SEED})
dt=1/30; tau=0.1; alpha=dt/tau; H=32; grid_size=7; n_retino=49; sigma=1.0
centers = make_retinotopic_centers(grid_size)
rnn = RankNIntegrator(input_size=4, hidden_size=H, rank=${RANK}, alpha=alpha, flatness=1.0)
readout = nn.Linear(H + n_retino, 2)
model = V1Model(rnn, readout)
trainable = list(model.parameters())
config = {'version': 'rnn_readout','dt':dt,'tau':tau,'hidden_size':H,'mlp_hidden':0,
          'mlp_layers':0,'mlp_flatness':1.0,'grid_size':grid_size,'n_retino':n_retino,
          'sigma':sigma,'training_set':'monkey','rank':${RANK},'training':{'lr':0.001}}
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
        data.append((inp[:,2:6],np.tile(r,(inp.shape[0],1)),t,fo,grounded))
    return data
vd=build(ds_val); ad=build(ds_train)
def stk(dl):
    return(torch.tensor(np.stack([d[0]for d in dl])),torch.tensor(np.stack([d[1]for d in dl])),
           torch.tensor(np.stack([d[2]for d in dl])),torch.tensor([d[3]for d in dl]),
           torch.tensor([d[4]for d in dl]))
td,tr,tt,tf,_=stk(vd); ad2,ar,at,af,ag=stk(ad)
opt=torch.optim.AdamW(trainable,lr=1e-3)
sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=500,factor=0.5,min_lr=1e-5)
bl=float('inf'); pc=0; t0=time.time()
for ep in range(20000):
    model.train(); tp,_=model(td,tr,tf); loss=((tp-tt)**2).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
    if loss.item()<bl: bl=loss.item();pc=0;torch.save(model.state_dict(),f'{out_dir}/model.pt')
    else: pc+=1
    if ep%100==0:
        with torch.no_grad():
            tp2,_=model(ad2,ar,af)
            bg=((tp2[ag==1]-at[ag==1])**2).mean().item();bh=((tp2[ag==0]-at[ag==0])**2).mean().item()
        print(f'  epoch {ep:5d}  loss={loss.item():.6f}  behav={bg:.4f}/{bh:.4f}  lr={opt.param_groups[0][\"lr\"]:.1e}  ({time.time()-t0:.0f}s)',flush=True)
    sch.step(loss.item())
    if pc>=2000: print(f'  early stop at epoch {ep}',flush=True); break
model.load_state_dict(torch.load(f'{out_dir}/model.pt',weights_only=True)); model.eval()
with torch.no_grad():
    tp,_=model(ad2,ar,af)
    bg=((tp[ag==1]-at[ag==1])**2).mean().item();bh=((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded: behav={bg:.6f}');print(f'  holdout:  behav={bh:.6f}');print(f'  behav gap: {bh-bg:.6f}')
"

echo "Done task $TID (rank=${RANK} seed=${SEED})"

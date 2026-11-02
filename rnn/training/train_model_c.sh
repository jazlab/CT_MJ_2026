#!/bin/bash
#SBATCH --job-name=train_model_c
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-9
#SBATCH --time=06:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Model c (Fig. 4c, S4.1b, S4.2b): full-rank RNN pretrained on a 1D ramp and frozen (Timing Module), then a deep
# FFN readout (6 x 128, flatness 0.125) trained with the ramp loss. 10 seeds -> rnn/models/model_c/fullrank/s{0..9}

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

run_stage1() {
    local OUT_DIR=$1
    local SEED=$2
    $PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN
from src.rnn.training.train_integrator_stage1 import generate_batch
out_dir='${OUT_DIR}'; seed=${SEED}
torch.manual_seed(seed); np.random.seed(seed)
dt=1/30; tau=0.1; alpha=dt/tau; H=32
rnn = FullRankRNN(input_size=1, hidden_size=H, alpha=alpha)
readout = nn.Linear(H, 1)
params = list(rnn.parameters()) + list(readout.parameters())
opt = torch.optim.AdamW(params, lr=3e-3)
bl = float('inf'); t0 = time.time()
with open(f'{out_dir}/config_stage1.json','w') as f:
    json.dump({'stage':1,'dt':dt,'tau':tau,'hidden_size':H},f,indent=2)
for ep in range(10000):
    inp,tgt = generate_batch(256, dt=dt)
    h = rnn(inp); rp = readout(h)
    valid = ~torch.isnan(tgt); loss = ((rp[valid]-tgt[valid])**2).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if loss.item() < bl:
        bl = loss.item()
        torch.save({'rnn':rnn.state_dict(),'readout':readout.state_dict()}, f'{out_dir}/model_stage1.pt')
    if ep%500==0: print(f'  s1 ep {ep:5d} loss={loss.item():.6f} best={bl:.6f} ({time.time()-t0:.0f}s)',flush=True)
    if bl<1e-3: print(f'  converged ep {ep}',flush=True); break
print(f'  s1 final: {bl:.6f}')
"
}

run_model_c() {
    local SEED=$1
    local OUT_DIR="$MODELS/model_c/fullrank/s${SEED}"
    mkdir -p "$OUT_DIR"
    echo "model_c fullrank seed=${SEED}"
    run_stage1 "$OUT_DIR" "$SEED"
    $PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN, ReadoutMLP
from src.rnn.training.train_rank1_rnn_readout import compute_ramp_loss
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode
out_dir='${OUT_DIR}'; seed=${SEED}
torch.manual_seed(seed); np.random.seed(seed)
dt=1/30; tau=0.1; alpha=dt/tau; H=32; grid_size=7; n_retino=49; sigma=1.0
centers = make_retinotopic_centers(grid_size)
s1 = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)
rnn = FullRankRNN(input_size=4, hidden_size=H, alpha=alpha)
rnn.W_hh.weight.data = s1['rnn']['W_hh.weight']
rnn.bias.data = s1['rnn']['bias']; rnn.W_ih.bias.data = s1['rnn']['W_ih.bias']
rnn.W_hh.weight.requires_grad = False; rnn.bias.requires_grad = False
ramp_readout = nn.Linear(H, 1)
ramp_readout.weight.data = s1['readout']['weight']; ramp_readout.bias.data = s1['readout']['bias']
for p in ramp_readout.parameters(): p.requires_grad = False
mlp = ReadoutMLP(H+n_retino, 128, n_layers=6, flatness=0.125)
class Model(nn.Module):
    def __init__(s, rnn, ramp, mlp):
        super().__init__(); s.rnn=rnn; s.ramp_readout=ramp; s.readout_mlp=mlp
    def forward(s, d, st, fo):
        B,T,_=d.shape; h=s.rnn(d); ramp=s.ramp_readout(h)
        hs=torch.stack([h[i,fo[i]] for i in range(B)])
        ss=torch.stack([st[i,fo[i]] for i in range(B)])
        return s.readout_mlp(torch.cat([hs,ss],dim=1)), ramp, {'h_integrator':h}
model = Model(rnn, ramp_readout, mlp)
trainable = [p for p in model.parameters() if p.requires_grad]
config = {'version': 'rnn_readout','dt':dt,'tau':tau,'hidden_size':H,'mlp_hidden':128,
          'mlp_layers':6,'mlp_flatness':0.125,'grid_size':grid_size,'n_retino':n_retino,
          'sigma':sigma,'training_set':'monkey','rnn_type':'full_rank',
          'pretrained':True,'frozen_integrator':True,'ramp_lambda':30.0,'training':{'lr':0.001}}
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
        data.append((inp[:,2:6],np.tile(r,(inp.shape[0],1)),ramp,t,fo,grounded))
    return data
vd=build(ds_val); ad=build(ds_train)
def stk(dl):
    return(torch.tensor(np.stack([d[0]for d in dl])),torch.tensor(np.stack([d[1]for d in dl])),
           torch.tensor(np.stack([d[2]for d in dl])),torch.tensor(np.stack([d[3]for d in dl])),
           torch.tensor([d[4]for d in dl]),torch.tensor([d[5]for d in dl]))
td,tr,tramp,tt,tf,_=stk(vd); ad2,ar,aramp,at,af,ag=stk(ad)
opt=torch.optim.AdamW(trainable,lr=1e-3)
sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=500,factor=0.5,min_lr=1e-5)
bl=float('inf');pc=0;t0=time.time()
for ep in range(20000):
    model.train(); tp,rp,_=model(td,tr,tf)
    lb=((tp-tt)**2).mean(); lr_=compute_ramp_loss(rp,tramp)
    loss=lb+30.0*lr_
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step()
    if loss.item()<bl: bl=loss.item();pc=0;torch.save(model.state_dict(),f'{out_dir}/model.pt')
    else: pc+=1
    if ep%100==0:
        with torch.no_grad():
            tp2,rp2,_=model(ad2,ar,af)
            bg=((tp2[ag==1]-at[ag==1])**2).mean().item();bh=((tp2[ag==0]-at[ag==0])**2).mean().item()
        print(f'  ep {ep:5d} loss={loss.item():.6f} behav={bg:.4f}/{bh:.4f} lr={opt.param_groups[0][\"lr\"]:.1e} ({time.time()-t0:.0f}s)',flush=True)
    sch.step(loss.item())
    if pc>=2000: print(f'  early stop ep {ep}',flush=True);break
model.load_state_dict(torch.load(f'{out_dir}/model.pt',weights_only=True));model.eval()
with torch.no_grad():
    tp,rp,_=model(ad2,ar,af)
    bg=((tp[ag==1]-at[ag==1])**2).mean().item();bh=((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded:{bg:.6f} holdout:{bh:.6f} gap:{bh-bg:.6f}')
"
}


run_model_c ${SEEDS[$TID]}

echo "Done task $TID"

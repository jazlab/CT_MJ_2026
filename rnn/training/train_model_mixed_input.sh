#!/bin/bash
#SBATCH --job-name=train_model_mixed_input
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-9
#SBATCH --time=04:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Mixed-input model (S4.3): no embedding; [Flow(4D); Start(49D)] = 53D -> pretrained + frozen full-rank RNN(32)
# -> h(32D) -> ReadoutMLP(6 x 128, flatness 0.125) -> 2D. Stage 1: pretrain the RNN on a 1D ramp; stage 2: freeze
# it, train the readout only. 10 seeds -> rnn/models/model_mixed_input/s{0..9}

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

OUT_DIR="$MODELS/model_mixed_input/s${SEED}"
mkdir -p "$OUT_DIR"

echo "model_mixed_input seed=${SEED}"

# Stage 1: pretrain integrator on 1D ramp
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

# Stage 2: freeze integrator (input_size=53), train readout (h→2)
$PY -c "
import sys, os, json, time
sys.path.insert(0, '.')
import torch, torch.nn as nn, numpy as np
from src.rnn.training.train_rnn_readout import FullRankRNN, ReadoutMLP
from src.rnn.training.train_modular_readout import make_retinotopic_centers, retinotopic_encode
from src.rnn.training.train_rank1_rnn_readout import compute_ramp_loss
from src.rnn.tasks.dmfc_task_v2 import DMFCTaskV2

out_dir = '${OUT_DIR}'
torch.manual_seed(${SEED}); np.random.seed(${SEED})
dt=1/30; tau=0.1; alpha=dt/tau; H=32; grid_size=7; n_retino=49; sigma=1.0
centers = make_retinotopic_centers(grid_size)

# Load pretrained integrator
s1 = torch.load(f'{out_dir}/model_stage1.pt', weights_only=True)

# Build RNN with input_size=53 (flow 4D + start 49D)
rnn = FullRankRNN(input_size=53, hidden_size=H, alpha=alpha)
# Load recurrent weights from stage 1 (W_hh, bias; W_ih changes size 1→53)
rnn.W_hh.weight.data = s1['rnn']['W_hh.weight']
rnn.bias.data = s1['rnn']['bias']
# Freeze integrator
for p in rnn.parameters():
    p.requires_grad = False

ramp_readout = nn.Linear(H, 1)
ramp_readout.weight.data = s1['readout']['weight']
ramp_readout.bias.data = s1['readout']['bias']

# Readout takes only h(32D), not [h;start]
readout_mlp = ReadoutMLP(H, 128, n_layers=6, flatness=0.125)

# Trainable: readout_mlp + ramp_readout + W_ih of RNN (input projection, unfrozen)
# Actually W_ih needs to be trainable since it maps 53D→32D (new vs stage1's 1D→32D)
rnn.W_ih.weight.requires_grad = True
rnn.W_ih.bias.requires_grad = True
trainable = list(readout_mlp.parameters()) + list(ramp_readout.parameters()) + [rnn.W_ih.weight, rnn.W_ih.bias]
nt = sum(p.numel() for p in trainable)
nf = sum(1 for p in rnn.parameters() if not p.requires_grad)
print(f'  trainable params: {nt}, frozen: W_hh + bias')

config = {'version': 'joint_input_rnn_readout','dt':dt,'tau':tau,'hidden_size':H,'mlp_hidden':128,
          'mlp_layers':6,'mlp_flatness':0.125,'grid_size':grid_size,'n_retino':n_retino,
          'sigma':sigma,'training_set':'monkey','rank':'full',
          'pretrained':True,'frozen_integrator':True,'ramp_lambda':30.0,
          'joint_input':True,'readout_input':'h_only',
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
        # Concatenate flow(4D) and start(49D) as joint input
        joint = np.concatenate([inp[:,2:6], np.tile(r,(inp.shape[0],1))], axis=1)  # (T,53)
        data.append((joint, t, fo, grounded, ramp))
    return data

vd=build(ds_val); ad=build(ds_train)

def stk(dl):
    return (torch.tensor(np.stack([d[0] for d in dl])),
            torch.tensor(np.stack([d[1] for d in dl])),
            torch.tensor([d[2] for d in dl]),
            torch.tensor([d[3] for d in dl]),
            torch.tensor(np.stack([d[4] for d in dl])))

td,tt,tf,_,tramp = stk(vd)
ad2,at,af,ag,aramp = stk(ad)

def forward(joint_input, foff):
    B = joint_input.shape[0]
    h_int = rnn(joint_input)
    ramp_pred = ramp_readout(h_int)
    h_snap = torch.stack([h_int[i,foff[i]] for i in range(B)])
    target = readout_mlp(h_snap)
    return target, ramp_pred, {'h_integrator': h_int}

opt = torch.optim.AdamW(trainable, lr=1e-3)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=500, factor=0.5, min_lr=1e-5)
bl=float('inf'); pc=0; t0=time.time(); ramp_lambda=30.0

for ep in range(20000):
    readout_mlp.train()
    tp, rp, _ = forward(td, tf)
    disp_loss = ((tp - tt)**2).mean()
    rl = compute_ramp_loss(rp, tramp)
    loss = disp_loss + ramp_lambda * rl
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    if loss.item() < bl:
        bl = loss.item(); pc = 0
        torch.save({'rnn': rnn.state_dict(), 'ramp_readout': ramp_readout.state_dict(),
                     'readout_mlp': readout_mlp.state_dict()}, f'{out_dir}/model.pt')
    else:
        pc += 1
    if ep%100==0:
        readout_mlp.eval()
        with torch.no_grad():
            tp2, _, _ = forward(ad2, af)
            bg=((tp2[ag==1]-at[ag==1])**2).mean().item()
            bh=((tp2[ag==0]-at[ag==0])**2).mean().item()
        print(f'  epoch {ep:5d}  loss={loss.item():.6f}  disp={disp_loss.item():.6f}  ramp={rl.item():.6f}  behav={bg:.4f}/{bh:.4f}  lr={opt.param_groups[0][\"lr\"]:.1e}  ({time.time()-t0:.0f}s)',flush=True)
    sch.step(loss.item())
    if pc>=2000: print(f'  early stop at epoch {ep}',flush=True); break

# Final eval
ckpt = torch.load(f'{out_dir}/model.pt', weights_only=True)
rnn.load_state_dict(ckpt['rnn']); readout_mlp.load_state_dict(ckpt['readout_mlp'])
readout_mlp.eval()
with torch.no_grad():
    tp,_,_ = forward(ad2, af)
    bg=((tp[ag==1]-at[ag==1])**2).mean().item()
    bh=((tp[ag==0]-at[ag==0])**2).mean().item()
    print(f'  grounded: behav={bg:.6f}'); print(f'  holdout:  behav={bh:.6f}'); print(f'  behav gap: {bh-bg:.6f}')
"

echo "Done seed ${SEED}"

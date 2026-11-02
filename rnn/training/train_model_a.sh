#!/bin/bash
#SBATCH --job-name=train_model_a
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --array=0-19
#SBATCH --time=02:00:00
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Model a (Fig. 4a, S4.1a): Flow -> full-rank RNN(32) -> linear readout([h; Start]); trained end to end, 20 seeds
# -> rnn/models/model_a/fullrank/s{0..19}

REPO=/path/to/CT_MJ_2026                 # set to this repository
ROOT="$REPO/rnn/vendored"
MODELS="$REPO/rnn/models"
cd "$ROOT"

export LD_LIBRARY_PATH=$HOME/.conda/envs/rnn_cpu/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=.

PY=$HOME/.conda/envs/rnn_cpu/bin/python
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
TID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$TID]}

OUT_DIR="$MODELS/model_a/fullrank/s${SEED}"
mkdir -p "$OUT_DIR"

echo "model_a fullrank seed=${SEED}"
$PY src/rnn/training/train_rnn_readout.py \
    --out_dir "$OUT_DIR" --lr 1e-3 --seed $SEED --sigma 1.0 \
    --training_set monkey --mlp_layers 0

echo "Done task $TID"

#!/bin/bash
#SBATCH --job-name=lobs5_es
#SBATCH --output=logs_es/lobs5_es_%j.out
#SBATCH --error=logs_es/lobs5_es_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=12:00:00

# Configuration (override via environment)
TOKEN_MODE=${TOKEN_MODE:-22}
BACKGROUND_MODE=${BACKGROUND_MODE:-historical_replay}
REPLAY_DATA_PATH=${REPLAY_DATA_PATH:-/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-checkpoints/lobs5_xxx}

mkdir -p logs_es

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

python -u -B -m es_lobs5.run_es_train \
    --lobs5_checkpoint="${CHECKPOINT_PATH}" \
    --noiser=eggroll \
    --sigma=0.01 \
    --lr=0.001 \
    --lora_rank=4 \
    --n_threads=128 \
    --n_epochs=1000 \
    --n_steps=100 \
    --token_mode="${TOKEN_MODE}" \
    --background_mode="${BACKGROUND_MODE}" \
    $([ "$BACKGROUND_MODE" = "historical_replay" ] && echo "--replay_data_path=${REPLAY_DATA_PATH}") \
    --seed=42 \
    2>&1 | tee logs_es/training_${SLURM_JOB_ID}.log

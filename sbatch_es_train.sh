#!/bin/bash
#SBATCH --job-name=es-prod-${MODEL_PRESET:-custom}-24h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/es_train_%j.out
#SBATCH --error=logs/es_train_%j.err
#SBATCH --partition=workq

# ==============================================================================
# ES-LOBS5 统一 SBATCH 模板 (Unified SBATCH Template)
# ==============================================================================
#
# 使用方式 (Usage):
#
#   # 使用预设配置
#   MODEL_PRESET=L1 sbatch sbatch_es_train.sh
#   MODEL_PRESET=D5 sbatch sbatch_es_train.sh
#   MODEL_PRESET=K3 sbatch sbatch_es_train.sh
#
#   # 自定义参数
#   MODEL_PRESET=L1 TASK_SIZE=20 sbatch sbatch_es_train.sh
#
#   # 完全自定义
#   N_STEPS=50 TASK_SIZE=100 NOISER=eggroll sbatch sbatch_es_train.sh
#
# 可用预设 (Available Presets):
#   D5, G5     - n_steps=10, bg_msgs=100, task_size=500, noiser=eggroll
#   H5         - n_steps=100, bg_msgs=10, task_size=500, noiser=eggroll
#   I5         - n_steps=100, bg_msgs=10, task_size=100, noiser=eggroll
#   J1-J4      - n_steps=10, bg_msgs=10, task_size=500, noiser=eggroll (different file_idx)
#   K1-K4      - n_steps=10, bg_msgs=10, task_size=100, noiser=eggroll (different file_idx)
#   L1-L4      - n_steps=10, bg_msgs=10, task_size=10, noiser=eggrollbs (different file_idx)
#
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
echo "============================================================"
echo "ES-LOBS5 Production Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Preset: ${MODEL_PRESET:-custom}"
echo "Start: $(date)"
echo "============================================================"

mkdir -p logs
mkdir -p checkpoints

# Activate conda
source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Working directory
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Python path
export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv

# ------------------------------------------------------------------------------
# Build Command
# ------------------------------------------------------------------------------
CMD="python es_train_main.py"

# Add preset if specified
if [[ -n "$MODEL_PRESET" ]]; then
    CMD="$CMD --preset $MODEL_PRESET"
fi

# Add optional overrides from environment variables
[[ -n "$NOISER" ]] && CMD="$CMD --noiser $NOISER"
[[ -n "$N_STEPS" ]] && CMD="$CMD --n_steps $N_STEPS"
[[ -n "$TASK_SIZE" ]] && CMD="$CMD --task_size $TASK_SIZE"
[[ -n "$BG_MSGS" ]] && CMD="$CMD --background_msgs_per_step $BG_MSGS"
[[ -n "$FILE_IDX" ]] && CMD="$CMD --file_idx $FILE_IDX"
[[ -n "$N_PERTURBATIONS" ]] && CMD="$CMD --n_perturbations $N_PERTURBATIONS"
[[ -n "$N_EPOCHS" ]] && CMD="$CMD --n_epochs $N_EPOCHS"
[[ -n "$SIGMA" ]] && CMD="$CMD --sigma $SIGMA"
[[ -n "$LR" ]] && CMD="$CMD --lr $LR"
[[ -n "$GROUP_SIZE" ]] && CMD="$CMD --group_size $GROUP_SIZE"
[[ -n "$SEED" ]] && CMD="$CMD --seed $SEED"
[[ -n "$WANDB_NAME" ]] && CMD="$CMD --wandb_name $WANDB_NAME"
[[ -n "$DATA_PATH" ]] && CMD="$CMD --data_path $DATA_PATH"
[[ -n "$CHECKPOINT_DIR" ]] && CMD="$CMD --checkpoint_dir $CHECKPOINT_DIR"
[[ "$NO_WANDB" == "1" ]] && CMD="$CMD --no_wandb"

echo ""
echo "Running: $CMD"
echo ""

# ------------------------------------------------------------------------------
# Run Training
# ------------------------------------------------------------------------------
$CMD

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Training Complete!"
echo "End: $(date)"
echo "============================================================"

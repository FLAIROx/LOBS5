#!/bin/bash
# =============================================================================
# ES-LOBS5 Benchmark - 批量提交脚本
# 每个配置单独提交一个任务，并行执行
# =============================================================================

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
mkdir -p logs_benchmark

# 通用配置
CHECKPOINT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/logical-serenity-19_4dhsl6me"
REPLAY_DATA="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022"
N_EPOCHS=5
LORA_RANK=4

# =============================================================================
# J1: 单 GPU, 100 steps
# =============================================================================
submit_j1() {
    echo "=== J1: 单 GPU, 100 steps ==="
    for N in 8 16 32 64 128 256 512 1024; do
        sbatch --job-name="J1_${N}t" \
               --output="logs_benchmark/J1_${N}t_%j.out" \
               --error="logs_benchmark/J1_${N}t_%j.err" \
               --gres=gpu:1 --mem=0 --time=2:00:00 \
               --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5 && \
                       export PYTHONPATH=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:\$PYTHONPATH && \
                       cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 && \
                       python -u -B -m es_lobs5.run_es_train \
                           --lobs5_checkpoint=${CHECKPOINT} \
                           --noiser=eggroll --sigma=0.01 --lr=0.001 \
                           --lora_rank=${LORA_RANK} \
                           --n_threads=${N} --n_epochs=${N_EPOCHS} --n_steps=100 \
                           --token_mode=24 \
                           --background_mode=historical_replay \
                           --replay_data_path=${REPLAY_DATA} \
                           --wandb_project=es_benchmark_J1 \
                           --seed=42"
        echo "  Submitted J1: n_threads=${N}, steps=100, 1GPU"
    done
}

# =============================================================================
# J2: 单 GPU, 10 steps
# =============================================================================
submit_j2() {
    echo "=== J2: 单 GPU, 10 steps ==="
    for N in 64 256 512 1024 2048 4096; do
        sbatch --job-name="J2_${N}t" \
               --output="logs_benchmark/J2_${N}t_%j.out" \
               --error="logs_benchmark/J2_${N}t_%j.err" \
               --gres=gpu:1 --mem=0 --time=2:00:00 \
               --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5 && \
                       export PYTHONPATH=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:\$PYTHONPATH && \
                       cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 && \
                       python -u -B -m es_lobs5.run_es_train \
                           --lobs5_checkpoint=${CHECKPOINT} \
                           --noiser=eggroll --sigma=0.01 --lr=0.001 \
                           --lora_rank=${LORA_RANK} \
                           --n_threads=${N} --n_epochs=${N_EPOCHS} --n_steps=10 \
                           --token_mode=24 \
                           --background_mode=historical_replay \
                           --replay_data_path=${REPLAY_DATA} \
                           --wandb_project=es_benchmark_J2 \
                           --seed=42"
        echo "  Submitted J2: n_threads=${N}, steps=10, 1GPU"
    done
}

# =============================================================================
# J3: 4 GPU, 100 steps
# =============================================================================
submit_j3() {
    echo "=== J3: 4 GPU, 100 steps ==="
    for N in 32 128 512 1024 2048 4096; do
        sbatch --job-name="J3_${N}t" \
               --output="logs_benchmark/J3_${N}t_%j.out" \
               --error="logs_benchmark/J3_${N}t_%j.err" \
               --gres=gpu:4 --mem=0 --time=2:00:00 \
               --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5 && \
                       export PYTHONPATH=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:\$PYTHONPATH && \
                       cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 && \
                       python -u -B -m es_lobs5.run_es_train \
                           --lobs5_checkpoint=${CHECKPOINT} \
                           --noiser=eggroll --sigma=0.01 --lr=0.001 \
                           --lora_rank=${LORA_RANK} \
                           --n_threads=${N} --n_epochs=${N_EPOCHS} --n_steps=100 \
                           --token_mode=24 \
                           --background_mode=historical_replay \
                           --replay_data_path=${REPLAY_DATA} \
                           --wandb_project=es_benchmark_J3 \
                           --seed=42"
        echo "  Submitted J3: n_threads=${N}, steps=100, 4GPU"
    done
}

# =============================================================================
# J4: 4 GPU, 10 steps
# =============================================================================
submit_j4() {
    echo "=== J4: 4 GPU, 10 steps ==="
    for N in 256 1024 4096 8192 16384; do
        sbatch --job-name="J4_${N}t" \
               --output="logs_benchmark/J4_${N}t_%j.out" \
               --error="logs_benchmark/J4_${N}t_%j.err" \
               --gres=gpu:4 --mem=0 --time=2:00:00 \
               --wrap="source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5 && \
                       export PYTHONPATH=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:\$PYTHONPATH && \
                       cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 && \
                       python -u -B -m es_lobs5.run_es_train \
                           --lobs5_checkpoint=${CHECKPOINT} \
                           --noiser=eggroll --sigma=0.01 --lr=0.001 \
                           --lora_rank=${LORA_RANK} \
                           --n_threads=${N} --n_epochs=${N_EPOCHS} --n_steps=10 \
                           --token_mode=24 \
                           --background_mode=historical_replay \
                           --replay_data_path=${REPLAY_DATA} \
                           --wandb_project=es_benchmark_J4 \
                           --seed=42"
        echo "  Submitted J4: n_threads=${N}, steps=10, 4GPU"
    done
}

# =============================================================================
# 使用方法
# =============================================================================
case ${1:-all} in
    j1) submit_j1 ;;
    j2) submit_j2 ;;
    j3) submit_j3 ;;
    j4) submit_j4 ;;
    all)
        submit_j1
        submit_j2
        submit_j3
        submit_j4
        ;;
    *)
        echo "Usage: $0 [j1|j2|j3|j4|all]"
        exit 1
        ;;
esac

echo ""
echo "Done! Check jobs with: squeue -u \$USER"

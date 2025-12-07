#!/bin/bash
#SBATCH --job-name=data_analysis
#SBATCH --output=logs/data_analysis_%j.out
#SBATCH --error=logs/data_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00


# Create logs directory if not exists
mkdir -p logs

echo "=========================================="
echo "Data Distribution Analysis Job"
echo "=========================================="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Time: $(date)"
echo "=========================================="

# Source conda
source ~/miniforge3/etc/profile.d/conda.sh

# Activate environment
conda activate lobs5

# Run analysis script
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

python -u compute_normalization_stats.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

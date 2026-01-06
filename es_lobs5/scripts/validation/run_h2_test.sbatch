#!/bin/bash
#SBATCH --job-name=es_h2_test
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --output=logs_validation/h2_test_%j.out
#SBATCH --error=logs_validation/h2_test_%j.err

echo "=============================================="
echo "H2 Multi-GPU Validation Test"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/bin/activate es-lob
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
mkdir -p logs_validation

export PYTHONPATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/JaxMARL-HFT:$PYTHONPATH"
PYTHON="/lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/envs/es-lob/bin/python -u"

$PYTHON es_lobs5/scripts/validation/h2_multi_gpu_validation.py

echo "Exit status: $?"

#!/bin/bash
# Submit parameter counting jobs for all 8 configurations

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
mkdir -p logs_lobmax/param_counts

# All presets to measure
PRESETS=(
    "125M"
    "360M"
    "1.3B"
    "3B"
    "1.9B-A153M"
    "5.8B-A473M"
    "23B-A1.9B"
    "44B-A3.8B"
)

echo "Submitting 8 parameter counting jobs..."
echo ""

for preset in "${PRESETS[@]}"; do
    # Create a temporary batch script for this preset
    cat > /tmp/count_${preset}.batch << BATCHEOF
#!/bin/bash
#SBATCH --job-name=count_${preset}
#SBATCH --output=logs_lobmax/param_counts/count_${preset}_%j.out
#SBATCH --error=logs_lobmax/param_counts/count_${preset}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --exclude=nid[010696-010718],nid010152,nid010110,nid[011112-011115],nid011294,nid[010083-010086],nid[010561-010564],nid011108

cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobmax
module load cuda/12.6

export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusparse/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cufft/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nvjitlink/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cusolver/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:\$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH

export JAX_PLATFORMS="cuda"

# Large models need --use-eval-shape to avoid OOM
if [[ "${preset}" == "23B-A1.9B" || "${preset}" == "44B-A3.8B" ]]; then
    python -u -B count_single_model.py --preset ${preset} --use-eval-shape
else
    python -u -B count_single_model.py --preset ${preset}
fi
BATCHEOF

    # Submit the job
    jobid=$(sbatch /tmp/count_${preset}.batch | awk '{print $4}')
    echo "  ${preset}: Job $jobid submitted"
done

echo ""
echo "All jobs submitted! Check status with: squeue -u \$USER"
echo "Results will be in: logs_lobmax/param_counts/"

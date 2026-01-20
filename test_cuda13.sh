#!/bin/bash
#SBATCH --job-name=cuda13-test
#SBATCH --time=00:05:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --output=cuda13_test_%j.out
#SBATCH --error=cuda13_test_%j.err

echo "============================================"
echo "CUDA 13 Verification Test"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate the cuda13 environment
source ~/.bashrc
conda activate cuda13

echo "============================================"
echo "1. NVCC Version"
echo "============================================"
nvcc --version

echo ""
echo "============================================"
echo "2. NVIDIA-SMI (Driver & GPU Info)"
echo "============================================"
nvidia-smi

echo ""
echo "============================================"
echo "3. CUDA Environment Variables"
echo "============================================"
echo "CUDA_HOME: ${CUDA_HOME:-'not set'}"
echo "CUDA_PATH: ${CUDA_PATH:-'not set'}"
echo "PATH includes cuda: $(echo $PATH | tr ':' '\n' | grep -i cuda || echo 'no cuda in PATH')"

echo ""
echo "============================================"
echo "4. Quick CUDA Device Query (Python)"
echo "============================================"
python3 -c "
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader'],
                          capture_output=True, text=True)
    print('Driver info from nvidia-smi:')
    print(result.stdout)
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "============================================"
echo "Test Complete!"
echo "============================================"

#!/bin/bash
#SBATCH --job-name=cuda13-container
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --output=cuda13_container_%j.out
#SBATCH --error=cuda13_container_%j.err

echo "============================================"
echo "CUDA 13 Container Verification Test"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Container path
CONTAINER="/projects/s5e/quant/containers/cuda13-devel.sif"

echo "============================================"
echo "1. Host Driver Info"
echo "============================================"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo ""

echo "============================================"
echo "2. NVCC Version Inside Container"
echo "============================================"
singularity exec --nv ${CONTAINER} nvcc --version
echo ""

echo "============================================"
echo "3. nvidia-smi Inside Container"
echo "============================================"
singularity exec --nv ${CONTAINER} nvidia-smi
echo ""

echo "============================================"
echo "4. CUDA Runtime Version Test (Python)"
echo "============================================"
singularity exec --nv ${CONTAINER} python3 -c "
import ctypes
import os

# Try to load CUDA runtime
try:
    libcudart = ctypes.CDLL('libcudart.so')

    # Get CUDA runtime version
    version = ctypes.c_int()
    result = libcudart.cudaRuntimeGetVersion(ctypes.byref(version))
    if result == 0:
        major = version.value // 1000
        minor = (version.value % 1000) // 10
        print(f'CUDA Runtime Version: {major}.{minor}')
    else:
        print(f'cudaRuntimeGetVersion failed with error: {result}')

    # Get CUDA driver version
    driver_version = ctypes.c_int()
    result = libcudart.cudaDriverGetVersion(ctypes.byref(driver_version))
    if result == 0:
        major = driver_version.value // 1000
        minor = (driver_version.value % 1000) // 10
        print(f'CUDA Driver Version: {major}.{minor}')
    else:
        print(f'cudaDriverGetVersion failed with error: {result}')

except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "============================================"
echo "5. Simple CUDA Device Query"
echo "============================================"
singularity exec --nv ${CONTAINER} python3 -c "
import ctypes

try:
    libcudart = ctypes.CDLL('libcudart.so')

    # Get device count
    count = ctypes.c_int()
    result = libcudart.cudaGetDeviceCount(ctypes.byref(count))
    if result == 0:
        print(f'CUDA Device Count: {count.value}')
    else:
        print(f'cudaGetDeviceCount failed: {result}')

except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "============================================"
echo "Test Complete!"
echo "============================================"

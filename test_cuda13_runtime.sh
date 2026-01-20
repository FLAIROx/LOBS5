#!/bin/bash
#SBATCH --job-name=cuda13-runtime
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --output=cuda13_runtime_%j.out
#SBATCH --error=cuda13_runtime_%j.err

echo "============================================"
echo "CUDA 13 Runtime Verification Test"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

CONTAINER="/projects/s5e/quant/containers/cuda13-devel.sif"
WORKDIR="/tmp/cuda_test_$$"

# Create work directory
mkdir -p $WORKDIR

# Write a simple CUDA program
cat > $WORKDIR/test_cuda.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("CUDA Runtime Version: ");
        int runtimeVersion;
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("%d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
        printf("CUDA Driver Version: ");
        int driverVersion;
        cudaDriverGetVersion(&driverVersion);
        printf("%d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
        return 1;
    }

    printf("=== CUDA Runtime Test ===\n");
    printf("Number of CUDA devices: %d\n", deviceCount);

    // Get CUDA versions
    int runtimeVersion, driverVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);

    // Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    }

    // Simple vector addition test
    const int N = 1000;
    size_t size = N * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }

    printf("\n=== Kernel Execution Test ===\n");
    if (success) {
        printf("Vector addition: PASSED\n");
    } else {
        printf("Vector addition: FAILED\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n=== CUDA 13 Runtime Test Complete! ===\n");
    return 0;
}
EOF

echo "============================================"
echo "1. Compiling CUDA program with nvcc 13.0"
echo "============================================"
# Set TMPDIR to avoid permission issues with nvcc temp files
export TMPDIR=$WORKDIR
singularity exec --nv -B $WORKDIR:$WORKDIR -B $WORKDIR:/tmp --env TMPDIR=$WORKDIR $CONTAINER nvcc -o $WORKDIR/test_cuda $WORKDIR/test_cuda.cu 2>&1
compile_result=$?
echo "Compile result: $compile_result"

if [ $compile_result -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "2. Running CUDA program"
    echo "============================================"
    singularity exec --nv -B $WORKDIR:$WORKDIR -B $WORKDIR:/tmp $CONTAINER $WORKDIR/test_cuda 2>&1
else
    echo "Compilation failed!"
fi

# Cleanup
rm -rf $WORKDIR

echo ""
echo "============================================"
echo "Test Complete!"
echo "============================================"

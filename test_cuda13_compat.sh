#!/bin/bash
#SBATCH --job-name=cuda13-compat
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --output=cuda13_compat_%j.out
#SBATCH --error=cuda13_compat_%j.err

echo "============================================"
echo "CUDA 13 Forward Compatibility Test"
echo "============================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

CONTAINER="/projects/s5e/quant/containers/cuda13-devel.sif"
COMPAT_DIR="/projects/s5e/quant/cuda13-compat"
WORKDIR="/tmp/cuda_compat_test_$$"

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
        printf("Error Code: %d\n", (int)err);

        // Still try to get version info
        int runtimeVersion, driverVersion;
        cudaRuntimeGetVersion(&runtimeVersion);
        cudaDriverGetVersion(&driverVersion);
        printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
        printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
        return 1;
    }

    printf("=== CUDA 13 Forward Compatibility Test ===\n");
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

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            printf("Mismatch at index %d: expected %f, got %f\n", i, h_a[i] + h_b[i], h_c[i]);
            break;
        }
    }

    printf("\n=== Kernel Execution Test ===\n");
    if (success) {
        printf("Vector addition: PASSED ✓\n");
    } else {
        printf("Vector addition: FAILED ✗\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n=== CUDA 13 Forward Compatibility Test Complete! ===\n");
    return 0;
}
EOF

echo "============================================"
echo "1. Host Driver Info"
echo "============================================"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo ""

echo "============================================"
echo "2. Compat Libraries"
echo "============================================"
ls -la ${COMPAT_DIR}/
echo ""

echo "============================================"
echo "3. Compiling with nvcc 13.0"
echo "============================================"
export TMPDIR=$WORKDIR
singularity exec --nv \
    -B $WORKDIR:$WORKDIR \
    -B $WORKDIR:/tmp \
    --env TMPDIR=$WORKDIR \
    $CONTAINER nvcc -o $WORKDIR/test_cuda $WORKDIR/test_cuda.cu 2>&1
compile_result=$?
echo "Compile result: $compile_result"

if [ $compile_result -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "4. Running with Forward Compatibility"
    echo "============================================"
    echo "Using compat libs from: ${COMPAT_DIR}"

    # Run with LD_LIBRARY_PATH pointing to compat libs FIRST
    singularity exec --nv \
        -B $WORKDIR:$WORKDIR \
        -B ${COMPAT_DIR}:${COMPAT_DIR} \
        --env LD_LIBRARY_PATH="${COMPAT_DIR}:\$LD_LIBRARY_PATH" \
        $CONTAINER $WORKDIR/test_cuda 2>&1
    run_result=$?
    echo ""
    echo "Run result: $run_result"
else
    echo "Compilation failed!"
fi

# Cleanup
rm -rf $WORKDIR

echo ""
echo "============================================"
echo "Test Complete!"
echo "============================================"

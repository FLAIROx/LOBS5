# Base Conda Environment — Package Manifest

**Path**: `/projects/s5e/quant/miniforge3` (base env)
**Python**: 3.12.11
**Date**: 2026-02-24
**Verified**: Job 2451587 (2N/8GPU, CURTAIL_EPOCHS=50, BSZ=10) — training + eval OK

## Critical ML Packages

| Package | Version | Notes |
|---------|---------|-------|
| jax | 0.9.0.1 | Main compute framework |
| jaxlib | 0.9.0.1 | |
| jax-cuda12-pjrt | 0.9.0.1 | CUDA 12 PJRT plugin |
| jax-cuda12-plugin | 0.9.0.1 | |
| jax-triton | 0.3.0 | Patched for JAX 0.9 + Triton 3.4 compat |
| flax | 0.12.2 | Neural network library |
| optax | 0.2.6 | Optimizer library |
| orbax-checkpoint | 0.11.32 | Checkpointing |
| chex | 0.1.91 | JAX testing utilities |
| ml_dtypes | 0.5.3 | ML data types (bf16 etc.) |
| tensorstore | 0.1.80 | Tensor storage for checkpoints |
| triton | 3.4.0 | GPU kernel compiler |

## CUDA / GPU

| Package | Version | Notes |
|---------|---------|-------|
| nvidia-nccl-cu12 | 2.29.3 | NCCL (used via LD_LIBRARY_PATH override) |
| nvidia-cudnn-cu12 | 9.18.1.3 | |
| nvidia-cublas-cu12 | 12.9.1.4 | |
| nvidia-cuda-nvcc-cu12 | 12.9.86 | |
| nvidia-cuda-runtime-cu12 | 12.9.79 | |
| nvidia-cusolver-cu12 | 11.7.5.82 | |
| nvidia-cusparse-cu12 | 12.5.10.65 | |
| nvidia-cufft-cu12 | 11.4.1.4 | |
| nvidia-cuda-cupti-cu12 | 12.9.79 | |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | |
| nvidia-nvjitlink-cu12 | 12.9.86 | |
| nvidia-nvshmem-cu12 | 3.5.19 | |
| nvidia-cuda-cccl-cu12 | 12.9.27 | |

## Data & Logging

| Package | Version | Notes |
|---------|---------|-------|
| wandb | 0.21.3 | Weights & Biases |
| tensorboard | 2.20.0 | |
| tensorboard-plugin-profile | 2.15.0 | XLA profiling |
| xprof | 2.20.7 | |

## Scientific Computing

| Package | Version | Notes |
|---------|---------|-------|
| numpy | 2.3.3 | |
| scipy | 1.16.3 | |
| pandas | 2.3.2 | |
| scikit-learn | 1.8.0 | |
| matplotlib | 3.10.8 | |
| seaborn | 0.13.2 | |
| plotly | 6.5.1 | |
| statsmodels | 0.14.6 | |

## PyTorch Ecosystem (coexists with JAX)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.8.0+cu129 | |
| torchvision | 0.23.0 | |
| flash_attn | 2.8.3 | Flash Attention |
| deepspeed | 0.17.6 | |
| transformer_engine | 2.11.0+c188b533 | |
| transformers | 4.56.0 | HuggingFace |
| accelerate | 1.10.1 | |
| peft | 0.17.1 | Parameter-efficient fine-tuning |
| megatron-core | 0.12.2 | |
| timm | 1.0.22 | |
| datasets | 4.1.1 | HuggingFace datasets |

## Other Notable

| Package | Version | Notes |
|---------|---------|-------|
| ray | 2.49.2 | Distributed computing |
| tensorflow | 2.20.0 | |
| keras | 3.11.3 | |
| einops | 0.8.1 | Tensor operations |
| safetensors | 0.6.2 | Safe model serialization |
| polars | 1.34.0 | Fast dataframes |
| openai | 2.14.0 | |

## Migration Notes (lob → base)

Key differences from the previous `lob` conda env:

| Package | lob env | base env | Impact |
|---------|---------|----------|--------|
| jax/jaxlib | 0.6.1 | 0.9.0.1 | Major upgrade, no code changes needed |
| numpy | 1.26.4 | 2.3.3 | Major upgrade, compatible |
| flax | 0.11.2 | 0.12.2 | Minor upgrade |
| optax | 0.2.4 | 0.2.6 | Patch upgrade |
| nvidia-nccl-cu12 | 2.28.9 | 2.29.3 | Fixes ARM CAS hang bug |
| torch | 2.9.1 | 2.8.0+cu129 | lob was newer |
| triton | 3.5.1 | 3.4.0 | lob was newer |

The `lob` env was retired on 2026-02-24 due to Lustre project quota issues
(project 154201, 1MB limit, 5.3TB used). Training verified on base env with
Job 2451587 (2N/8GPU).

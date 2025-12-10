#!/bin/bash
#SBATCH --job-name=preproc2022
#SBATCH --output=logs/preproc_2022_%j.out
#SBATCH --error=logs/preproc_2022_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --contiguous
#SBATCH --exclude=nid[010696-010716],nid010718

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

export JAX_PLATFORMS=cpu

python /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/preproc_parallel.py \
    --data_dir /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2022/data/GOOG/raw/ \
    --save_dir /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2022/ \
    --skip_existing \
    --num_workers 32

echo "Preprocessing complete!"

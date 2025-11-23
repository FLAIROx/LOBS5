# python3 run_train.py \
#         --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
#         --blocks=16 --bsz=40 --d_model=1024 --dataset=lobster-prediction --merging=padded \
#         --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded' \
#         --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOGJAN2023_encoded' \
#         --data_mode='encoded' \
#          --clip_eigs=True --activation_fn=half_glu1 \
#         --dt_global=False --epochs=5 --jax_seed=42 --lr_factor=1 --n_layers=12 \
#         --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0003 --ssm_size_base=1024 \
#         --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
#         --use_book_data=True --use_simple_book=False --book_transform=True  \
#         --masking=none \
#         --num_devices=4 --n_data_workers=4 \
#         --debug_loading=False \
#         --enable_profiler=False \
#         --random_offsets_train=True \
#         --shuffle_train=True \
#         --debug_overfit=False \
#         --lr_patience=5 \
#         --USE_WANDB=True \
#         --wandb_project=lobs5-full-autoreg \
#         --wandb_entity=kang-oxford 
#         # --wandb_entity=kang-oxford 2>&1 | grep -v "sol_gpu_cost_model"
#         # --restore='/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/ruby-aardvark-62_98nov1i7' \
#         # --restore_step=37
#         #--restore='checkpoints/eager-shadow-750_af39bb9u/'
#         #5135
#         # --curtail_epochs=5135 \

# -u: unbuffered output for real-time logging
# -B don't write .pyc files
python3 -u -B run_train.py \
        --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
        --blocks=16 --bsz=64 --d_model=1024 --dataset=lobster-prediction --merging=padded \
        --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021' \
        --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2' \
        --data_mode='preproc' \
         --clip_eigs=True --activation_fn=half_glu1 \
        --dt_global=False --epochs=5 --jax_seed=42 --lr_factor=1 --n_layers=12 \
        --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0003 --ssm_size_base=1024 \
        --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
        --use_book_data=True --use_simple_book=False --book_transform=True  \
        --masking=none \
        --num_devices=4 --n_data_workers=4 \
        --debug_loading=False \
        --enable_profiler=False \
        --random_offsets_train=True \
        --shuffle_train=True \
        --debug_overfit=False \
        --lr_patience=5 \
        --USE_WANDB=True \
        --wandb_project=lobs5-full-autoreg-tok24 \
        --wandb_entity=kang-oxford 
        # --wandb_entity=kang-oxford 2>&1 | grep -v "sol_gpu_cost_model"
        # --restore='/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/ruby-aardvark-62_98nov1i7' \
        # --restore_step=37
        #--restore='checkpoints/eager-shadow-750_af39bb9u/'
        #5135
        # --curtail_epochs=5135 \
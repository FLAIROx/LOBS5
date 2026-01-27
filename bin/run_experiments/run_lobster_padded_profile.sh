python3 run_train.py --USE_WANDB=True \
                    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
                    --blocks=16 --bsz=4 --d_model=1024 --dataset=lobster-prediction --merging=padded \
                    --dir_name='/home/myuser/data/processed_data/GOOG/2022' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=400 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=1024 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True  \
                    --masking=none \
                    --num_devices=1 --n_data_workers=2 \
                    --debug_loading=False \
                    --enable_profiler=True \
                    --random_offsets_train=True \
                    --shuffle_train=True \
                    --debug_overfit=False \
                    --lr_patience=4 \
                    # --restore='/home/myuser/data/checkpoints/s5_new/twilight-sound-77_s42sujip/' \
                    # --restore_step=23
                    #--restore='checkpoints/eager-shadow-750_af39bb9u/'
                    #5135
                    # --curtail_epochs=5135 \



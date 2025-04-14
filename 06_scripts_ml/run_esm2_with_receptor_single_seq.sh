CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=12348 main_train.py \
    --batch_size 32 \
    --warmup_epochs 1 \
    --epochs 500 \
    --save_period 5 \
    --eval_period 5 \
    --data_dir ../datasets/stratify \
    --wandb_group krasileva \
    --model esm2_with_receptor_single_seq \
    # --disable_wandb \
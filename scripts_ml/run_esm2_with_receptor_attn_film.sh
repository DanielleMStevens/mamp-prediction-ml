CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=13346 main_train.py \
    --batch_size 32 \
    --warmup_epochs 1 \
    --epochs 200 \
    --save_period 5 \
    --eval_period 5 \
    --data_dir ../datasets/mmseqs50 \
    --wandb_group krasileva \
    --model esm2_with_receptor_attn_film \
    # --disable_wandb \
    # --model_checkpoint_path out/checkpoint-129.pth \

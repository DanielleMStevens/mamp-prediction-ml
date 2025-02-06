torchrun --nproc_per_node=1 --master_port=12346 main_train.py \
    --batch_size 32 \
    --warmup_epochs 10 \
    --epochs 200 \
    --save_period 5 \
    --eval_period 5 \
    --data_dir ../datasets/mmseqs50 \
    --wandb_group krasileva \
    --model amplify_with_receptor \
    # --disable_wandb
    # --model_checkpoint_path out/checkpoint-129.pth \
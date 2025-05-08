python 06_scripts_ml/06_main_train.py --model esm2_with_receptor --data_dir 05_datasets --device cpu --epochs 20 --save_period 10 --model_checkpoint_path ../model_results/esm2_with_receptor_05_datasets

# datasets: *_random.csv

#CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=10346 main_train.py \
#   --batch_size 32 \
#    --warmup_epochs 1 \
#    --epochs 50 \
#    --save_period 5 \
#    --eval_period 5 \
#    --data_dir ../datasets/stratify \
#    --wandb_group krasileva \
#    --model esm2_with_receptor \
#    --disable_wandb \
# --model_checkpoint_path out/checkpoint-129.pth \


python 06_scripts_ml/06_main_train.py --model esm2_with_receptor --data_dir 05_datasets/stratify/ --disable_wandb --device cpu --epochs 60
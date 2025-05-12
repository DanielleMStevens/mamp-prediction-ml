# ------- Run 1: ESM2 with Receptor Model - 05_datasets/*_random.csv -------
# Saved As: 04_random_esm2_with_receptor_05_datasets

python 06_scripts_ml/06_main_train.py \
    --model esm2_with_receptor \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path ../model_results/esm2_with_receptor_05_datasets/test_preds.pth \
    --output_dir ../model_results/esm2_with_receptor_05_datasets \
    --data_info_path 05_datasets/test_random.csv

# ------- Run 2: ESM2 with Receptor Model - 05_datasets/*_immuno_stratify.csv -------
# Saved As: 05_immuno_stratify_esm2_with_receptor_05_datasets

python 06_scripts_ml/06_main_train.py \
    --model esm2_with_receptor \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path ../model_results/esm2_with_receptor_05_datasets/test_preds.pth \
    --output_dir ../model_results/esm2_with_receptor_05_datasets \
    --data_info_path 05_datasets/test_immuno_stratify.csv


# ------- Run 3: ESM2 with All Chemical Features Model - 05_datasets/*_data_with_all_train_random.csv -------
# Saved As: 06_esm2_all_chemical_features_05_datasets

python 06_scripts_ml/06_main_train.py \
    --model esm2_all_chemical_features \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

# ------- Run 4: ESM2 with All Chemical Features Model - 05_datasets/*_data_with_all_train_immuno_stratify.csv -------
# Saved As: 07_esm2_all_chemical_features_05_datasets
python 06_scripts_ml/06_main_train.py \
    --model esm2_all_chemical_features \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path ../model_results/07_esm2_all_chemical_features_05_datasets/test_preds.pth \
    --output_dir ../model_results/07_esm2_all_chemical_features_05_datasets \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv


# ------- Run 5: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_train_random.csv -------
# Saved As: 08_esm2_bfactor_weighted_05_datasets

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 



# ------- Run 6: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# Saved As: 09_esm2_bfactor_weighted_05_datasets

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 5 


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

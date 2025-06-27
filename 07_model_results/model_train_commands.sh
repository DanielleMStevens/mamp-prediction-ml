# ------- Run 1: ESM2 with Receptor Model - 05_datasets/*_random.csv -------
# Saved As: 01_random_esm2_with_receptor

python 06_scripts_ml/06_main_train.py \
    --model esm2_with_receptor \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/01_random_esm2_with_receptor/test_preds.pth \
    --output_dir 07_model_results/01_random_esm2_with_receptor \
    --data_info_path 05_datasets/test_random.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/01_random_esm2_with_receptor/correct_classification_report.tsv \
    07_model_results/01_random_esm2_with_receptor/misclassification_report.tsv

# ------- Run 2: ESM2 with Receptor Model - 05_datasets/*_immuno_stratify.csv -------
# Saved As: 02_immuno_stratify_esm2_with_receptor

python 06_scripts_ml/06_main_train.py \
    --model esm2_with_receptor \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/02_immuno_stratify_esm2_with_receptor/test_preds.pth \
    --output_dir 07_model_results/02_immuno_stratify_esm2_with_receptor \
    --data_info_path 05_datasets/test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/02_immuno_stratify_esm2_with_receptor/correct_classification_report.tsv \
    07_model_results/02_immuno_stratify_esm2_with_receptor/misclassification_report.tsv



# ------- Run 3: ESM2 with All Chemical Features Model - 05_datasets/*_data_with_all_train_random.csv -------
# Saved As: 03_random_esm2_all_chemical_features

python 06_scripts_ml/06_main_train.py \
    --model esm2_all_chemical_features \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/06_random_esm2_all_chemical_features/test_preds.pth \
    --output_dir 07_model_results/06_random_esm2_all_chemical_features \
    --data_info_path 05_datasets/test_data_with_all_test_random.csv


Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/03_random_esm2_all_chemical_features/correct_classification_report.tsv \
    07_model_results/03_random_esm2_all_chemical_features/misclassification_report.tsv


# ------- Run 4: ESM2 with All Chemical Features Model - 05_datasets/*_data_with_all_train_immuno_stratify.csv -------
# Saved As: 04_immuno_stratify_esm2_all_chemical_features

python 06_scripts_ml/06_main_train.py \
    --model esm2_all_chemical_features \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/07_immuno_stratify_esm2_all_chemical_features/test_preds.pth \
    --output_dir 07_model_results/07_immuno_stratify_esm2_all_chemical_features \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/04_immuno_stratify_esm2_all_chemical_features/correct_classification_report.tsv \
    07_model_results/04_immuno_stratify_esm2_all_chemical_features/misclassification_report.tsv

# ------- Run 5: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_train_random.csv -------
# Saved As: 05_random_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/05_random_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/05_random_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_random.csv


Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/05_random_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/05_random_esm2_bfactor_weighted/misclassification_report.tsv

# ------- Run 6: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# Saved As: 06_immuno_stratify_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 5 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/06_immuno_stratify_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/06_immuno_stratify_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/06_immuno_stratify_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/06_immuno_stratify_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 7: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze last layer only
# Saved As: 07_esm2_t6_8M_UR50D_last_layer_only_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/07_esm2_t6_8M_UR50D_last_layer_only_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/07_esm2_t6_8M_UR50D_last_layer_only_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/07_esm2_t6_8M_UR50D_last_layer_only_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/07_esm2_t6_8M_UR50D_last_layer_only_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 8: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# ---------- tesing model size: esm2_t12_35M_UR50D, unfreeze last layer only
# Saved As: 08_esm2_t12_35M_UR50D_last_layer_only_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/08_esm2_t12_35M_UR50D_last_layer_only_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/08_esm2_t12_35M_UR50D_last_layer_only_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/08_esm2_t12_35M_UR50D_last_layer_only_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/08_esm2_t12_35M_UR50D_last_layer_only_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 9: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# ---------- tesing model size: esm2_t30_150M_UR50D, unfreeze last layer only
# Saved As: 09_esm2_t30_150M_UR50D_last_layer_only_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/09_esm2_t30_150M_UR50D_last_layer_only_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/09_esm2_t30_150M_UR50D_last_layer_only_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/09_esm2_t30_150M_UR50D_last_layer_only_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/09_esm2_t30_150M_UR50D_last_layer_only_esm2_bfactor_weighted/misclassification_report.tsv

# ------- Run 10: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# ---------- tesing model size: esm2_t33_650M_UR50D, unfreeze last layer only
# Saved As: 10_esm2_t33_650M_UR50D_last_layer_only_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/10_esm2_t33_650M_UR50D_last_layer_only_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/10_esm2_t33_650M_UR50D_last_layer_only_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/10_esm2_t33_650M_UR50D_last_layer_only_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/10_esm2_t33_650M_UR50D_last_layer_only_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 11: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 2 layers 
# Saved As: 11_esm2_t6_8M_UR50D_two_layers_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/11_esm2_t6_8M_UR50D_two_layers_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/11_esm2_t6_8M_UR50D_two_layers_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/11_esm2_t6_8M_UR50D_two_layers_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/11_esm2_t6_8M_UR50D_two_layers_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 12: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 3 layers 
# Saved As: 12_esm2_t6_8M_UR50D_three_layers_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --epochs 20 \
    --save_period 10 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/12_esm2_t6_8M_UR50D_three_layers_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/12_esm2_t6_8M_UR50D_three_layers_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/12_esm2_t6_8M_UR50D_three_layers_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/12_esm2_t6_8M_UR50D_three_layers_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 13: ESM2 with Position Weighted Model - 05_datasets/train_data_with_synthetic_negatives.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 12, use synthetic negatives (0.1 ratio)
# Saved As: 13_esm2_t6_8M_UR50D_syn_data_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --batch_size 12 \
    --epochs 20 \
    --save_period 10 

python 07_model_results/01_make_confusion_matrix.py \
    --predictions_path 07_model_results/13_esm2_t6_8M_UR50D_syn_data_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/13_esm2_t6_8M_UR50D_syn_data_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/13_esm2_t6_8M_UR50D_syn_data_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/13_esm2_t6_8M_UR50D_syn_data_esm2_bfactor_weighted/misclassification_report.tsv

# ------- Run 14: ESM2 with Position Weighted Model - 05_datasets/train_data_with_synthetic_negatives_enhanced.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 12, use synthetic negatives enhanced (1.0 ratio)
# Saved As: 14_esm2_t6_8M_UR50D_syn_data_enhanced_esm2_bfactor_weighted


python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --batch_size 12 \
    --epochs 20 \
    --save_period 10 

python 07_model_results/01_make_confusion_matrix.py \
    --predictions_path 07_model_results/14_esm2_t6_8M_UR50D_syn_data_enhanced_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/14_esm2_t6_8M_UR50D_syn_data_enhanced_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/14_esm2_t6_8M_UR50D_syn_data_enhanced_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/14_esm2_t6_8M_UR50D_syn_data_enhanced_esm2_bfactor_weighted/misclassification_report.tsv

# ------- Run 15: ESM2 with Position Weighted Model - 05_datasets/train_data_with_synthetic_negatives_semienhanced.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 12, use synthetic negatives enhanced (0.5 ratio)
# Saved As: 15_esm2_t6_8M_UR50D_syn_data_semienhanced_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --batch_size 12 \
    --epochs 20 \
    --save_period 10 

python 07_model_results/01_make_confusion_matrix.py \
    --predictions_path 07_model_results/15_esm2_t6_8M_UR50D_syn_data_semienhanced_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/15_esm2_t6_8M_UR50D_syn_data_semienhanced_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/15_esm2_t6_8M_UR50D_syn_data_semienhanced_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/15_esm2_t6_8M_UR50D_syn_data_semienhanced_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 16: ESM2 with Position Weighted Model - 05_datasets/train_data_with_synthetic_negatives_semienhanced.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 16, use synthetic negatives enhanced (0.5 ratio)
# Saved As: 16_esm2_t6_8M_UR50D_semi_syn_data_larger_batch_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --batch_size 16 \
    --epochs 20 \
    --save_period 10


python 07_model_results/01_make_confusion_matrix.py \
    --predictions_path 07_model_results/16_esm2_t6_8M_UR50D_semi_syn_data_larger_batch_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/16_esm2_t6_8M_UR50D_semi_syn_data_larger_batch_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/16_esm2_t6_8M_UR50D_semi_syn_data_larger_batch_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/16_esm2_t6_8M_UR50D_semi_syn_data_larger_batch_esm2_bfactor_weighted/misclassification_report.tsv

# ------- Run 17: ESM2 with Position Weighted Model - 05_datasets/train_data_with_synthetic_negatives.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 12, use synthetic negatives (0.1 ratio)
# ---------- adjust weight class in pytorch as (1.0, 1.0, 2.0) - line 317 of esm_positon_weighted.py
# Saved As: 17_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --batch_size 12 \
    --epochs 20 \
    --save_period 10 

python 07_model_results/01_make_confusion_matrix.py \
    --predictions_path 07_model_results/17_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/17_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/17_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/17_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted/misclassification_report.tsv


# ------- Run 18: ESM2 with Position Weighted Model - 05_datasets/train_data_with_synthetic_negatives.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 12, use synthetic negatives (0.1 ratio)
# ---------- adjust weight class in pytorch as (1.0, 1.0, 3.0) - line 317 of esm_positon_weighted.py
# Saved As: 18_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --batch_size 12 \
    --epochs 20 \
    --save_period 10 

python 07_model_results/01_make_confusion_matrix.py \
    --predictions_path 07_model_results/18_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted/test_preds.pth \
    --output_dir 07_model_results/18_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/18_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted/correct_classification_report.tsv \
    07_model_results/18_esm2_t6_8M_UR50D_syn_data_class_weights_esm2_bfactor_weighted/misclassification_report.tsv








# ------- Runnning Final Mode Parameters: ESM2 with Position Weighted Model - 05_datasets/*_data_with_all_test_immuno_stratify.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 12
# Saved As: 00_mamp_ml

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir 05_datasets \
    --device cpu \
    --batch_size 12 \
    --epochs 20 \
    --save_period 5 

python 07_model_results/02_make_confusion_matrix.py \
    --predictions_path 07_model_results/00_mamp_ml/test_preds.pth \
    --output_dir 07_model_results/00_mamp_ml \
    --data_info_path 05_datasets/test_data_with_all_test_immuno_stratify.csv

Rscript 07_model_results/00_visualize_model_predictions.R \
    07_model_results/00_mamp_ml/correct_classification_report.tsv \
    07_model_results/00_mamp_ml/misclassification_report.tsv


############################################################################
# testing mamp-ml performance on independent test_data_set
############################################################################

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --eval_only_data_path 09_testing_and_dropout/validation_data_set/data_validation_all.csv \
    --model_checkpoint_path 07_model_results/00_mamp_ml_best_params/checkpoint-19.pth \
    --device cpu \
    --disable_wandb


python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --eval_only_data_path 09_testing_and_dropout/validation_data_set/data_validation_all.csv \
    --model_checkpoint_path 07_model_results/02_immuno_stratify_esm2_with_receptor/checkpoint-19.pth \
    --device cpu \
    --disable_wandb

############################################################################
# testing if SeqOnly performs better than mamp-ml for dropout case
############################################################################

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --eval_only_data_path 09_testing_and_dropout/dropout_case/data_validation_all.csv \
    --model_checkpoint_path 07_model_results/00_mamp_ml/checkpoint-19.pth \
    --device cpu \
    --disable_wandb

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --eval_only_data_path 09_testing_and_dropout/dropout_case/data_validation_all.csv \
    --model_checkpoint_path 07_model_results/02_immuno_stratify_esm2_with_receptor/checkpoint-19.pth \
    --device cpu \
    --disable_wandb


# ------- Runnning Final Model: ESM2 with Position Weighted Model - mamp_ml/final_model_training_data.csv -------
# ---------- tesing model size: esm2_t6_8M_UR50D, unfreeze 1 layers, adjust batch size to 12
# Saved As: mamp_ml_final_model

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --data_dir mamp_ml/training_data \
    --device cpu \
    --batch_size 12 \
    --epochs 20 \
    --save_period 5 

python 06_scripts_ml/06_main_train.py \
    --model esm2_bfactor_weighted \
    --eval_only_data_path 09_testing_and_dropout/test_data_set/data_validation_all.csv \
    --model_checkpoint_path mamp_ml/mamp_ml_weights.pth \
    --device cpu \
    --disable_wandb
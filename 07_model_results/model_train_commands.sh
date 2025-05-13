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


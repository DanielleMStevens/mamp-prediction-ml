# Example command to train the masked model

# Define paths (replace with your actual paths)
DATA_DIR="path/to/your/data_directory" # Directory containing train/test CSVs with 'Header_Name'
BFACTOR_CSV="04_Preprocessing_results/bfactor_winding_lrr_segments.csv" # Path to B-factor info
OUTPUT_DIR="path/to/your/output/directory" # Where model checkpoints and logs will be saved

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Run the training script
python 06_scripts_ml/06_main_train_masked.py \
    --model esm2_with_receptor_masked \
    --data_dir "${DATA_DIR}" \
    --bfactor_csv_path "${BFACTOR_CSV}" \
    --epochs 50 \
    --batch_size 8 \
    --lr 3e-4 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --eval_period 5 \
    --save_period 10 \
    --device cpu 
    # --disable_wandb \ # Uncomment this if you don't want to log to Weights & Biases
    # --model_checkpoint_path path/to/load/checkpoint \ # Uncomment to load a specific checkpoint
    # --resume path/to/resume/checkpoint # Uncomment to resume training from a checkpoint
    # --output_dir "${OUTPUT_DIR}" # Output dir is now set automatically in the script, but you could override

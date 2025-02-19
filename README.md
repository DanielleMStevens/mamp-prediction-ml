# Prediction of MAMP-Receptor Interactions through a Protein Language Model

## Overview

This repository contains the code for the paper "Prediction of MAMP-Receptor Interactions through a Protein Language Model" by Danielle Stevens, et al.

## Installation

```


# for lrr annotator - run inside LRR-Annotation folder
pip install -e .   


conda create -n esmfold python=3.9
conda install -c conda-forge openmm=7.7.0
conda install -c conda-forge biopython
conda install -c conda-forge wxpython

conda install https://anaconda.org/nvidia/cuda-nvcc/12.6.85/download/linux-64/cuda-nvcc-12.6.85-0.conda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -c hcc

export CUDA_HOME=/usr/local/cuda
nvcc -V
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


Attempt 2
```
pip3 install torch torchvision torchaudio
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal  # For GPU acceleration
```

# MAMP Training Script Documentation

## Overview
This script (`05_main_train.py`) implements a training pipeline for Membrane-Active Molecule Prediction (MAMP) models. It supports multiple model architectures, distributed training, and extensive evaluation capabilities.

## Key Components

### 1. Model Architecture Support
The script supports multiple model architectures through the `model_dict`:
- ESM2-based models (base, mid-layer features)
- GLM-based models
- AlphaFold-based models
- Receptor interaction models (ESM and GLM variants)
- AMP (Antimicrobial Peptide) models
- Contrastive learning models

### 2. Dataset Classes
Three main dataset types are supported via `dataset_dict`:
- `PeptideSeqDataset`: Basic peptide sequence dataset
- `AlphaFoldDataset`: Dataset for AlphaFold feature processing
- `PeptideSeqWithReceptorDataset`: Dataset that includes receptor information

### 3. Command Line Arguments
The script uses argparse with several parameter groups:

#### Basic Parameters
- `--seed`: Random seed for reproducibility
- `--model`: Model architecture selection (required)
- `--data_dir`: Training data directory
- `--eval_only_data_path`: Path for evaluation-only mode

#### Model Parameters
- `--backbone`: Choice of backbone model (e.g., ESM2, AlphaFold)
- `--head_dim`: Model head dimension
- `--freeze_at`: Layer freezing control
- `--finetune_backbone`: Path to pretrained weights

#### Training Parameters
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--warmup_epochs`: Number of warmup epochs
- `--cross_eval_kfold`: K-fold cross-validation control

#### Loss Parameters
- `--lambda_single`: Single sequence loss weight
- `--lambda_double`: Double sequence loss weight
- `--lambda_pos`: Positive example weighting

#### Evaluation Parameters
- `--eval_period`: Evaluation frequency
- `--save_period`: Checkpoint saving frequency
- `--dist_eval`: Distributed evaluation control

### 4. Training Pipeline
The main training loop includes:

1. **Initialization**
   - Distributed training setup
   - Model and optimizer initialization
   - Dataset and dataloader preparation

2. **Training Process**
   - Epoch-based training with `train_one_epoch`
   - Periodic evaluation on test data
   - Regular model checkpointing
   - Progress tracking via WandB

3. **Evaluation**
   - ROC and PR curve generation
   - Metric computation and logging
   - Support for both single-run and k-fold evaluation

### 5. Cross-Validation Support
The script includes k-fold cross-validation functionality:
- Stratified fold splitting based on EC3 classification
- Independent model training for each fold
- Separate evaluation and metric tracking per fold
- Fold-specific data saving and checkpointing

### 6. Logging and Visualization
Comprehensive logging through WandB:
- Training metrics and loss curves
- Evaluation metrics (AUROC, AUPRC, etc.)
- Model checkpoints
- ROC and PR curve visualizations

## Usage Examples

1. **Training Mode**
```bash
python 05_main_train.py --model esm2 --data_dir path/to/data --epochs 50
```

2. **Evaluation Mode**
```bash
python 05_main_train.py --model esm2 --eval_only_data_path path/to/test.csv
```

3. **Cross-Validation**
```bash
python 05_main_train.py --model esm2 --data_dir path/to/data --cross_eval_kfold 5
```

## Output Structure
The script organizes outputs in a structured manner:
```
model_results/
└── {model_name}_{dataset_name}/
    ├── checkpoints/
    ├── plots/
    │   ├── roc_curves/
    │   └── pr_curves/
    ├── fold_results/  # For cross-validation
    └── metrics.csv
```

## Model Architectures

### ESM2-based Models
- `ESMModel`: Base ESM2 model
- `ESMMidModel`: ESM2 with mid-layer feature extraction
- `ESMWithReceptorModel`: ESM2 with receptor interaction
- `ESMWithReceptorSingleSeqModel`: ESM2 with single sequence receptor
- `ESMWithReceptorAttnFilmModel`: ESM2 with attention and FiLM layers
- `ESMContrastiveModel`: ESM2 with contrastive learning

### GLM-based Models
- `GLMModel`: Base GLM model
- `GLMWithReceptorModel`: GLM with receptor interaction
- `GLMWithReceptorSingleSeqModel`: GLM with single sequence receptor

### Other Models
- `AlphaFoldModel`: Based on AlphaFold architecture
- `AMPModel`: Antimicrobial peptide prediction model
- `AMPWithReceptorModel`: AMP model with receptor interaction

## Dependencies
- PyTorch
- Weights & Biases (wandb)
- NumPy
- Pandas
- scikit-learn
- ESM2 (for ESM-based models)
- AlphaFold (for AF-based models)

## Contributing
When contributing to this project, please:
1. Follow the existing code style
2. Add appropriate documentation for new features
3. Update the README.md with details of significant changes
4. Add tests for new functionality

## License
[Add appropriate license information]
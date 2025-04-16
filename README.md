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


## Prepare dataset for training and testing
```
./run_data_preparation_pipeline.sh

# if there are permission issues, run the following first then rerun the above
chmod -x run_data_preparation_pipeline.sh
```

# Training model
```
python scripts_ml/05_main_train.py --model esm2_with_receptor --data_dir datasets/stratify --disable_wandb --device cpu --epochs 60
```
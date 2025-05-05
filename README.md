# Prediction of MAMP-Receptor Interactions through a Protein Language Model

## Overview

This repository contains the code for the paper "Prediction of MAMP-Receptor Interactions through a Protein Language Model" by Danielle Stevens, et al.

## Installation

```
# installing local colabfold and LRR-Annotation
conda create --name=localfold python=3.11
conda activate localfold
conda install mamba
mamba update --all
mamba install cuda -c nvidia
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
bash install_colabbatch_linux.sh

# if installing on apple silicon, intsead of mamba install cuda -c nvidia
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal  # For GPU support

pip install matplotlib-inline
conda install -c conda-forge wxpython 
pip install -e ./01_LRR_Annotation

# installing esm for model building - yes there is one conda environment for structure building and one for pLM prediction
conda deactivate
bash install_esmfold.env.sh
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


##### old code ignore for now
```
conda install -c conda-forge openmm=7.7.0
conda install -c conda-forge biopython
conda install -c conda-forge wxpython

conda install https://anaconda.org/nvidia/cuda-nvcc/12.6.85/download/linux-64/cuda-nvcc-12.6.85-0.conda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -c hcc

export CUDA_HOME=/usr/local/cuda
nvcc -V
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```
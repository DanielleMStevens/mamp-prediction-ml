# Prediction of MAMP-Receptor Interactions through Protein Language Models

## Overview

This repository contains the code for the paper "Prediction of MAMP-Receptor Interactions through Protein Language Models" by Danielle Stevens, et al.

## Installation

```
conda create -n esmfold python=3.9
s
conda install -c conda-forge openmm=7.7.0
conda install -c conda-forge biopython
conda install -c conda-forge wxpython

# for lrr annotator - run inside LRR-Annotation folder
pip install -e .   

conda install https://anaconda.org/nvidia/cuda-nvcc/12.6.85/download/linux-64/cuda-nvcc-12.6.85-0.conda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -c hcc

export CUDA_HOME=/usr/local/cuda
nvcc -V

conda env update -f environment_esm.yml

# 

# Then install hhsuite with Homebrew or pip
brew install hmmer
brew install brewsci/bio/hhsuite

https://anaconda.org/conda-forge/biopython/1.85/download/win-64/biopython-1.85-py311he736701_1.conda
```

Attempt 2
```
pip3 install torch torchvision torchaudio
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal  # For GPU acceleration
```
# Prediction of MAMP-Receptor Interactions through Protein Language Models

## Overview

This repository contains the code for the paper "Prediction of MAMP-Receptor Interactions through Protein Language Models" by Danielle Stevens, et al.

## Installation

```
conda create -n esmfold python=3.9
conda activate esmfold
conda install -c conda-forge openmm=7.7.0
conda install -c conda-forge biopython

conda install https://anaconda.org/nvidia/cuda-nvcc/12.6.85/download/linux-64/cuda-nvcc-12.6.85-0.conda
conda env update -f environment_esm.yml

# export CUDA_HOME=/usr/local/cuda

# Then install hhsuite with Homebrew or pip
brew install hmmer
brew install brewsci/bio/hhsuite

https://anaconda.org/conda-forge/biopython/1.85/download/win-64/biopython-1.85-py311he736701_1.conda
```


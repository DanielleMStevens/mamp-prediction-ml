# Prediction of MAMP-Receptor Interactions through Protein Language Models

## Overview

This repository contains the code for the paper "Prediction of MAMP-Receptor Interactions through Protein Language Models" by Danielle Stevens, et al.

## Installation

```
conda create -n esmfold python=3.9
conda activate esmfold
conda install -c conda-forge openmm=7.7.0
conda env update -f environment_esm.yml

# Then install hhsuite with Homebrew
brew install hmmer
brew install brewsci/bio/hhsuite
```


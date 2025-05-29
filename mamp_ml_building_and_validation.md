# Prediction of MAMP-Receptor Interactions through a Protein Language Model

## 00. Overview

This repository contains the code for the paper "Prediction of MAMP-Receptor Interactions through a Protein Language Model" by Danielle Stevens, et al.

The code is broken down into the following: Prepping the data for training and validation, model hyperparameter optization and assessement

```
.
# code base for LRR_Annotation
├── 01_LRR_Annotation/
│   ├── analyze_bfactor_peaks.py
│   ├── extract_lrr_sequences.py
├── 02_in_data/
│   ├── All_LRR_PRR_ligand_data.xlsx
├── 03_out_data/
│   ├── lrr_annotation_plots/*.pdf # plots from LRR-Annotation on boundaries for LRR domain
│   ├── modeled_structures/
│   │   ├── pdb_for_lrr_annotator/*.pdb # converted pdb files from AlphaFold models for LRR-Annotation
│   │   ├── pdb_for_lrr_annotator/*_env/ # AlphaFold models output
│   │   ├── alphafold_model_stats.txt # tracked output of alphafold stats
│   ├── training_data_summary/ # ignore for now
│   ├── lrr_annotation_results.txt # summary of LRR-Annotation domain extract from XX script
│   ├── lrr_domain_sequences.fasta # fasta file of just LRR domain from XX script
│   ├── receptor_full_length.fasta # fasta file of full length receptor sequence from 06_scripts_ml/01_prep_receptor_sequences_for_modeling.R
├── 04_Preprocessing_results/
│   ├── Train_plots/

```


## 01. Prepping train + validation datasets

All the data for model training and validation is found in the excel sheet (All_LRR_PRR_ligand_data.xlsx) in 02_in_data. The file path is the following:
```
.
├── 02_in_data/
│   ├── All_LRR_PRR_ligand_data.xlsx
```

This data needs to be transformed in to a fasta file to run though AlphaFold in a semi-automated manner. 
```
# run this script on the command line 
Rscript 06_scripts_ml/01_prep_receptor_sequences_for_modeling.R

# it will generate a fasta file in the following file path:
.
├── 02_in_data/
│   ├── All_LRR_PRR_ligand_data.xlsx
```

Once the fasta file is generated, we will run it though local colabfold to generate predictive structures for each receptor sequence. To do so efficeintly, it is recommnded to run on a GPU (ideally NVIDIA). We initally used our local HPC, so the first few commands are for our HPC GPU's and may need to be modified for your system. This will generate AlphaFold structures (only one) for each unique receptor sequence.

```
# ------------------ each time a new gpu is started ------------------
module load anaconda3
conda activate localfold
module load gcc/10.5.0
export PATH="/global/scratch/users/dmstev/localcolabfold/colabfold-conda/bin:$PATH"

# ------------------ to run the model ------------------
colabfold_batch --num-models 1 ./03_out_data/receptor_full_length.fasta ./03_out_data/modeled_structures/receptor_only/

where: 
--num-models 1 means one model will be generate per sequence
./03_out_data/receptor_full_length.fasta is the input file 
./03_out_data/modeled_structures/receptor_only/ is the output directory path
```

Once all the structures are generated, we will prepare and run them through LRR-Annoation. 
```
# this script will generate some summary stats from AlphaFold models
python 06_scripts_ml/02_alphafold_to_lrr_annotation.py

python 06_scripts_ml/03_parse_lrr_annotations.py

# script will split the data depending on the 
python 06_scripts_ml/04_data_prep_for_training.py
```










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
#!/bin/bash
#SBATCH --job-name=wheat
#SBATCH --account=ac_kvkallow
#SBATCH --partition=savio3_gpu
#SBATCH --qos=v100_gpu3_normal
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mail-user=dmstev@berkeley.edu
#SBATCH --mail-type=ALL

cd /global/scratch/users/dmstev/mamp_prediction_ml/13_monocot_test/

module load anaconda3
conda activate localfold
export PATH="/global/scratch/users/dmstev/localcolabfold/colabfold-conda/bin:$PATH"

mamp-ml predict Wheat_FLS2_search.xlsx --device cuda 

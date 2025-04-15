module load anaconda3
conda activate localfold
module load gcc/10.5.0
export PATH="/global/scratch/users/dmstev/localcolabfold/colabfold-conda/bin:$PATH"

colabfold_batch ./03_out_data/receptor_full_length.fasta ./03_out_data/modeled_structures/receptor_only/

# NCBI Datasets

## remove small proteins (less than 250 in length)
conda install -c bioconda seqkit
seqkit seq -m 250 -g VITVvi_vCabSauv08_v1.fasta > VITVvi_vCabSauv08_v1_filtered.fasta

## download tmhmm 


```
pip3 install pybiolib
srun --pty -A ac_kvkallow -p savio2_1080ti --qos=savio_normal -t 4:00:00 --gres=gpu:1 --job-name=mamp_prediciton_ml --mail-user=dmstev@berkeley.edu --mail-type=ALL bash -I


# Split fasta into chunks of 1000 sequences
seqkit split2 -s 1000 VITVvi_vCabSauv08_v1_filtered.fasta -O split_fastas/

# Create output directory for DeepTMHMM results
mkdir deeptmhmm_results

# Run DeepTMHMM on each chunk
for file in split_fastas/*.fasta; do
  base=$(basename "$file" .fasta)
  biolib run DTU/DeepTMHMM --fasta "$file" --outdir "deeptmhmm_results/${base}"
done

# Combine all results into one file
cat deeptmhmm_results/*/predictions.txt > combined_deeptmhmm_predictions.txt

```

# Search all the protein fasta files for the kinase doamin
download hmm model for kinase domain: https://www.ebi.ac.uk/interpro/entry/pfam/PF00069/logo/
```
conda install -c bioconda hmmer
hmmsearch -A kinase_alignment.stk --tblout kinase_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 PF00069.hmm VITVvi_vCabSauv08_v1_filtered.fasta 

# Convert the output from hmmersearch into a fasta file
esl-reformat fasta kinase_alignment.stk > reformat_kinase_hits.fasta

# run script to extract just 
Rscript 11_grape_analysis/00_parse_kinase_hits.R    
```

```
mamba create --name deepsig
mamba activate deepsig 
mamba install -c conda-forge tensorflow 
mamba install -c bioconda -c anaconda deepsig 
 python --version   
 python3 -m pip install 'tensorflow[and-cuda]'  


 module load ml/tensorflow/2.14.0-py3.9.0
mamba create --name deepsig -c bioconda -c anaconda deepsig  
mamba create --name deepsig 

conda install -c bioconda phobius
mamba create --name deepsig -c bioconda -c anaconda deepsig
 ```
# Analysing grape genome to see if we can use mamp-ml to find other convergently evolved csp22 repceptor

In 2020, it was reported that one variety grape could recognize csp22 by ROS response (see Burbank et al. 2020). We hypotheize due the evolutionary distance between grape and tomato, the receptor that recognizes csp22 ligand is likely convergent evolved. This is compounded by the citrus receptor, SCORE, which was recently discovered (Ngou et al. 2025). Luckily, SCORE appears to share similar chemical properties, particularly in the binding pocket, as the tomato receptor CORE. We think this is why SCORE could be zero-shot predicted with decent accuracy (for immunogenic outcomes). Using this idea, we aim to use mamp-ml to screen grape receptors for good candidates.

After downloading the grape proteome from the Cantu lab, we will first remove small proteins.

```
# remove small proteins (less than 250 in length)
conda install -c bioconda seqkit
seqkit seq -m 250 -g VITVvi_vCabSauv08_v1.fasta > VITVvi_vCabSauv08_v1_filtered.fasta

grep -o '>' VITVvi_vCabSauv08_v1_filtered.fasta | wc -l
57036
```

Next, we want to filter for membrane-associated proteins as surface-localized receptors are membrane-bound. We can use DeepTmHmm to filter for these proteins.
```
# download tmhmm 
pip3 install pybiolib

# Split fasta into chunks of 250 sequences
seqkit split2 -s 250 VITVvi_vCabSauv08_v1_filtered.fasta -O split_fastas/

# Create output directory for DeepTMHMM results
mkdir deeptmhmm_results

# Run DeepTMHMM on each chunk
for file in split_fastas/*.fasta; do
  base=$(basename "$file" .fasta)
  biolib run DTU/DeepTMHMM --fasta "$file" --verbose
  part_num=$(echo $base | grep -o 'part_[0-9]*' | grep -o '[0-9]*')
  mv biolib_results "biolib_results_${part_num}"
done

#biolib run --local 'DTU/DeepTMHMM:1.0.24' --fasta VITVvi_vCabSauv08_v1_filtered.part_115.fasta


# Combine all results into one file
cat deeptmhmm_results/*_predicted_topologies.3line > combined_deeptmhmm_predictions.txt
grep -o '>' combined_deeptmhmm_predictions.txt | wc -l
57036
```
We will then filter out any proteins with the 'glob' tag. These proteins are unlikely to go to the membrane.
```
Rscript 11_grape_analysis/00_parse_deeptmhmm_hits.R 
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

Total entries processed: 57036 
Entries kept (TM/SP/SP+TM): 14215 
Entries excluded (GLOB): 42821 
Output written to: 11_grape_analysis/filtered_tm_sp_proteins.fasta 
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
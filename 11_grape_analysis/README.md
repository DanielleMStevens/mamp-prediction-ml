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

# Combine all results into one file
cat deeptmhmm_results/*_predicted_topologies.3line > combined_deeptmhmm_predictions.txt
grep -o '>' combined_deeptmhmm_predictions.txt | wc -l
57036
```
We will then filter out any proteins with the 'glob' tag. These proteins are unlikely to go to the membrane.
```
Rscript 11_grape_analysis/00_parse_deeptmhmm_hits.R 

Total entries processed: 57036 
Entries kept (TM/SP/SP+TM): 14215 
Entries excluded (GLOB): 42821 
Output written to: 11_grape_analysis/filtered_tm_sp_proteins.fasta 
```

We next are going to use HMMER to filter for LRR and kinase domain hits. The former will allow us to filter for LRR-PRR proteins and the latter will help us seperate RLPs and RLKs. We can download the hmm profiles from pfam (https://www.ebi.ac.uk/interpro/entry/pfam/).

First we will download several pfam models for LRRs (PF07725.hmm, PF12799.hmm, PF13855.hmm, PF18831.hmm, PF01462.hmm, PF07723.hmm, PF08263.hmm, PF13516.hmm, PF18805.hmm, PF18837.hmm). Then download the kinase domain file (PF00069 and PF07714).

```
conda install -c bioconda hmmer

# run for all LRR hmm profiles
hmmsearch -A LRR1_alignment.stk --tblout LRR1_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF07725.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR2_alignment.stk --tblout LRR2_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF12799.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR3_alignment.stk --tblout LRR3_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF13855.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR4_alignment.stk --tblout LRR4_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF18831.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR5_alignment.stk --tblout LRR5_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF01462.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR6_alignment.stk --tblout LRR6_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF07723.hmm filtered_tm_sp_proteins.fasta
hmmsearch -A LRR7_alignment.stk --tblout LRR7_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF08263.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR8_alignment.stk --tblout LRR8_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF13516.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR9_alignment.stk --tblout LRR9_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF18805.hmm filtered_tm_sp_proteins.fasta 
hmmsearch -A LRR10_alignment.stk --tblout LRR10_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF18837.hmm filtered_tm_sp_proteins.fasta 

# Convert the output from hmmersearch into a fasta files
esl-reformat fasta LRR1_alignment.stk > LRR1_alignment.fasta
esl-reformat fasta LRR2_alignment.stk > LRR2_alignment.fasta
esl-reformat fasta LRR3_alignment.stk > LRR3_alignment.fasta
esl-reformat fasta LRR6_alignment.stk > LRR6_alignment.fasta
esl-reformat fasta LRR7_alignment.stk > LRR7_alignment.fasta
esl-reformat fasta LRR8_alignment.stk > LRR8_alignment.fasta

# combine all lrr hits fasta into one file
cat lrr_hits/*_alignment.fasta > combine_lrr_hmmer_hits.fasta

# edit 01_parse_hmmer_hits file paths
# Define file paths
kinase_hits_file <- "11_grape_analysis/combine_lrr_hmmer_hits.fasta"
full_length_file <- "11_grape_analysis/filtered_tm_sp_proteins.fasta"
output_file <- "11_grape_analysis/full_length_lrr_hits.fasta"

# run script to extract just 
Rscript 11_grape_analysis/01_parse_hmmer_hits.R
grep -o '>' full_length_lrr_hits.fasta | wc -l
1089
```
We can then finally seperate our hits for RLPs versus RLKs (have kinase domain). For RLKs, some of the hits will likely be developmental receptors. So we can make a tree and try to filter for primarily receptors that are near FLS2 (XII) clade.

```
hmmsearch -A kinase_alignment.stk --tblout kinase_domains.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF00069.hmm full_length_lrr_hits.fasta 

# Convert the output from hmmersearch into a fasta file
esl-reformat fasta kinase_alignment.stk > reformat_kinase_hits.fasta

# edit 01_parse_hmmer_hits file paths
# Define file paths
lrr_hits_file <- "11_grape_analysis/reformat_kinase_hits.fasta"
full_length_file <- "11_grape_analysis/filtered_tm_sp_proteins.fasta"
output_file <- "11_grape_analysis/full_length_kinase_hits.fasta"

# run script to extract just 
Rscript 11_grape_analysis/01_parse_hmmer_hits.R
grep -o '>' full_length_kinase_hits.fasta | wc -l
690
```

We will then rerun the same script for a subset of receptors and make a quick tree to see the receptor distribution.
```
hmmsearch -A kinase_alignment_references.stk --tblout kinase_domains_references.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 pfam_models/PF00069.hmm receptors_for_tree_building.fasta 
esl-reformat fasta kinase_alignment_references.stk > reformat_kinase_reference_hits.fasta

cat reformat_kinase_reference_hits.fasta reformat_kinase_hits.fasta > all_kinase_hits_for_tree.fasta

mafft --auto all_kinase_hits_for_tree.fasta > all_kinase_hits_for_tree_alignment
FastTree all_kinase_hits_for_tree_alignment > all_kinase_hits.tre
```


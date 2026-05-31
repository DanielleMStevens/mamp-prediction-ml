# Analysing grape genome to see if we can use mamp-ml to find other convergently evolved csp22 repceptor

In 2020, it was reported that one variety grape could recognize csp22 by ROS response (see Burbank et al. 2020). We hypotheize due the evolutionary distance between grape and tomato, the receptor that recognizes csp22 ligand is likely convergent evolved. This is compounded by the citrus receptor, SCORE, which was recently discovered (Ngou et al. 2025). Luckily, SCORE appears to share similar chemical properties, particularly in the binding pocket, as the tomato receptor CORE. We think this is why SCORE could be zero-shot predicted with decent accuracy (for immunogenic outcomes). Using this idea, we aim to use mamp-ml to screen grape receptors for good candidates.

After downloading the grape proteome from the Cantu lab, we will first remove small proteins.

```
# creat conda environment
conda create --name grape_analy
conda activate grape_analy

# remove small proteins (less than 250 in length)
conda install -c bioconda seqkit
seqkit seq -m 250 -g VITVvi_vZin03_v1.fasta > VITVvi_vZin03_v1_filtered.fasta

# check how many proteins are left after filtering
grep -o '>' VITVvi_vZin03_v1_filtered.fasta | wc -l
32005
```

Next, we want to filter for membrane-associated proteins as surface-localized receptors are membrane-bound. We can use DeepTmHmm to filter for these proteins.
```
# download tmhmm 
pip3 install pybiolib

# Split fasta into chunks of 250 sequences
mkdir 02_split_fastas
seqkit split2 -s 250 VITVvi_vZin03_v1_filtered.fasta -O 02_split_fastas/

# Create output directory for DeepTMHMM results
mkdir 03_deeptmhmm_results

# Run DeepTMHMM on each chunk - NOTE: this tends to time out, instead I made an account and ran each one online
for file in 02_split_fastas/*.fasta; do
  base=$(basename "$file" .fasta)
  biolib run DTU/DeepTMHMM --fasta "$file" --verbose
  part_num=$(echo $base | grep -o 'part_[0-9]*' | grep -o '[0-9]*')
  mv biolib_results "biolib_results_${part_num}"
done

# Combine all results into one file
cat 03_deeptmhmm_results/*_predicted_topologies.3line > combined_deeptmhmm_predictions.txt
grep -o '>' combined_deeptmhmm_predictions.txt | wc -l
32005
```
We will then filter out any proteins with the 'glob' tag. These proteins are unlikely to go to the membrane.
```
# run this R script in the main directory
Rscript 11_grape_analysis/Zinfandel_cl_03/00_parse_deeptmhmm_hits.R 

Total entries processed: 32005 
Entries kept (TM/SP/SP+TM): 9052 
Entries excluded (GLOB): 22953 
Output written to: 11_grape_analysis/Zinfandel_cl_03/filtered_tm_sp_proteins.fasta 
```

We next are going to use HMMER to filter for LRR and kinase domain hits. The former will allow us to filter for LRR-PRR proteins and the latter will help us seperate RLPs and RLKs. We can download the hmm profiles from pfam (https://www.ebi.ac.uk/interpro/entry/pfam/).

First we will download several pfam models for LRRs (PF07725.hmm, PF12799.hmm, PF13855.hmm, PF18831.hmm, PF01462.hmm, PF07723.hmm, PF08263.hmm, PF13516.hmm, PF18805.hmm, PF18837.hmm). Then download the kinase domain file (PF00069 and PF07714). These will be stored in the 04_pfam_models directory.

```
# install hmmer
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

# ----------------- summary of hits ---------------
# LRR1 - PF07725 - no hits
# LRR2 - PF12799 - 5129 hits
# LRR3 - PF13855 - 5212 hits
# LRR4 - PF18831 - no hits
# LRR5 - PF01462 - no hits
# LRR6 - PF07723 - 7 hits
# LRR7 - PF08263 - 512 hits
# LRR8 - PF13516 - 340 hits
# LRR9 - PF18805 - no hits
# LRR10 - PF18837 - no hits

# Convert the output from hmmersearch into a fasta files
esl-reformat fasta LRR2_alignment.stk > LRR2_alignment.fasta
esl-reformat fasta LRR3_alignment.stk > LRR3_alignment.fasta
esl-reformat fasta LRR6_alignment.stk > LRR6_alignment.fasta
esl-reformat fasta LRR7_alignment.stk > LRR7_alignment.fasta
esl-reformat fasta LRR8_alignment.stk > LRR8_alignment.fasta

# make folder to store hmmer results - run in Zinfandel_cl_03 folder
mkdir 05_lrr_hits

# move files and combine all lrr hits fasta into one file
mv *_alignment.stk *_domains.txt *_alignment.fasta ./05_lrr_hits

# combine the hits into a single file
cat 05_lrr_hits/*_alignment.fasta > combine_lrr_hmmer_hits.fasta

# edit 01_parse_hmmer_hits file paths
# Define file paths
lrr_hits_file <- "11_grape_analysis/Zinfandel_cl_03/combine_lrr_hmmer_hits.fasta"
full_length_file <- "11_grape_analysis/Zinfandel_cl_03/filtered_tm_sp_proteins.fasta"
output_file <- "11_grape_analysis/Zinfandel_cl_03/full_length_lrr_hits.fasta"

# run script to extract just LRR containing proteins with proper LRR domain hits - run in main mamp_prediction_ml folder
Rscript 11_grape_analysis/Zinfandel_cl_03/01_parse_hmmer_hits.R

# fun in Zinfandel_cl_03 folder
grep -o '>' full_length_lrr_hits.fasta | wc -l
824
```
We can then finally seperate our hits for RLPs versus RLKs (have kinase domain). For RLKs, some of the hits will likely be developmental receptors. So we can make a tree and try to filter for primarily receptors that are near FLS2 (XII) clade. We will store the referene proteins, kinase hits, and inital tree building in the 06_initial_kinase_domain_tree directory. We will collect the following reference proteins:
</br>

| Species |  Accession/Locus Tag |Gene|
| ------- | ---- | --------- |
| Arabidopsis thaliana|AT5G46330|FLS2 |
|Nicotiana benthamiana|Niben101Scf03455g01008|FLS2|
| Vitis riparia|PQ283347|FLS2|
|Quercus variabilis|UTN00789|FLS2|
|Arabidopsis thaliana|At5g20480|EFR|
|Solanum lycopersicum|Solyc04g009640|FLS3|
|Solanum lycopersicum|XP_069151269|CORE|
|Nicotiana benthamiana|Niben101Scf02323g01010|CORE|
|Glycine max|Glyma_08g083300|FLS2|

Manually collect the reference proteins to store (above) in the following protein fasta file: receptors_for_tree_building.fasta

```
# make the 06_initial_kinase_domain_tree directory - run also in Zinfandel_cl_03 folder
mkdir 06_initial_kinase_domain_tree 

# search for kinase domain - run also in Zinfandel_cl_03 folder
hmmsearch -A kinase_domain.stk --tblout kinase_domain.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 04_pfam_models/PF00069.hmm full_length_lrr_hits.fasta 
hmmsearch -A kinase_reference.stk --tblout kinase_reference.txt -E 1 --domE 1 --incE 0.01 --incdomE 0.04 --cpu 8 04_pfam_models/PF00069.hmm 06_initial_kinase_domain_tree/receptors_for_tree_building.fasta 

# Convert the output from hmmersearch into a fasta file
esl-reformat fasta kinase_domain.stk > reformat_kinase_domain.fasta
esl-reformat fasta kinase_reference.stk > reformat_kinase_reference.fasta

# edit 01_parse_hmmer_hits file paths and rerun 
# Script to extract full-length sequences based on kinase domain hits
# extracts corresponding full-length sequences from filtered_tm_sp_proteins.fasta

# Define file paths
lrr_hits_file <- "11_grape_analysis/Zinfandel_cl_03/reformat_kinase_domain.fasta"
full_length_file <- "11_grape_analysis/Zinfandel_cl_03/filtered_tm_sp_proteins.fasta"
output_file <- "11_grape_analysis/Zinfandel_cl_03/full_length_kinase_hits.fasta"

# run script to extract - run in main mamp_prediction_ml folder 
Rscript 11_grape_analysis/Zinfandel_cl_03/01_parse_hmmer_hits.R

# run in Zinfandel_cl_03 folder
grep -o '>' full_length_kinase_hits.fasta | wc -l
435
```

We will then rerun the same script for a subset of receptors and make a quick tree to see the receptor distribution.
```
# make a new fasta file with both kinase hits and reference proteins
cat reformat_kinase_domain.fasta reformat_kinase_reference.fasta > all_kinase_hits_for_tree.fasta

# move all the hits to 06_initial_kinase_domain_tree
mv kinase_domain.stk kinase_domain.txt reformat_kinase_domain.fasta ./06_initial_kinase_domain_tree
mv kinase_reference.stk kinase_reference.txt reformat_kinase_reference.fasta ./06_initial_kinase_domain_tree

# alights and build tree
mafft --auto all_kinase_hits_for_tree.fasta > all_kinase_hits_for_tree_alignment
FastTree all_kinase_hits_for_tree_alignment > all_kinase_hits.tre
```

We can quickly visualize the tree in itol and see which clade has most of the likely defense-associated receptors based on the evolution of the kinase domain. We will then extract those receptors. Using itol, we can select those receptor sequences that fall within the defense-oriented clade. Once we have those receptor sequences, we will check for duplicate gebes are remove those that are duplicates.

```
# run in the 11_grape_analysis folder
Rscript 02_extract_defense_RLKs.R 

Total lines in clade file: 68 
Unique genes in clade file: 65 
Duplicates removed: 3 
Genes found in FASTA: 65 
Genes NOT found in FASTA: 0 
```

We then ran them through AlphaFold and LRR-Annotation to assess their LRR number count. Both CORE and SCORE have 20 LRRs, so we would expect a grape version should be around that number (give or so +/- 5).
```
# each time a new gpu is started
module load anaconda3
conda activate localfold
module load gcc/10.5.0
export PATH="/global/scratch/users/dmstev/localcolabfold/colabfold-conda/bin:$PATH"

colabfold_batch --num-models 1 ./11_grape_analysis/vitus_RLK_defense_clade_sequences.fasta ./11_grape_analysis/07_receptor_only/

python 03_alphafold_to_lrr_annotation_grape.py structure_scores.txt

python 01_LRR_Annotation/analyze_bfactor_peaks.py

# count number of proteins with LRR.Repeat.Number > 13 & < 29
63
```

We will then manually remove receptors to test from bfactor_grape_rlks_sum.txt results based on the criteria above as well as the plot from lrr-annotation (lrr_annotation_plots) for any abnormalities. Once we have a final list, we will run through mamp-ml on google colab.

Remove the following due to too few LRRs (>13): g160060, g427230, g086680 (t1,5,6,7,9), g232640 (t1,2,3,4,5,6), g339150 (t1,2,3,4), g386150 (t1,2), g030310 (t1,2), g190190 (t1,4,6), g434250, g466110 (t1,2), g470860 (t1,4,6,7), g537100 (t1,2,3,4,5,6), g027160, g151710, g256070 (t1,2), g279760, g416580, g457140, g617750 (t1,2), g124480 (t1,2), g161440, g244220 (t1,3), g268690, g504970, g505260, g577020 (t2,3), g059220, g059880 (onlt t2), g163900, g379190, g114950, g370700, g151550, g153310, g352910, g428420, g151670

Too many LRRs (30+): g428390, g096500, g096360, g497870, g425940, g426180, g208780, g497910, g403320, g208870, g151570, g426040, g160380, g160340, g426070, g208880, g159710, g426110, g425970, g208810, g159730

Other odd, unlikely domain structure: g125390, g319920, g152700, g416790